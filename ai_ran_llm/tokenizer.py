"""Domain tokenizer: RAN measurement reports <-> token sequences.

A report is serialised with a fixed layout, e.g. (hist_len=5, 4 neighbours)::

    <bos> <spd> V6 <sinr> Q3 <srv> C9 R-92 R-93 R-93 R-95 R-96
    <nbr> C4 D-5 D-3 D-1 D+2 D+5  <nbr> C10 ...  <ans>

Serving RSRP is absolute; neighbour RSRP is given relative to the serving cell at
the same instant (D tokens), which is the quantity handover decisions hinge on.

and the model answers with a decision plus a short rationale::

    <ho> C4 <why> serving falling neighbor rising gain G+4 <eos>
    <stay> <why> serving stable gain G-3 <eos>

Every numeric value gets its own quantised token (RSRP in 1 dB, SINR in 1 dB,
speed in 10 km/h bins, predicted gain in 1 dB), so the vocabulary stays small
and every token carries radio meaning. `numeric_features` gives numeric tokens a
smooth value encoding so neighbouring values start with similar embeddings.
"""

import numpy as np

from .config import ObsConfig
from .simulator import Observation

SPECIALS = ["<pad>", "<bos>", "<eos>", "<spd>", "<sinr>", "<srv>", "<nbr>",
            "<ans>", "<ho>", "<stay>", "<why>"]
WORDS = ["serving", "neighbor", "rising", "falling", "stable", "gain", "low_sinr"]
MAX_CELLS = 64
RSRP_RANGE = (-140, -40)
SINR_RANGE = (-20, 40)
SPEED_BINS = 13          # 0-9, 10-19, ... 120+ km/h
GAIN_RANGE = (-15, 15)
DELTA_RANGE = (-30, 30)
# Location context families (only in the vocabulary when ObsConfig.uses_context)
RATIO_RANGE = (-10, 10)      # L: 10 * log10(d_serving / d_neighbour)
RADIAL_RANGE = (-8, 8)       # S: radial speed towards (-) / away from (+) a cell, 4 m/s bins
RADIAL_STEP_MS = 4.0
MAPGAIN_RANGE = (-20, 20)    # M: radio-map RSRP gain of a neighbour over serving, dB
PROB_RANGE = (0, 10)         # P: next-cell probability in 10 % steps
COUNT_RANGE = (0, 6)         # N: confidence of route statistics, log2 bins of the history count
TREND_DB = 1.5


def _trend_word(delta: float) -> str:
    if delta > TREND_DB:
        return "rising"
    if delta < -TREND_DB:
        return "falling"
    return "stable"


class HandoverTokenizer:
    def __init__(self, obs_cfg: ObsConfig | None = None):
        self.obs_cfg = obs_cfg or ObsConfig()
        tokens = list(SPECIALS) + list(WORDS)
        self.cell_base = len(tokens)
        tokens += [f"C{i}" for i in range(MAX_CELLS)]
        self.rsrp_base = len(tokens)
        tokens += [f"R{v}" for v in range(RSRP_RANGE[0], RSRP_RANGE[1] + 1)]
        self.sinr_base = len(tokens)
        tokens += [f"Q{v}" for v in range(SINR_RANGE[0], SINR_RANGE[1] + 1)]
        self.speed_base = len(tokens)
        tokens += [f"V{i}" for i in range(SPEED_BINS)]
        self.gain_base = len(tokens)
        tokens += [f"G{v:+d}" for v in range(GAIN_RANGE[0], GAIN_RANGE[1] + 1)]
        self.delta_base = len(tokens)
        tokens += [f"D{v:+d}" for v in range(DELTA_RANGE[0], DELTA_RANGE[1] + 1)]
        if self.obs_cfg.uses_context:            # appended: base token ids never move
            self.UNK = len(tokens)
            tokens.append("<unk>")
            self.ratio_base = len(tokens)
            tokens += [f"L{v:+d}" for v in range(RATIO_RANGE[0], RATIO_RANGE[1] + 1)]
            self.radial_base = len(tokens)
            tokens += [f"S{v:+d}" for v in range(RADIAL_RANGE[0], RADIAL_RANGE[1] + 1)]
            self.mapgain_base = len(tokens)
            tokens += [f"M{v:+d}" for v in range(MAPGAIN_RANGE[0], MAPGAIN_RANGE[1] + 1)]
            self.prob_base = len(tokens)
            tokens += [f"P{v}" for v in range(PROB_RANGE[0], PROB_RANGE[1] + 1)]
            self.count_base = len(tokens)
            tokens += [f"N{v}" for v in range(COUNT_RANGE[0], COUNT_RANGE[1] + 1)]
        self.itos = tokens
        self.stoi = {s: i for i, s in enumerate(tokens)}
        for s in SPECIALS + WORDS:
            setattr(self, s.strip("<>").upper() if s.startswith("<") else "W_" + s.upper(), self.stoi[s])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def serving_extra(self) -> int:
        """Context tokens after the serving block: radial speed (position), route confidence (trajectory)."""
        return int(self.obs_cfg.use_position) + int(self.obs_cfg.use_trajectory)

    @property
    def neighbour_extra(self) -> int:
        """Context tokens per neighbour: L S (position), M M (radio map), P (trajectory)."""
        o = self.obs_cfg
        return 2 * int(o.use_position) + 2 * int(o.use_radio_map) + int(o.use_trajectory)

    @property
    def prompt_len(self) -> int:
        h, k = self.obs_cfg.hist_len, self.obs_cfg.n_neighbors
        return 7 + h + self.serving_extra + k * (2 + h + self.neighbour_extra) + 1

    # ---- scalar -> token id -------------------------------------------------
    def cell(self, c):
        return self.cell_base + np.asarray(c, dtype=np.int64)

    def rsrp(self, v):
        v = np.clip(np.rint(v), *RSRP_RANGE).astype(np.int64)
        return self.rsrp_base + v - RSRP_RANGE[0]

    def sinr(self, v):
        v = np.clip(np.rint(v), *SINR_RANGE).astype(np.int64)
        return self.sinr_base + v - SINR_RANGE[0]

    def speed(self, kmh):
        return self.speed_base + np.clip(np.asarray(kmh) // 10, 0, SPEED_BINS - 1).astype(np.int64)

    def gain(self, v):
        v = np.clip(np.rint(v), *GAIN_RANGE).astype(np.int64)
        return self.gain_base + v - GAIN_RANGE[0]

    def delta(self, v):
        v = np.clip(np.rint(v), *DELTA_RANGE).astype(np.int64)
        return self.delta_base + v - DELTA_RANGE[0]

    def _ctx(self, v, base: int, rng: tuple, scale: float = 1.0):
        """Quantise a context value; NaN (unknown) -> <unk>."""
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v)
        q = np.clip(np.rint(np.where(ok, v, 0.0) * scale), *rng).astype(np.int64)
        return np.where(ok, base + q - rng[0], self.UNK)

    def ratio(self, v):
        return self._ctx(v, self.ratio_base, RATIO_RANGE, 10.0)

    def radial(self, v_ms):
        return self._ctx(v_ms, self.radial_base, RADIAL_RANGE, 1.0 / RADIAL_STEP_MS)

    def mapgain(self, v_db):
        return self._ctx(v_db, self.mapgain_base, MAPGAIN_RANGE)

    def prob(self, p):
        return self._ctx(p, self.prob_base, PROB_RANGE, 10.0)

    def count(self, n):
        n = np.asarray(n, dtype=float)
        return self._ctx(np.where(np.isfinite(n), np.floor(np.log2(np.maximum(n, 0) + 1)), np.nan),
                         self.count_base, COUNT_RANGE)

    def numeric_features(self, dim: int) -> np.ndarray:
        """(vocab, dim) sinusoidal encoding of each numeric token's value (0 elsewhere)."""
        feats = np.zeros((self.vocab_size, dim))
        freqs = 1.0 / (100.0 ** (np.arange(dim // 2) / max(dim // 2 - 1, 1)))
        families = [(self.rsrp_base, RSRP_RANGE, 1.0), (self.sinr_base, SINR_RANGE, 1.0),
                    (self.speed_base, (0, SPEED_BINS - 1), 5.0), (self.gain_base, GAIN_RANGE, 1.0),
                    (self.delta_base, DELTA_RANGE, 1.0)]
        if self.obs_cfg.uses_context:
            families += [(self.ratio_base, RATIO_RANGE, 1.0), (self.radial_base, RADIAL_RANGE, 1.0),
                         (self.mapgain_base, MAPGAIN_RANGE, 1.0), (self.prob_base, PROB_RANGE, 1.0),
                         (self.count_base, COUNT_RANGE, 1.0)]
        for base, (lo, hi), scale in families:
            v = np.arange(lo, hi + 1)[:, None] * scale * freqs[None]
            feats[base:base + hi - lo + 1, 0:2 * len(freqs):2] = np.sin(v)
            feats[base:base + hi - lo + 1, 1:2 * len(freqs):2] = np.cos(v)
        return feats

    def is_cell(self, tok: int) -> bool:
        return self.cell_base <= tok < self.cell_base + MAX_CELLS

    # ---- reports ------------------------------------------------------------
    def encode_prompts(self, obs: Observation) -> np.ndarray:
        """Vectorised: Observation of U UEs -> (U, prompt_len) int64 array.

        With location context enabled, context tokens follow the serving block and every
        neighbour block; values missing from ``obs.context`` become ``<unk>``.
        """
        U = len(obs.serving)
        K = obs.nbr_ids.shape[1]
        o = self.obs_cfg
        col = lambda x: np.full((U, 1), x, dtype=np.int64)
        ctx = obs.context or {}
        get = lambda key, shape: np.asarray(ctx[key], dtype=float) if key in ctx else np.full(shape, np.nan)
        parts = [col(self.BOS), col(self.SPD), self.speed(obs.speed_kmh)[:, None],
                 col(self.SINR), self.sinr(obs.sinr_db)[:, None],
                 col(self.SRV), self.cell(obs.serving)[:, None], self.rsrp(obs.serving_hist)]
        if o.uses_context:
            radial = get("radial_speed", (U, K + 1))
            if o.use_position:
                parts.append(self.radial(radial[:, 0])[:, None])
            if o.use_trajectory:
                parts.append(self.count(get("next_count", (U,)))[:, None])
            ratio, now, ahead, prob = (get(k, (U, K)) for k in ("dist_ratio", "map_gain_now", "map_gain_ahead",
                                                                "next_prob"))
        for k in range(K):
            parts += [col(self.NBR), self.cell(obs.nbr_ids[:, k])[:, None],
                      self.delta(obs.nbr_hist[:, k] - obs.serving_hist)]
            if o.use_position:
                parts += [self.ratio(ratio[:, k])[:, None], self.radial(radial[:, k + 1])[:, None]]
            if o.use_radio_map:
                parts += [self.mapgain(now[:, k])[:, None], self.mapgain(ahead[:, k])[:, None]]
            if o.use_trajectory:
                parts.append(self.prob(prob[:, k])[:, None])
        parts.append(col(self.ANS))
        out = np.concatenate(parts, axis=1)
        assert out.shape[1] == self.prompt_len
        return out

    def encode_answer(self, target: int, gain: float, serving_hist, nbr_ids, nbr_hist, sinr_db) -> list[int]:
        words = ["serving", _trend_word(serving_hist[-1] - serving_hist[0])]
        if target >= 0:
            k = int(np.where(nbr_ids == target)[0][0])
            words += ["neighbor", _trend_word(nbr_hist[k][-1] - nbr_hist[k][0])]
            head = [self.HO, int(self.cell(target))]
        else:
            head = [self.STAY]
        ids = head + [self.WHY] + [self.stoi[w] for w in words] + [self.W_GAIN, int(self.gain(gain))]
        if sinr_db < 0:
            ids.append(self.W_LOW_SINR)
        return ids + [self.EOS]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids if int(i) != self.PAD)

    def explain(self, answer_ids) -> dict:
        """Turn generated answer tokens into a structured decision + sentence."""
        toks = [self.itos[int(i)] for i in answer_ids]
        action = "HANDOVER" if toks and toks[0] == "<ho>" else "STAY"
        target = int(toks[1][1:]) if action == "HANDOVER" and len(toks) > 1 and toks[1].startswith("C") else None
        why = toks[toks.index("<why>") + 1:] if "<why>" in toks else []
        why = [w for w in why if w != "<eos>"]
        phrases, i = [], 0
        while i < len(why):
            w = why[i]
            if w in ("serving", "neighbor") and i + 1 < len(why):
                phrases.append(f"{w} cell {why[i + 1]}")
                i += 2
            elif w == "gain" and i + 1 < len(why) and why[i + 1].startswith("G"):
                phrases.append(f"predicted RSRP gain {why[i + 1][1:]} dB over the next second")
                i += 2
            elif w == "low_sinr":
                phrases.append("serving SINR is low")
                i += 1
            else:
                i += 1
        head = f"Hand over to cell {target}" if action == "HANDOVER" else "Stay on serving cell"
        return {"action": action, "target_cell": target,
                "rationale": head + (": " + ", ".join(phrases) if phrases else "") + "."}
