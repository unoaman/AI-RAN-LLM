"""Inference: grammar-constrained decoding, closed-loop policy and xApp-style API."""

from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelConfig, ObsConfig
from .model import HandoverGPT
from .simulator import Episode, Observation
from .tokenizer import GAIN_RANGE, WORDS, HandoverTokenizer


class HandoverLLM:
    """Wraps a trained HandoverGPT checkpoint.

    Decoding is constrained so the model can only emit a well-formed answer:
    ``<stay>`` or ``<ho>`` followed by one of the *reported* neighbour cells, then
    a rationale drawn from the explanation vocabulary. It can never command a
    handover to a cell the UE did not measure.
    """

    def __init__(self, model: HandoverGPT, tok: HandoverTokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tok = tok
        self.device = device
        why_ids = [tok.stoi[w] for w in WORDS] + list(range(tok.gain_base, tok.gain_base + GAIN_RANGE[1] - GAIN_RANGE[0] + 1))
        self._why_ids = torch.tensor(why_ids + [tok.EOS])

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "HandoverLLM":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = HandoverGPT(ModelConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, HandoverTokenizer(ObsConfig(**ckpt["obs_cfg"])), device)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, prompts: np.ndarray):
        """One forward pass per batch. Returns (p_stay (U,), p_ho_cell (U, K)) where
        column k is P(<ho> then the k-th reported neighbour)."""
        tok = self.tok
        P = prompts.shape[1]
        x = torch.from_numpy(np.concatenate([prompts, np.full((len(prompts), 1), tok.HO)], axis=1)).to(self.device)
        logits, _ = self.model(x)
        act = torch.softmax(logits[:, P - 1, [tok.STAY, tok.HO]], dim=-1)
        nbr_cols = torch.from_numpy(np.nonzero(prompts[0] == tok.NBR)[0] + 1)  # neighbour cell-id positions
        nbr_tok = x[:, nbr_cols]                                                # (U, K)
        cell_logits = torch.gather(logits[:, P], 1, nbr_tok)                   # restricted to reported cells
        # a cell listed twice (padding of short reports) must not split its probability
        dup = torch.zeros_like(nbr_tok, dtype=torch.bool)
        for j in range(1, nbr_tok.shape[1]):
            dup[:, j] = (nbr_tok[:, :j] == nbr_tok[:, j:j + 1]).any(dim=1)
        cell_logits = cell_logits.masked_fill(dup, float("-inf"))
        p_cell = torch.softmax(cell_logits, dim=-1)
        return act[:, 0].cpu().numpy(), (act[:, 1:2] * p_cell).cpu().numpy()

    def decide_batch(self, obs: Observation, ho_threshold=0.5):
        """Hand over to the most likely neighbour when P(<ho> to it) >= `ho_threshold`.

        Handovers are rare, so a threshold below 0.5 trades a few extra handovers
        for earlier ones. `ho_threshold` is a scalar or a (U, K) array with one threshold
        per reported neighbour (see :class:`ReturnGuard`); the most likely neighbour
        that clears its own threshold wins. Returns (target (U,) with -1 = stay, confidence (U,)).
        """
        p_stay, p_ho = self.score(self.tok.encode_prompts(obs))
        ok = p_ho >= np.asarray(ho_threshold)
        k = np.where(ok, p_ho, -1.0).argmax(axis=1)
        rows = np.arange(len(k))
        do_ho = ok[rows, k]
        target = np.where(do_ho, obs.nbr_ids[rows, k], -1)
        return target, np.where(do_ho, p_ho[rows, k], p_stay)

    # --------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompt: np.ndarray, max_new: int = 12, prefix: list[int] | None = None) -> list[int]:
        """Greedy grammar-constrained generation of the full answer for one report.

        `prefix` forces the start of the answer (e.g. the decision) so the model
        only writes the rationale for it.
        """
        tok = self.tok
        nbr_cells = torch.from_numpy(prompt[np.nonzero(prompt == tok.NBR)[0] + 1])
        out: list[int] = list(prefix or [])
        seq = torch.from_numpy(np.concatenate([prompt, np.asarray(out, dtype=prompt.dtype)])).to(self.device)[None]
        while len(out) < max_new:
            if not out:
                allowed = torch.tensor([tok.STAY, tok.HO])
            elif out == [tok.HO]:
                allowed = nbr_cells
            elif out == [tok.STAY] or (len(out) == 2 and out[0] == tok.HO):
                allowed = torch.tensor([tok.WHY])
            else:
                allowed = self._why_ids
            logits, _ = self.model(seq[:, -self.model.cfg.block_size:])
            nxt = int(allowed[logits[0, -1, allowed.to(self.device)].argmax()])
            out.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], device=self.device)], dim=1)
            if nxt == tok.EOS:
                break
        return out

    # -------------------------------------------------------------- report API
    def handle_report(self, report: dict, ho_threshold: float = 0.35, min_confidence: float = 0.3,
                      a3_hyst_db: float = 3.0, a3_ttt: int = 3) -> dict:
        """xApp entry point for one JSON measurement report.

        report = {"serving_cell": 3, "speed_kmh": 60, "sinr_db": 4.5,
                  "serving_rsrp": [5 dBm values, oldest first],
                  "neighbors": [{"cell_id": 7, "rsrp": [5 values]}, ...]}

        The decision uses the same rule as the closed-loop policy: hand over to the
        most likely neighbour when P(handover to it) >= `ho_threshold`. The model
        then writes the rationale for that decision. If the chosen action's
        probability is below `min_confidence`, a conservative A3 rule is applied to
        the same report (safety fallback), and the answer says so.
        """
        return self.decide_reports([report], ho_threshold, min_confidence, a3_hyst_db, a3_ttt, explain=True)[0]

    def decide_reports(self, reports: list[dict], ho_threshold: float = 0.35, min_confidence: float = 0.3,
                       a3_hyst_db: float = 3.0, a3_ttt: int = 3, explain: bool = True) -> list[dict]:
        """Batched `handle_report`: one forward pass scores every report.

        With ``explain=False`` no rationale is generated (≈10x faster); the
        ``rationale`` field then only states the decision. The RAN integration
        (`ai_ran_llm.ran.controller`) uses this to decide for many UEs at once.
        """
        if not reports:
            return []
        obs = stack_observations([report_to_observation(r, self.tok.obs_cfg) for r in reports])
        prompts = self.tok.encode_prompts(obs)
        p_stay, p_ho = self.score(prompts)
        answers = []
        for i in range(len(reports)):
            k = int(p_ho[i].argmax())
            if p_ho[i, k] >= ho_threshold:
                prefix, conf = [self.tok.HO, int(self.tok.cell(obs.nbr_ids[i, k]))], float(p_ho[i, k])
            else:
                prefix, conf = [self.tok.STAY], float(p_stay[i])
            ids = self.generate(prompts[i], prefix=prefix) if explain else prefix
            answer = self.explain_answer(ids)
            answer.update(confidence=round(conf, 4), p_stay=round(float(p_stay[i]), 4),
                          p_handover=_first_per_cell(obs.nbr_ids[i], p_ho[i]),
                          source="llm")
            if conf < min_confidence:
                gap = obs.nbr_hist[i, :, -a3_ttt:] - obs.serving_hist[i, -a3_ttt:]
                ok = (gap > a3_hyst_db).all(axis=1)
                if ok.any():
                    c = int(obs.nbr_ids[i][np.argmax(np.where(ok, obs.nbr_hist[i, :, -1], -np.inf))])
                    answer.update(action="HANDOVER", target_cell=c, source="a3_fallback",
                                  rationale=f"Low model confidence; A3 fallback: cell {c} exceeds serving by "
                                            f">{a3_hyst_db} dB for the last {a3_ttt} reports.")
                else:
                    answer.update(action="STAY", target_cell=None, source="a3_fallback",
                                  rationale="Low model confidence; A3 fallback: no neighbour meets the A3 entry "
                                            "condition.")
            answers.append(answer)
        return answers

    def explain_answer(self, ids: list[int]) -> dict:
        out = self.tok.explain(ids)
        out["tokens"] = self.tok.decode(ids)
        return out


def report_to_observation(report: dict, obs_cfg: ObsConfig) -> Observation:
    h, k = obs_cfg.hist_len, obs_cfg.n_neighbors

    def hist(v):
        v = list(np.atleast_1d(np.asarray(v, dtype=float)))
        return np.asarray(([v[0]] * (h - len(v)) + v)[-h:])   # left-pad / truncate to hist_len

    nbrs = sorted(report["neighbors"], key=lambda n: -hist(n["rsrp"])[-1])[:k]
    if not nbrs:
        raise ValueError("report must contain at least one neighbour")
    while len(nbrs) < k:          # pad with a very weak copy of the last one
        nbrs.append({"cell_id": nbrs[-1]["cell_id"], "rsrp": [-140.0] * h})
    context = None
    if "context" in report or any("context" in n for n in nbrs):
        # Location context from a Location xApp (docs/LOCATION_AWARE_HANDOVER.md §12):
        # report["context"] = {"radial_speed": m/s to serving, "next_count": n}
        # neighbors[i]["context"] = {"dist_ratio", "radial_speed", "map_gain_now",
        #                            "map_gain_ahead", "next_prob"}; missing = unknown
        context = empty_context(1, k)
        rc = report.get("context", {})
        context["radial_speed"][0, 0] = rc.get("radial_speed", np.nan)
        context["next_count"][0] = rc.get("next_count", np.nan)
        for j, n in enumerate(nbrs):
            nc = n.get("context", {})
            for key in ("dist_ratio", "map_gain_now", "map_gain_ahead", "next_prob"):
                context[key][0, j] = nc.get(key, np.nan)
            context["radial_speed"][0, j + 1] = nc.get("radial_speed", np.nan)
    return Observation(
        serving=np.array([int(report["serving_cell"])]),
        serving_hist=hist(report["serving_rsrp"])[None],
        nbr_ids=np.array([[int(n["cell_id"]) for n in nbrs]]),
        nbr_hist=np.stack([hist(n["rsrp"]) for n in nbrs])[None],
        sinr_db=np.array([float(report.get("sinr_db", 0.0))]),
        speed_kmh=np.array([float(report.get("speed_kmh", 30.0))]),
        context=context,
    )


def _first_per_cell(cells, probs) -> dict:
    out = {}
    for c, p in zip(cells, probs):
        out.setdefault(int(c), round(float(p), 4))
    return out


CONTEXT_KEYS = ("dist_ratio", "radial_speed", "map_gain_now", "map_gain_ahead", "next_prob", "next_count")


def empty_context(n_ue: int, k: int) -> dict:
    """All-unknown (NaN) location context for `n_ue` UEs with `k` neighbours."""
    shape = {"radial_speed": (n_ue, k + 1), "next_count": (n_ue,)}
    return {key: np.full(shape.get(key, (n_ue, k)), np.nan) for key in CONTEXT_KEYS}


def stack_observations(observations: list[Observation]) -> Observation:
    """Concatenate single- or multi-UE observations along the UE axis."""
    out = Observation(*(np.concatenate([getattr(o, f) for o in observations])
                        for f in ("serving", "serving_hist", "nbr_ids", "nbr_hist", "sinr_db", "speed_kmh")))
    if any(o.context for o in observations):
        ctxs = [o.context or empty_context(len(o.serving), o.nbr_ids.shape[1]) for o in observations]
        out.context = {k: np.concatenate([np.asarray(c.get(k, empty_context(len(o.serving), o.nbr_ids.shape[1])[k]),
                                                     dtype=float) for c, o in zip(ctxs, observations)])
                       for k in CONTEXT_KEYS}
    return out


@dataclass
class ReturnGuard:
    """Model-side hysteresis against ping-pong (docs/DESIGN.md §21.17).

    Within `window_s` of a handover, handing back to the cell the UE just left needs
    P(<ho> to it) >= `threshold` *and* that cell `margin_db` stronger than serving (latest
    filtered RSRP). Other targets keep the normal threshold, and so does the return when
    serving SINR is below `rescue_sinr_db`: the guard must not hold a UE on a failing link
    (a flat hold-off did exactly that, see RAN_INTEGRATION.md §6).

    Tracks each UE's previous cell and last handover time from the serving cell it
    sees in consecutive reports, so it works with any report source.
    """

    window_s: float = 2.0
    threshold: float = 0.9
    margin_db: float = 5.0
    rescue_sinr_db: float = -6.0      # -8 trades a little outage for about half the ping-pong (§21.17)

    def reset(self) -> None:
        self.last = None

    def thresholds(self, t_s: float, obs: Observation, base: float) -> np.ndarray:
        """(U, K) per-neighbour thresholds for the report at time `t_s` (seconds)."""
        serving = np.asarray(obs.serving)
        if getattr(self, "last", None) is None or len(self.last) != len(serving):
            self.prev = np.full(len(serving), -1)
            self.ho_t = np.full(len(serving), -np.inf)
        else:
            changed = serving != self.last
            self.prev = np.where(changed, self.last, self.prev)
            self.ho_t = np.where(changed, t_s, self.ho_t)
        self.last = serving.copy()
        thr = np.full(obs.nbr_ids.shape, float(base))
        recent = (t_s - self.ho_t <= self.window_s) & (np.asarray(obs.sinr_db) >= self.rescue_sinr_db)
        back = recent[:, None] & (obs.nbr_ids == self.prev[:, None])
        weak = obs.nbr_hist[:, :, -1] - obs.serving_hist[:, -1:] < self.margin_db
        thr[back] = max(self.threshold, base)
        thr[back & weak] = np.inf
        return thr


class LLMPolicy:
    """Closed-loop policy for :func:`ai_ran_llm.simulator.run_policy`."""

    def __init__(self, llm: HandoverLLM, ho_threshold: float = 0.5, guard: ReturnGuard | None = None):
        self.llm = llm
        self.ho_threshold = ho_threshold
        self.guard = guard

    def reset(self, ep: Episode, obs_cfg: ObsConfig) -> None:
        if self.guard is not None:
            self.guard.reset()

    def decide(self, ep: Episode, t: int, obs: Observation) -> np.ndarray:
        thr = self.ho_threshold if self.guard is None else self.guard.thresholds(t * ep.sim.dt_s, obs, self.ho_threshold)
        return self.llm.decide_batch(obs, thr)[0]
