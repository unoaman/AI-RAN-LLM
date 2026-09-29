"""Inference: grammar-constrained decoding, closed-loop policy and xApp-style API."""

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
        p_cell = torch.softmax(cell_logits, dim=-1)
        return act[:, 0].cpu().numpy(), (act[:, 1:2] * p_cell).cpu().numpy()

    def decide_batch(self, obs: Observation, ho_threshold: float = 0.5):
        """Hand over to the most likely neighbour when P(<ho> to it) >= `ho_threshold`.

        Handovers are rare, so a threshold below 0.5 trades a few extra handovers
        for earlier ones. Returns (target (U,) with -1 = stay, confidence (U,)).
        """
        p_stay, p_ho = self.score(self.tok.encode_prompts(obs))
        k = p_ho.argmax(axis=1)
        best = p_ho[np.arange(len(k)), k]
        do_ho = best >= ho_threshold
        target = np.where(do_ho, obs.nbr_ids[np.arange(len(k)), k], -1)
        return target, np.where(do_ho, best, p_stay)

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
        obs = report_to_observation(report, self.tok.obs_cfg)
        prompt = self.tok.encode_prompts(obs)
        p_stay, p_ho = self.score(prompt)
        k = int(p_ho[0].argmax())
        if p_ho[0, k] >= ho_threshold:
            prefix, conf = [self.tok.HO, int(self.tok.cell(obs.nbr_ids[0, k]))], float(p_ho[0, k])
        else:
            prefix, conf = [self.tok.STAY], float(p_stay[0])
        answer = self.explain_answer(self.generate(prompt[0], prefix=prefix))
        answer.update(confidence=round(conf, 4), p_stay=round(float(p_stay[0]), 4),
                      p_handover={int(c): round(float(p), 4) for c, p in zip(obs.nbr_ids[0], p_ho[0])},
                      source="llm")
        if conf < min_confidence:
            gap = obs.nbr_hist[0, :, -a3_ttt:] - obs.serving_hist[0, -a3_ttt:]
            ok = (gap > a3_hyst_db).all(axis=1)
            if ok.any():
                c = int(obs.nbr_ids[0][np.argmax(np.where(ok, obs.nbr_hist[0, :, -1], -np.inf))])
                answer.update(action="HANDOVER", target_cell=c, source="a3_fallback",
                              rationale=f"Low model confidence; A3 fallback: cell {c} exceeds serving by "
                                        f">{a3_hyst_db} dB for the last {a3_ttt} reports.")
            else:
                answer.update(action="STAY", target_cell=None, source="a3_fallback",
                              rationale="Low model confidence; A3 fallback: no neighbour meets the A3 entry condition.")
        return answer

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
    return Observation(
        serving=np.array([int(report["serving_cell"])]),
        serving_hist=hist(report["serving_rsrp"])[None],
        nbr_ids=np.array([[int(n["cell_id"]) for n in nbrs]]),
        nbr_hist=np.stack([hist(n["rsrp"]) for n in nbrs])[None],
        sinr_db=np.array([float(report.get("sinr_db", 0.0))]),
        speed_kmh=np.array([float(report.get("speed_kmh", 30.0))]),
    )


class LLMPolicy:
    """Closed-loop policy for :func:`ai_ran_llm.simulator.run_policy`."""

    def __init__(self, llm: HandoverLLM, ho_threshold: float = 0.5):
        self.llm = llm
        self.ho_threshold = ho_threshold

    def decide(self, ep: Episode, t: int, obs: Observation) -> np.ndarray:
        return self.llm.decide_batch(obs, self.ho_threshold)[0]
