"""Build supervised handover corpora from simulated drives."""

import json

import numpy as np

from .config import ObsConfig, SimConfig
from .policies import A3Policy, OraclePolicy, oracle_decision
from .simulator import generate_episode, run_policy
from .tokenizer import HandoverTokenizer


def _behaviour_policy(rng: np.random.Generator, obs_cfg: ObsConfig):
    """Mix of policies driving the rollout, so the corpus covers good *and*
    off-optimal serving-cell states (late, early and missed handovers)."""
    r = rng.uniform()
    if r < 0.35:
        return OraclePolicy(obs_cfg)
    if r < 0.6:
        return OraclePolicy(obs_cfg, exec_prob=rng.uniform(0.1, 0.5), seed=int(rng.integers(1 << 31)))
    return A3Policy(hyst_db=float(rng.uniform(0.0, 5.0)), ttt_steps=int(rng.integers(1, 8)))


def generate_dataset(n_episodes: int, n_ue: int = 32, n_steps: int = 600, seed: int = 0,
                     sim: SimConfig | None = None, obs_cfg: ObsConfig | None = None,
                     block_size: int = 64, easy_keep_prob: float = 0.1,
                     verbose: bool = True) -> dict:
    """Returns {"tokens": (N, block_size) int64 padded with <pad>, "prompt_len": int}.

    Labels always come from the look-ahead teacher. Samples where every
    neighbour is >6 dB below the serving cell are trivially STAY and are
    subsampled with `easy_keep_prob`.
    """
    sim = sim or SimConfig()
    obs_cfg = obs_cfg or ObsConfig()
    tok = HandoverTokenizer(obs_cfg)
    rng = np.random.default_rng(seed)
    rows, n_ho = [], 0

    for e in range(n_episodes):
        ep = generate_episode(n_ue, n_steps, rng, sim)
        _, _, seen = run_policy(ep, _behaviour_policy(rng, obs_cfg), obs_cfg, record=True)
        for t, obs in seen:
            if t < obs_cfg.hist_stride * (obs_cfg.hist_len - 1) or t + obs_cfg.oracle_horizon >= n_steps:
                continue
            target, gain = oracle_decision(ep, t, obs, obs_cfg)
            margin = obs.nbr_hist[:, 0, -1] - obs.serving_hist[:, -1]
            keep = (target >= 0) | (margin > -6.0) | (rng.uniform(size=len(target)) < easy_keep_prob)
            prompts = tok.encode_prompts(obs)
            for u in np.nonzero(keep)[0]:
                ans = tok.encode_answer(int(target[u]), float(gain[u]), obs.serving_hist[u],
                                        obs.nbr_ids[u], obs.nbr_hist[u], float(obs.sinr_db[u]))
                row = np.full(block_size, tok.PAD, dtype=np.int64)
                seq = np.concatenate([prompts[u], ans])
                row[: len(seq)] = seq
                rows.append(row)
                n_ho += int(target[u] >= 0)
        if verbose:
            print(f"episode {e + 1}/{n_episodes}: {len(rows)} samples ({n_ho} handovers)", flush=True)

    tokens = np.stack(rows)
    rng.shuffle(tokens)
    return {"tokens": tokens, "prompt_len": np.int64(tok.prompt_len)}


def prompt_to_text(tok: HandoverTokenizer, prompt_ids) -> str:
    """Render a tokenised report as plain English (for general-purpose LLM fine-tuning)."""
    t = [tok.itos[int(i)] for i in prompt_ids]
    h = tok.obs_cfg.hist_len
    speed_bin = int(t[2][1:])
    lines = [f"UE speed {speed_bin * 10}-{speed_bin * 10 + 9} km/h, serving SINR {t[4][1:]} dB.",
             f"Serving cell {t[6][1:]}, L3-filtered RSRP history (dBm, oldest first): "
             + ", ".join(x[1:] for x in t[7:7 + h]) + "."]
    i = 7 + h
    while i < len(t) and t[i] == "<nbr>":
        lines.append(f"Neighbour cell {t[i + 1][1:]}, RSRP relative to serving (dB): "
                     + ", ".join(x[1:] for x in t[i + 2:i + 2 + h]) + ".")
        i += 2 + h
    return "\n".join(lines)


def export_jsonl(tokens: np.ndarray, prompt_len: int, path: str, limit: int | None = None,
                 obs_cfg: ObsConfig | None = None) -> int:
    """Write chat-style instruction data usable to fine-tune any open LLM."""
    tok = HandoverTokenizer(obs_cfg)
    system = ("You are a near-RT RIC mobility xApp. Given a UE measurement report, decide whether to "
              "hand over and to which reported neighbour cell, and explain why.")
    n = 0
    with open(path, "w") as f:
        for row in tokens[:limit]:
            ans = [i for i in row[prompt_len:] if i != tok.PAD]
            rec = {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_to_text(tok, row[:prompt_len])},
                {"role": "assistant", "content": tok.explain(ans)["rationale"]},
            ]}
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
