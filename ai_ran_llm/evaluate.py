"""Closed-loop benchmark: HandoverLLM vs 3GPP A3 vs the look-ahead teacher."""

import numpy as np

from .config import ObsConfig, SimConfig
from .policies import A3Policy, OraclePolicy
from .simulator import Metrics, generate_episode, run_policy

COLUMNS = [("ho_per_ue_min", "HO/UE/min"), ("ping_pong_pct", "ping-pong %"), ("rlf_per_ue_min", "RLF/UE/min"),
           ("hof_per_ue_min", "HOF/UE/min"), ("mean_sinr_db", "SINR dB"), ("mean_se_bps_hz", "SE b/s/Hz"),
           ("outage_pct", "outage %")]


def benchmark(policies: dict, n_episodes: int = 5, n_ue: int = 64, n_steps: int = 600, seed: int = 10_000,
              sim: SimConfig | None = None, obs_cfg: ObsConfig | None = None) -> dict:
    """`policies` maps name -> zero-arg factory. Every policy sees identical drives."""
    obs_cfg = obs_cfg or ObsConfig()
    results = {name: Metrics() for name in policies}
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        ep = generate_episode(n_ue, n_steps, rng, sim)
        for name, make in policies.items():
            m, _ = run_policy(ep, make(), obs_cfg)
            results[name].merge(m)
    return {name: m.summary() for name, m in results.items()}


def default_policies(llm=None, obs_cfg: ObsConfig | None = None, min_confidence: float = 0.5) -> dict:
    obs_cfg = obs_cfg or ObsConfig()
    pol = {
        "A3 (1dB, 200ms)": lambda: A3Policy(1.0, 2),
        "A3 (2dB, 300ms)": lambda: A3Policy(2.0, 3),
        "A3 (3dB, 500ms)": lambda: A3Policy(3.0, 5),
    }
    if llm is not None:
        from .inference import LLMPolicy
        pol["HandoverLLM"] = lambda: LLMPolicy(llm, min_confidence)
    pol["Oracle (non-causal)"] = lambda: OraclePolicy(obs_cfg)
    return pol


def format_table(results: dict) -> str:
    w = max(len(n) for n in results) + 2
    head = "policy".ljust(w) + "".join(h.rjust(13) for _, h in COLUMNS)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lines.append(name.ljust(w) + "".join(f"{r[k]:13.3f}" for k, _ in COLUMNS))
    return "\n".join(lines)
