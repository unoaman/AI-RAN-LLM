"""Classical and teacher handover policies."""

import numpy as np

from .config import ObsConfig
from .simulator import Episode, Observation


class A3Policy:
    """3GPP event A3: neighbour > serving + hysteresis for time-to-trigger."""

    def __init__(self, hyst_db: float = 2.0, ttt_steps: int = 3):
        self.hyst_db = hyst_db
        self.ttt_steps = ttt_steps

    def reset(self, ep: Episode, obs_cfg: ObsConfig):
        self.counter = np.zeros((ep.n_ue, ep.n_cells), dtype=int)

    def decide(self, ep: Episode, t: int, obs: Observation) -> np.ndarray:
        u = np.arange(ep.n_ue)
        meas = ep.rsrp_meas[:, t, :]
        entering = meas > meas[u, obs.serving][:, None] + self.hyst_db
        entering[u, obs.serving] = False
        self.counter = np.where(entering, self.counter + 1, 0)
        ready = np.where(self.counter >= self.ttt_steps, meas, -np.inf)
        best = ready.argmax(axis=1)
        return np.where(np.isfinite(ready[u, best]), best, -1)

    def on_handover(self, mask: np.ndarray):
        self.counter[mask] = 0


def oracle_decision(ep: Episode, t: int, obs: Observation, obs_cfg: ObsConfig):
    """Look-ahead teacher used to label training data.

    Picks, among the *reported* neighbours, the cell with the highest mean
    large-scale RSRP over the next `oracle_horizon` steps, and hands over only if
    it beats the serving cell by `oracle_margin_db`. Non-causal: it sees the
    future, which is exactly what the LLM has to learn to anticipate.

    Returns (target (U,) with -1 = stay, predicted gain in dB (U,)).
    """
    u = np.arange(ep.n_ue)
    lo, hi = min(t + 1, ep.n_steps - 1), min(t + 1 + obs_cfg.oracle_horizon, ep.n_steps)
    future = ep.rsrp_true[:, lo:hi, :].mean(axis=1)            # (U, C)
    f_serv = future[u, obs.serving]
    f_nbr = np.take_along_axis(future, obs.nbr_ids, axis=1)    # (U, K)
    k = f_nbr.argmax(axis=1)
    best = obs.nbr_ids[u, k]
    gain = f_nbr[u, k] - f_serv
    target = np.where(gain > obs_cfg.oracle_margin_db, best, -1)
    return target, gain


class OraclePolicy:
    def __init__(self, obs_cfg: ObsConfig, exec_prob: float = 1.0, seed: int = 0):
        self.obs_cfg = obs_cfg
        self.exec_prob = exec_prob          # < 1 delays handovers -> off-optimal states
        self.rng = np.random.default_rng(seed)

    def decide(self, ep: Episode, t: int, obs: Observation) -> np.ndarray:
        target, _ = oracle_decision(ep, t, obs, self.obs_cfg)
        if self.exec_prob < 1.0:
            skip = self.rng.uniform(size=target.shape) > self.exec_prob
            target = np.where(skip, -1, target)
        return target
