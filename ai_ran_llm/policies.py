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


def _future_mean(ep: Episode, start: int, length: int) -> np.ndarray:
    lo = min(start, ep.n_steps - 1)
    return ep.rsrp_true[:, lo:min(lo + length, ep.n_steps), :].mean(axis=1)


def label_decision(ep: Episode, t: int, obs: Observation, obs_cfg: ObsConfig):
    """Smoothed teacher used for training labels.

    The raw teacher answers "hand over *exactly now*?", which flips on and off
    with future shadowing the model cannot see. This label instead asks:

    * **window:** will the teacher want to hand over to a reported neighbour at any
      of the next `label_window` steps (serving cell held fixed)? The earliest such
      step decides the target, so labels lean early rather than late.
    * **confirm:** does that target still beat the serving cell on average over the
      next `label_confirm_horizon` steps? This drops handovers that the teacher
      would soon reverse (ping-pong prone).

    With both set to 0 this is exactly :func:`oracle_decision`.
    Returns (target (U,) with -1 = stay, gain in dB (U,)).
    """
    u = np.arange(ep.n_ue)
    target, gain = oracle_decision(ep, t, obs, obs_cfg)
    for d in range(1, obs_cfg.label_window + 1):
        if t + d + 1 >= ep.n_steps:
            break
        future = _future_mean(ep, t + d + 1, obs_cfg.oracle_horizon)
        f_nbr = np.take_along_axis(future, obs.nbr_ids, axis=1)
        k = f_nbr.argmax(axis=1)
        g = f_nbr[u, k] - future[u, obs.serving]
        hit = (target < 0) & (g > obs_cfg.oracle_margin_db)
        target = np.where(hit, obs.nbr_ids[u, k], target)
        gain = np.where(hit, g, gain)
    if obs_cfg.label_confirm_horizon > 0:
        longer = _future_mean(ep, t + 1, obs_cfg.label_confirm_horizon)
        ok = longer[u, np.maximum(target, 0)] > longer[u, obs.serving]
        target = np.where(ok, target, -1)
    return target, gain


class OraclePolicy:
    def __init__(self, obs_cfg: ObsConfig, exec_prob: float = 1.0, seed: int = 0, smoothed: bool = False):
        self.obs_cfg = obs_cfg
        self.exec_prob = exec_prob          # < 1 delays handovers -> off-optimal states
        self.rng = np.random.default_rng(seed)
        self.decision = label_decision if smoothed else oracle_decision

    def decide(self, ep: Episode, t: int, obs: Observation) -> np.ndarray:
        target, _ = self.decision(ep, t, obs, self.obs_cfg)
        if self.exec_prob < 1.0:
            skip = self.rng.uniform(size=target.shape) > self.exec_prob
            target = np.where(skip, -1, target)
        return target
