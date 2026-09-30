"""Location context for HandoverLLM: the "Location xApp" logic.

Computes, per UE and report, the context tokens enabled by ``ObsConfig.use_position``,
``use_radio_map`` and ``use_trajectory`` (docs/LOCATION_AWARE_HANDOVER.md §12):

=====================  ===========================================================  ================
context key            meaning                                                      needs
=====================  ===========================================================  ================
``dist_ratio``         log10(d_serving / d_neighbour) per neighbour                  position
``radial_speed``       rate of change of the distance to serving + each neighbour    position
                       (m/s, negative = approaching), over the last 800 ms
``map_gain_now``       radio map: RSRP(neighbour) − RSRP(serving) at the UE          position + map
``map_gain_ahead``     the same at the position predicted ``forecast_s`` ahead       position + map
``next_prob``          P(next serving cell = neighbour | previous, current cell)     handover history
``next_count``         how many past handovers the estimate rests on                 handover history
=====================  ===========================================================  ================

The pieces:

* :func:`position_track` stands in for position estimation (AoA / TA / RTT fusion): the
  true simulator position plus a temporally correlated error (AR(1), σ ``pos_sigma_m``).
* :class:`RadioMap` averages reported RSRP per (cell, 20 m grid square) over past traffic.
* :class:`TransitionModel` mines sequences of serving cells (handover history) for
  next-cell probabilities, backing off from (previous, current) to (current).
* :class:`LocationService` builds both from *history drives* that are separate from any
  drive used for training or evaluation, so no drive's own future can leak into its inputs.
* :class:`LocationAwarePolicy` runs a context-enabled model in the closed-loop simulator.

In a real RAN the same inputs come from a Location xApp (positions from the RAN, maps and
statistics from logs) and reach the model through the report JSON; see
``inference.report_to_observation``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .config import ObsConfig, SimConfig

EXTENT_M = 2560.0


@dataclass
class LocationConfig:
    pos_sigma_m: float = 10.0       # positioning error (std per axis)
    pos_corr: float = 0.9           # AR(1) coefficient per 100 ms step (≈ 1 s memory)
    grid_m: float = 20.0            # radio-map resolution
    min_samples: int = 2            # radio-map squares with fewer samples count as unknown
    report_top: int = 9             # cells per report used for the map (serving + 8 neighbours)
    forecast_s: float = 1.0         # look-ahead for map_gain_ahead
    motion_window_s: float = 0.8    # window for velocity / radial speed
    min_transitions: int = 5        # below this, back off from (prev, cur) to (cur)
    context_dropout: float = 0.1    # share of training samples whose context is hidden (robustness)


def position_track(ep, rng: np.random.Generator, cfg: LocationConfig) -> np.ndarray:
    """Estimated positions (U, T, 2): true position + AR(1) error (stand-in for AoA/TA fusion)."""
    U, T = ep.n_ue, ep.n_steps
    err = np.zeros((U, T, 2))
    if cfg.pos_sigma_m > 0:
        a = cfg.pos_corr
        x = rng.normal(0, cfg.pos_sigma_m, (U, 2))
        for t in range(T):
            err[:, t] = x
            x = a * x + np.sqrt(1 - a * a) * rng.normal(0, cfg.pos_sigma_m, (U, 2))
    return ep.pos + err


def previous_cells(traj: np.ndarray) -> np.ndarray:
    """(U, T) previous distinct serving cell at each step (-1 before the first handover)."""
    U, T = traj.shape
    prev = np.full((U, T), -1)
    cur_prev = np.full(U, -1)
    for t in range(1, T):
        changed = traj[:, t] != traj[:, t - 1]
        cur_prev = np.where(changed, traj[:, t - 1], cur_prev)
        prev[:, t] = cur_prev
    return prev


class RadioMap:
    def __init__(self, n_cells: int, cfg: LocationConfig):
        self.cfg = cfg
        self.n = int(EXTENT_M / cfg.grid_m)
        self.sum = np.zeros((n_cells, self.n, self.n))
        self.cnt = np.zeros((n_cells, self.n, self.n))

    def _grid(self, p):
        g = np.floor(np.asarray(p) / self.cfg.grid_m + self.n / 2).astype(int)
        inside = (g >= 0).all(-1) & (g < self.n).all(-1)
        return np.clip(g[..., 0], 0, self.n - 1), np.clip(g[..., 1], 0, self.n - 1), inside

    def add(self, ep, est: np.ndarray) -> None:
        """Add every UE's reported RSRP (serving + strongest neighbours) at its estimated position."""
        gx, gy, inside = self._grid(est)
        top = np.argsort(-ep.rsrp_meas, axis=2)[:, :, : self.cfg.report_top]
        for k in range(top.shape[2]):
            c = top[:, :, k]
            v = np.take_along_axis(ep.rsrp_meas, c[..., None], 2)[..., 0]
            np.add.at(self.sum, (c[inside], gx[inside], gy[inside]), v[inside])
            np.add.at(self.cnt, (c[inside], gx[inside], gy[inside]), 1)

    def lookup(self, cells: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Mean RSRP of `cells` (N, M) at positions (N, 2); NaN where unknown."""
        gx, gy, inside = self._grid(pos)
        s = self.sum[cells, gx[:, None], gy[:, None]]
        n = self.cnt[cells, gx[:, None], gy[:, None]]
        ok = (n >= self.cfg.min_samples) & inside[:, None]
        return np.where(ok, s / np.maximum(n, 1), np.nan)

    def coverage(self) -> float:
        return float((self.cnt.max(0) >= self.cfg.min_samples).mean())


class TransitionModel:
    def __init__(self, cfg: LocationConfig):
        self.cfg = cfg
        self.t3: dict = defaultdict(lambda: defaultdict(int))
        self.t2: dict = defaultdict(lambda: defaultdict(int))

    def add(self, traj: np.ndarray) -> None:
        """Mine the sequences of distinct serving cells of every UE."""
        for row in traj:
            seq = [int(row[0])] + [int(c) for p, c in zip(row[:-1], row[1:]) if c != p]
            for i in range(1, len(seq)):
                self.t2[seq[i - 1]][seq[i]] += 1
                if i >= 2:
                    self.t3[(seq[i - 2], seq[i - 1])][seq[i]] += 1

    def probs(self, prev: np.ndarray, cur: np.ndarray, cand: np.ndarray):
        """P(next = cand[:, k] | prev, cur) (N, K) and the supporting count (N,); NaN if no history."""
        N, K = cand.shape
        p = np.full((N, K), np.nan)
        n = np.zeros(N)
        for i in range(N):
            dist = self.t3.get((int(prev[i]), int(cur[i]))) if prev[i] >= 0 else None
            if not dist or sum(dist.values()) < self.cfg.min_transitions:
                dist = self.t2.get(int(cur[i]))
            if not dist:
                continue
            tot = sum(dist.values())
            n[i] = tot
            p[i] = [dist.get(int(c), 0) / tot for c in cand[i]]
        return p, n

    def to_dict(self) -> dict:
        return {"t2": {str(a): dict(b) for a, b in self.t2.items()},
                "t3": {f"{a},{b}": dict(c) for (a, b), c in self.t3.items()}}

    @classmethod
    def from_dict(cls, d: dict, cfg: LocationConfig) -> "TransitionModel":
        m = cls(cfg)
        for a, b in d["t2"].items():
            m.t2[int(a)].update({int(k): v for k, v in b.items()})
        for ab, c in d["t3"].items():
            a, b = (int(x) for x in ab.split(","))
            m.t3[(a, b)].update({int(k): v for k, v in c.items()})
        return m


class LocationService:
    """Radio map + transition model learned from history drives."""

    def __init__(self, radio_map: RadioMap, transitions: TransitionModel, cfg: LocationConfig, sim: SimConfig):
        self.radio_map, self.transitions, self.cfg, self.sim = radio_map, transitions, cfg, sim

    @classmethod
    def build(cls, sim: SimConfig, n_drives: int = 20, n_ue: int = 32, n_steps: int = 600, seed: int = 777,
              cfg: LocationConfig | None = None, verbose: bool = False) -> "LocationService":
        """Simulate history drives under classical A3 handover (the network's current behaviour)
        and learn the radio map and handover-sequence statistics from them."""
        from .policies import A3Policy
        from .simulator import generate_episode, run_policy
        cfg = cfg or LocationConfig()
        rng = np.random.default_rng(seed)
        rmap, trans = None, TransitionModel(cfg)
        for d in range(n_drives):
            ep = generate_episode(n_ue, n_steps, rng, sim)
            rmap = rmap or RadioMap(ep.n_cells, cfg)
            rmap.add(ep, position_track(ep, np.random.default_rng([seed, d, 11]), cfg))
            _, traj = run_policy(ep, A3Policy(2.0, 3), ObsConfig())
            trans.add(traj)
            if verbose:
                print(f"history drive {d + 1}/{n_drives}: map coverage {rmap.coverage():.0%}", flush=True)
        return cls(rmap, trans, cfg, sim)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(os.path.join(path, "radio_map.npz"), sum=self.radio_map.sum.astype(np.float32),
                            cnt=self.radio_map.cnt.astype(np.float32))
        with open(os.path.join(path, "service.json"), "w") as f:
            json.dump({"location_config": asdict(self.cfg), "sim_config": asdict(self.sim),
                       "transitions": self.transitions.to_dict()}, f)

    @classmethod
    def load(cls, path: str) -> "LocationService":
        with open(os.path.join(path, "service.json")) as f:
            meta = json.load(f)
        cfg = LocationConfig(**meta["location_config"])
        d = np.load(os.path.join(path, "radio_map.npz"))
        rmap = RadioMap(d["sum"].shape[0], cfg)
        rmap.sum, rmap.cnt = d["sum"].astype(np.float64), d["cnt"].astype(np.float64)
        return cls(rmap, TransitionModel.from_dict(meta["transitions"], cfg), cfg, SimConfig(**meta["sim_config"]))


def compute_context(ep, t: int, obs, est: np.ndarray, prev_cell: np.ndarray,
                    service: LocationService | None, cfg: LocationConfig) -> dict:
    """All context arrays for the UEs of `obs` at step t (see module doc)."""
    U, K = obs.nbr_ids.shape
    lag = min(t, max(1, int(round(cfg.motion_window_s / ep.sim.dt_s))))
    dt = max(lag, 1) * ep.sim.dt_s
    p_now, p_old = est[:, t], est[:, t - lag]
    cells = np.concatenate([obs.serving[:, None], obs.nbr_ids], 1)
    site = ep.sites[cells]
    d_now = np.linalg.norm(p_now[:, None] - site, axis=-1) + 1.0
    d_old = np.linalg.norm(p_old[:, None] - site, axis=-1) + 1.0
    ctx = {"dist_ratio": np.log10(d_now[:, :1] / d_now[:, 1:]),
           "radial_speed": (d_now - d_old) / dt if lag > 0 else np.zeros((U, K + 1))}
    nan = np.full((U, K), np.nan)
    ctx["map_gain_now"], ctx["map_gain_ahead"], ctx["next_prob"] = nan, nan.copy(), nan.copy()
    ctx["next_count"] = np.full(U, np.nan)
    if service is not None:
        vel = (p_now - p_old) / dt
        for key, p in (("map_gain_now", p_now), ("map_gain_ahead", p_now + vel * cfg.forecast_s)):
            v = service.radio_map.lookup(cells, p)
            ctx[key] = v[:, 1:] - v[:, :1]
        ctx["next_prob"], ctx["next_count"] = service.transitions.probs(prev_cell, obs.serving, obs.nbr_ids)
    return ctx


def drop_context(ctx: dict, mask: np.ndarray) -> dict:
    """Hide the whole context of the UEs in `mask` (training-time robustness)."""
    return {k: np.where(mask.reshape((-1,) + (1,) * (v.ndim - 1)), np.nan, v) for k, v in ctx.items()}


class LocationAwarePolicy:
    """Closed-loop policy for ``run_policy`` that feeds a context-enabled model."""

    def __init__(self, llm, service: LocationService | None, ho_threshold: float = 0.35,
                 cfg: LocationConfig | None = None, seed: int = 0):
        self.llm, self.service, self.ho_threshold = llm, service, ho_threshold
        self.cfg = cfg or (service.cfg if service is not None else LocationConfig())
        self.seed = seed

    def reset(self, ep, obs_cfg) -> None:
        self.est = position_track(ep, np.random.default_rng([self.seed, 99]), self.cfg)
        self.prev = np.full(ep.n_ue, -1)
        self.last = None

    def decide(self, ep, t, obs) -> np.ndarray:
        if self.last is not None:
            changed = obs.serving != self.last
            self.prev = np.where(changed, self.last, self.prev)
        self.last = obs.serving.copy()
        obs.context = compute_context(ep, t, obs, self.est, self.prev, self.service, self.cfg)
        return self.llm.decide_batch(obs, self.ho_threshold)[0]
