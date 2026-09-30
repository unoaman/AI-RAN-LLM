"""Vectorised multi-cell mobility simulator.

The radio channel does not depend on which cell serves a UE, so an episode is
generated in two phases:

1. :func:`generate_episode` draws UE trajectories and the per-cell channel
   (path loss, correlated shadowing, fast fading, L3-filtered measurements).
2. :func:`run_policy` replays the episode with a handover policy in the loop and
   scores it (handovers, ping-pongs, radio link failures, SINR, throughput).
"""

from dataclasses import dataclass, field

import numpy as np

from .config import ObsConfig, SimConfig


def hex_sites(rings: int, isd: float) -> np.ndarray:
    """(C, 2) site coordinates of a hexagonal grid with `rings` rings."""
    pts = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if max(abs(q), abs(r), abs(q + r)) <= rings:
                pts.append((isd * (q + r / 2.0), isd * r * np.sqrt(3) / 2.0))
    pts.sort(key=lambda p: (round(p[1], 3), round(p[0], 3)))
    return np.asarray(pts, dtype=np.float64)


@dataclass
class Episode:
    sites: np.ndarray        # (C, 2)
    pos: np.ndarray          # (U, T, 2)
    speed_kmh: np.ndarray    # (U,)
    rsrp_true: np.ndarray    # (U, T, C) large-scale RSRP (path loss + shadowing)
    rsrp_inst: np.ndarray    # (U, T, C) with fast fading, drives SINR
    rsrp_meas: np.ndarray    # (U, T, C) L3-filtered UE measurements (what policies see)
    sim: SimConfig = field(repr=False)
    # Optional, from logged traces: serving cell and SINR at each report (U, T)
    logged_serving: np.ndarray | None = field(default=None, repr=False)
    logged_sinr_db: np.ndarray | None = field(default=None, repr=False)
    # Road mobility (SimConfig.mobility == "roads"): index into city.city_routes(sim) per UE
    route_id: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_ue(self) -> int:
        return self.rsrp_true.shape[0]

    @property
    def n_steps(self) -> int:
        return self.rsrp_true.shape[1]

    @property
    def n_cells(self) -> int:
        return self.rsrp_true.shape[2]

    def sinr_db(self, t: int, serving: np.ndarray, large_scale: bool = False) -> np.ndarray:
        """Downlink SINR of each UE towards its serving cell at step t.

        `large_scale=True` ignores fast fading (SINR averaged over many TTIs).
        """
        rsrp = self.rsrp_true if large_scale else self.rsrp_inst
        lin = 10.0 ** (rsrp[:, t, :] / 10.0)
        s = lin[np.arange(self.n_ue), serving]
        interf = self.sim.load * (lin.sum(axis=1) - s)
        noise = 10.0 ** (self.sim.noise_dbm / 10.0)
        return 10.0 * np.log10(s / (interf + noise))


def generate_episode(n_ue: int, n_steps: int, rng: np.random.Generator,
                     sim: SimConfig | None = None) -> Episode:
    sim = sim or SimConfig()
    sites = hex_sites(sim.rings, sim.isd_m)
    n_cells = len(sites)
    radius = sim.rings * sim.isd_m * 0.85
    if sim.mobility not in ("random", "roads"):
        raise ValueError(f"unknown mobility model {sim.mobility!r}")
    if sim.shadowing not in ("per_ue", "spatial"):
        raise ValueError(f"unknown shadowing model {sim.shadowing!r}")

    if sim.mobility == "roads":
        # --- mobility: routes on the city's road network (city.py) ---
        from .city import road_mobility
        pos, speed_kmh, route_id = road_mobility(n_ue, n_steps, rng, sim)
        return _channel(pos, speed_kmh, sites, rng, sim, route_id)

    # --- mobility: Gauss-Markov heading, constant per-UE speed, soft boundary ---
    r0 = radius * np.sqrt(rng.uniform(0, 1, n_ue))
    a0 = rng.uniform(0, 2 * np.pi, n_ue)
    xy = np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], axis=1)
    heading = rng.uniform(0, 2 * np.pi, n_ue)
    speed_kmh = rng.uniform(sim.min_speed_kmh, sim.max_speed_kmh, n_ue)
    step_m = speed_kmh / 3.6 * sim.dt_s

    pos = np.empty((n_ue, n_steps, 2))
    for t in range(n_steps):
        pos[:, t] = xy
        heading += rng.normal(0, 0.05, n_ue)
        outside = np.linalg.norm(xy, axis=1) > radius
        if outside.any():
            back = np.arctan2(-xy[outside, 1], -xy[outside, 0])
            heading[outside] = back + rng.normal(0, 0.5, outside.sum())
        xy = xy + step_m[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    return _channel(pos, speed_kmh, sites, rng, sim)


def _channel(pos, speed_kmh, sites, rng, sim: SimConfig, route_id=None) -> Episode:
    n_ue, n_steps, n_cells = pos.shape[0], pos.shape[1], len(sites)

    # --- large-scale channel: 3GPP macro path loss + distance-correlated shadowing ---
    d_km = np.maximum(np.linalg.norm(pos[:, :, None, :] - sites[None, None], axis=-1), 10.0) / 1000.0
    pathloss = 128.1 + 37.6 * np.log10(d_km)

    if sim.shadowing == "spatial":
        # tied to places: the same for every UE and drive at a given spot (city.py)
        from .city import spatial_shadowing
        shadow = spatial_shadowing(pos, sim)
    else:
        # AR(1) along each UE's path, decorrelating with distance travelled (Gudmundson)
        if sim.mobility == "random":
            step_m = speed_kmh / 3.6 * sim.dt_s
            a_t = np.broadcast_to(np.exp(-step_m / sim.shadow_decorr_m)[:, None, None], (n_ue, n_steps, 1))
        else:
            moved = np.linalg.norm(np.diff(pos, axis=1, append=pos[:, -1:]), axis=-1)
            a_t = np.exp(-moved / sim.shadow_decorr_m)[:, :, None]
        shadow = np.empty((n_ue, n_steps, n_cells))
        s = rng.normal(0, sim.shadow_sigma_db, (n_ue, n_cells))
        for t in range(n_steps):
            a = a_t[:, t]
            shadow[:, t] = s
            s = a * s + np.sqrt(1 - a ** 2) * rng.normal(0, sim.shadow_sigma_db, (n_ue, n_cells))

    rsrp_true = sim.tx_power_dbm - pathloss - shadow
    rsrp_inst = rsrp_true + rng.normal(0, sim.fading_sigma_db, rsrp_true.shape)

    # --- UE measurement + layer-3 filtering ---
    raw = rsrp_inst + rng.normal(0, sim.meas_sigma_db, rsrp_true.shape)
    meas = np.empty_like(raw)
    meas[:, 0] = raw[:, 0]
    for t in range(1, n_steps):
        meas[:, t] = (1 - sim.l3_alpha) * meas[:, t - 1] + sim.l3_alpha * raw[:, t]

    return Episode(sites, pos, speed_kmh, rsrp_true, rsrp_inst, meas, sim, route_id=route_id)


def save_episode(path: str, ep: Episode, **extra) -> None:
    """Store a drive's channel as float32 arrays (plus any `extra` arrays)."""
    np.savez_compressed(path, sites=ep.sites, pos=ep.pos.astype(np.float32),
                        speed_kmh=ep.speed_kmh.astype(np.float32),
                        rsrp_true=ep.rsrp_true.astype(np.float32), rsrp_inst=ep.rsrp_inst.astype(np.float32),
                        rsrp_meas=ep.rsrp_meas.astype(np.float32), **extra)


UNMEASURED_DBM = -160.0    # stands in for NaN (cell not measured) in logged traces


def load_episode(path: str, sim: SimConfig | None = None) -> Episode:
    """Load a drive saved by :func:`save_episode`, or a real trace in the same layout.

    Required: ``rsrp_meas`` (U, T, C), the per-UE RSRP of every cell in dBm at
    `dt_s` (100 ms) steps; NaN = not measured. Cells are indexed 0..C-1
    (C <= 64); keep your own PCI / NR-CGI mapping alongside.

    Optional:

    * ``rsrp_true``: what the teacher uses to look ahead (defaults to ``rsrp_meas``).
    * ``rsrp_inst``: used for SINR (defaults to ``rsrp_meas``).
    * ``speed_kmh`` (U,), ``pos`` (U, T, 2) and ``sites`` (C, 2).
    * ``serving`` and ``sinr_db`` (U, T): the serving cell and its SINR at each
      report, as logged by the network. With ``serving``, a corpus can be built
      from the states the real network actually visited.
    """
    from .tokenizer import MAX_CELLS

    d = np.load(path)
    get = lambda k: np.nan_to_num(d[k].astype(np.float64), nan=UNMEASURED_DBM) if k in d else None
    meas = get("rsrp_meas")
    U, T, C = meas.shape
    if C > MAX_CELLS:
        raise ValueError(f"{path}: {C} cells, the tokenizer supports at most {MAX_CELLS}")
    serving = d["serving"].astype(np.int64) if "serving" in d else None
    if serving is not None and (serving.shape != (U, T) or serving.min() < 0 or serving.max() >= C):
        raise ValueError(f"{path}: 'serving' must be (U, T) cell indices in [0, {C})")
    true, inst = get("rsrp_true"), get("rsrp_inst")
    return Episode(sites=d["sites"].astype(np.float64) if "sites" in d else np.zeros((C, 2)),
                   pos=d["pos"].astype(np.float64) if "pos" in d else np.zeros((U, T, 2)),
                   speed_kmh=d["speed_kmh"].astype(np.float64) if "speed_kmh" in d else np.zeros(U),
                   rsrp_true=meas if true is None else true, rsrp_inst=meas if inst is None else inst,
                   rsrp_meas=meas, sim=sim or SimConfig(), logged_serving=serving,
                   logged_sinr_db=d["sinr_db"].astype(np.float64) if "sinr_db" in d else None)


# ---------------------------------------------------------------------------
# Measurement reports
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Batched measurement report, one row per UE (what an xApp receives)."""

    serving: np.ndarray        # (U,)
    serving_hist: np.ndarray   # (U, H) filtered RSRP, oldest first
    nbr_ids: np.ndarray        # (U, K) strongest neighbours, strongest first
    nbr_hist: np.ndarray       # (U, K, H)
    sinr_db: np.ndarray        # (U,)
    speed_kmh: np.ndarray      # (U,)
    # Optional location context (ai_ran_llm.location.compute_context); NaN = unknown. Keys:
    # dist_ratio (U,K), radial_speed (U,K+1: serving first), map_gain_now / map_gain_ahead (U,K),
    # next_prob (U,K), next_count (U,)
    context: dict | None = None


def history_steps(t: int, obs: ObsConfig) -> np.ndarray:
    idx = t - obs.hist_stride * np.arange(obs.hist_len - 1, -1, -1)
    return np.clip(idx, 0, None)


def build_observation(ep: Episode, t: int, serving: np.ndarray, obs: ObsConfig) -> Observation:
    u = np.arange(ep.n_ue)
    hist = ep.rsrp_meas[:, history_steps(t, obs), :]           # (U, H, C)
    cur = ep.rsrp_meas[:, t, :].copy()
    cur[u, serving] = -np.inf
    nbr = np.argsort(-cur, axis=1)[:, : obs.n_neighbors]      # (U, K)
    nbr_hist = np.take_along_axis(hist.transpose(0, 2, 1), nbr[:, :, None], axis=1)
    return Observation(
        serving=serving.copy(),
        serving_hist=hist[u, :, serving],
        nbr_ids=nbr,
        nbr_hist=nbr_hist,
        sinr_db=ep.sinr_db(t, serving),
        speed_kmh=ep.speed_kmh.copy(),
    )


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    ue_seconds: float = 0.0
    handovers: int = 0
    ping_pongs: int = 0
    returns_5s: int = 0            # HO back to the previous cell within 5 s (ping-pong in a wider window)
    rlf: int = 0
    ho_failures: int = 0
    sinr_sum: float = 0.0
    se_sum: float = 0.0
    outage_steps: int = 0
    samples: int = 0

    def merge(self, other: "Metrics") -> "Metrics":
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def summary(self) -> dict:
        per_min = 60.0 / max(self.ue_seconds, 1e-9)
        return {
            "ho_per_ue_min": self.handovers * per_min,
            "ping_pong_pct": 100.0 * self.ping_pongs / max(self.handovers, 1),
            "return_5s_pct": 100.0 * self.returns_5s / max(self.handovers, 1),
            "rlf_per_ue_min": self.rlf * per_min,
            "hof_per_ue_min": self.ho_failures * per_min,
            "mean_sinr_db": self.sinr_sum / max(self.samples, 1),
            "mean_se_bps_hz": self.se_sum / max(self.samples, 1),
            "outage_pct": 100.0 * self.outage_steps / max(self.samples, 1),
        }


def run_policy(ep: Episode, policy, obs_cfg: ObsConfig, record: bool = False):
    """Replay `ep` with `policy` controlling handovers.

    `policy.decide(ep, t, obs)` returns an int array (U,) with the target cell or
    -1 to stay. Returns (Metrics, serving trajectory (U, T): the serving cell when
    each step's report was taken, before that step's decision) and, if `record`,
    also the list of (t, Observation) pairs seen by the policy.
    """
    sim = ep.sim
    U, T = ep.n_ue, ep.n_steps
    u = np.arange(U)
    serving = ep.rsrp_meas[:, 0, :].argmax(axis=1)
    prev_cell = np.full(U, -1)
    last_ho = np.full(U, -10**9)
    oos = np.zeros(U, dtype=int)
    blocked_until = np.zeros(U, dtype=int)   # outage after RLF / HOF (re-establishment)
    recovery_steps = int(round(1.0 / sim.dt_s))
    interrupt = sim.ho_interruption_s / sim.dt_s

    if hasattr(policy, "reset"):
        policy.reset(ep, obs_cfg)
    m = Metrics(ue_seconds=U * T * sim.dt_s)
    traj = np.empty((U, T), dtype=int)
    seen = []

    for t in range(T):
        sinr = ep.sinr_db(t, serving)
        active = blocked_until <= t

        # radio link monitoring (T310)
        oos = np.where(active & (sinr < sim.q_out_db), oos + 1, 0)
        rlf = oos >= sim.t310_steps
        if rlf.any():
            m.rlf += int(rlf.sum())
            serving = np.where(rlf, ep.rsrp_meas[:, t, :].argmax(axis=1), serving)
            blocked_until = np.where(rlf, t + recovery_steps, blocked_until)
            prev_cell = np.where(rlf, -1, prev_cell)
            oos[rlf] = 0

        obs = build_observation(ep, t, serving, obs_cfg)
        traj[:, t] = serving
        if record:
            seen.append((t, obs))
        target = np.asarray(policy.decide(ep, t, obs))
        do_ho = (target >= 0) & (target != serving) & (blocked_until <= t)

        se = np.log2(1.0 + 10.0 ** (sinr / 10.0))
        se = np.where(blocked_until > t, 0.0, se)
        if do_ho.any():
            # the HO command needs several TTIs on the old link: judge it on large-scale SINR
            hof = do_ho & (ep.sinr_db(t, serving, large_scale=True) < sim.hof_sinr_db)
            back = do_ho & (target == prev_cell)
            pp = back & (t - last_ho <= sim.ping_pong_steps)
            m.handovers += int(do_ho.sum())
            m.ping_pongs += int(pp.sum())
            m.returns_5s += int((back & (t - last_ho <= int(round(5.0 / sim.dt_s)))).sum())
            m.ho_failures += int(hof.sum())
            se = np.where(do_ho, se * (1 - interrupt), se)
            prev_cell = np.where(do_ho, serving, prev_cell)
            serving = np.where(do_ho, target, serving)
            last_ho = np.where(do_ho, t, last_ho)
            blocked_until = np.where(hof, t + recovery_steps, blocked_until)
            if hasattr(policy, "on_handover"):
                policy.on_handover(do_ho)

        m.sinr_sum += float(np.where(blocked_until > t, sim.q_out_db - 10, sinr).sum())
        m.se_sum += float(se.sum())
        m.outage_steps += int(((blocked_until > t) | (sinr < sim.q_out_db)).sum())
        m.samples += U

    return (m, traj, seen) if record else (m, traj)
