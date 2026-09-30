"""The "city": road-network mobility and location-tied shadowing.

Two opt-in upgrades to the simulator (``SimConfig.mobility = "roads"``,
``SimConfig.shadowing = "spatial"``). The original simulator draws a fresh
shadowing process along every UE path and moves UEs randomly, so neither a
radio map nor trajectory mining can help there. These upgrades make both
measurable:

* **Road mobility.** A street grid plus a few highways inside the service
  area; ``n_routes`` fixed routes (street, highway, walk) with Zipf-like
  popularity, so UEs repeat the same paths (commuting patterns). Speed depends
  on the route type; street UEs stop at intersections (traffic lights), and UEs
  turn back at route ends.
* **Spatial shadowing.** One Gaussian random field per cell, σ =
  ``shadow_sigma_db``, exponential spatial correlation with distance
  ``shadow_decorr_m`` (Gudmundson), sampled on a 10 m grid by circulant
  embedding and interpolated at UE positions. The field depends only on the
  place, so every UE at the same spot sees the same shadowing, in every drive.

Everything about the city (roads, routes, popularity, fields) is drawn from
``map_seed`` with its own random generator, so it never touches the episode's
random stream: the same city appears in every episode, like a real network.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import SimConfig

ROUTE_TYPES = {                      # speed range (km/h), share of routes
    "street": ((20.0, 50.0), 0.6),
    "highway": ((70.0, 120.0), 0.2),
    "walk": ((3.0, 6.0), 0.2),
}
FIELD_RES_M = 10.0
FIELD_N = 256                        # 256 x 10 m = 2.56 km square, centred on the network


@dataclass(frozen=True)
class Route:
    kind: str                        # street | highway | walk
    xy: np.ndarray                   # (P, 2) waypoints
    cum: tuple                       # cumulative length at each waypoint (m)
    weight: float                    # popularity

    @property
    def length(self) -> float:
        return self.cum[-1]

    def point(self, s: float) -> tuple[float, float]:
        return (float(np.interp(s, self.cum, self.xy[:, 0])), float(np.interp(s, self.cum, self.xy[:, 1])))


def _radius(sim: SimConfig) -> float:
    return sim.rings * sim.isd_m * 0.85


def _grid_walk(rng, sim: SimConfig, n_blocks: int) -> np.ndarray | None:
    R, sp = _radius(sim), sim.road_spacing_m
    k = int(R // sp)
    nodes = [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if math.hypot(i * sp, j * sp) <= R]
    cur = nodes[rng.integers(len(nodes))]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d = dirs[rng.integers(4)]
    path = [cur]
    for _ in range(n_blocks):
        options = [d] * 3 + [(-d[1], d[0]), (d[1], -d[0])]      # prefer straight, sometimes turn
        rng.shuffle(options)
        for nd in options:
            nxt = (cur[0] + nd[0], cur[1] + nd[1])
            if math.hypot(nxt[0] * sp, nxt[1] * sp) <= R and nxt not in path[-2:]:
                cur, d = nxt, nd
                path.append(cur)
                break
        else:
            break
    if len(path) < 3:
        return None
    return np.array(path, dtype=float) * sp


def _highway(rng, sim: SimConfig) -> np.ndarray:
    R = _radius(sim)
    theta = rng.uniform(0, math.pi)
    u = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-u[1], u[0]])
    off = rng.uniform(-0.6 * R, 0.6 * R)
    half = math.sqrt(R * R - off * off)
    return np.stack([off * n - half * u, off * n + half * u])


@lru_cache(maxsize=8)
def _routes_cached(key) -> tuple[Route, ...]:
    sim = SimConfig(**dict(key))
    rng = np.random.default_rng([sim.map_seed, 101])
    kinds = []
    for kind, (_, share) in ROUTE_TYPES.items():
        kinds += [kind] * max(1, round(share * sim.n_routes))
    routes = []
    for kind in kinds[: sim.n_routes]:
        xy = None
        while xy is None:
            if kind == "highway":
                xy = _highway(rng, sim)
            else:
                xy = _grid_walk(rng, sim, int(rng.integers(8, 21)) if kind == "street" else int(rng.integers(2, 6)))
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        routes.append((kind, xy, tuple(np.concatenate([[0.0], np.cumsum(seg)]).tolist())))
    ranks = rng.permutation(len(routes))
    weights = 1.0 / (ranks + 1.0)                      # Zipf: a few routes carry most UEs
    weights = weights / weights.sum()
    return tuple(Route(k, xy, cum, float(w)) for (k, xy, cum), w in zip(routes, weights))


def _key(sim: SimConfig, fields: tuple[str, ...]) -> tuple:
    return tuple((f, getattr(sim, f)) for f in fields)


def city_routes(sim: SimConfig) -> tuple[Route, ...]:
    """The city's routes (deterministic in ``map_seed`` and the geometry parameters)."""
    return _routes_cached(_key(sim, ("rings", "isd_m", "map_seed", "road_spacing_m", "n_routes")))


def road_mobility(n_ue: int, n_steps: int, rng: np.random.Generator, sim: SimConfig):
    """UEs driving / walking along the city's routes.

    Returns (pos (U, T, 2), cruise speed km/h (U,), route index (U,)).
    """
    routes = city_routes(sim)
    w = np.array([r.weight for r in routes])
    route_id = rng.choice(len(routes), size=n_ue, p=w)
    pos = np.empty((n_ue, n_steps, 2))
    speed = np.empty(n_ue)
    for u in range(n_ue):
        r = routes[route_id[u]]
        lo, hi = ROUTE_TYPES[r.kind][0]
        speed[u] = rng.uniform(lo, hi)
        step = speed[u] / 3.6 * sim.dt_s
        s = rng.uniform(0, r.length)
        direction = 1.0 if rng.uniform() < 0.5 else -1.0
        stop_p = sim.stop_prob if r.kind == "street" else (sim.stop_prob / 2 if r.kind == "walk" else 0.0)
        stopped = 0
        inner = r.cum[1:-1]
        for t in range(n_steps):
            pos[u, t] = r.point(s)
            if stopped > 0:
                stopped -= 1
                continue
            s_new = s + direction * step
            if stop_p > 0 and inner:
                # intersection crossed this step: in (s, s_new] forwards, [s_new, s) backwards, so a
                # UE standing on an intersection does not stop there again
                if direction > 0:
                    i = bisect.bisect_right(inner, s)
                    crossed = i < len(inner) and inner[i] <= s_new
                else:
                    i = bisect.bisect_left(inner, s) - 1
                    crossed = i >= 0 and inner[i] >= s_new
                if crossed and rng.uniform() < stop_p:
                    s_new = inner[i]
                    stopped = int(rng.uniform(2.0, 20.0) / sim.dt_s)
            if s_new >= r.length or s_new <= 0.0:          # end of route: turn back
                s_new = min(max(s_new, 0.0), r.length)
                direction = -direction
                stopped = max(stopped, int(rng.uniform(0.0, 5.0) / sim.dt_s))
            s = s_new
    return pos, speed, route_id


@lru_cache(maxsize=8)
def _fields_cached(key) -> np.ndarray:
    sim = SimConfig(**dict(key))
    from .simulator import hex_sites
    n_cells = len(hex_sites(sim.rings, sim.isd_m))
    rng = np.random.default_rng([sim.map_seed, 202])
    idx = np.minimum(np.arange(FIELD_N), FIELD_N - np.arange(FIELD_N)) * FIELD_RES_M
    r = np.hypot(idx[:, None], idx[None, :])
    cov = sim.shadow_sigma_db ** 2 * np.exp(-r / sim.shadow_decorr_m)
    spec = np.sqrt(np.clip(np.fft.fft2(cov).real, 0.0, None))
    fields = np.empty((n_cells, FIELD_N, FIELD_N), dtype=np.float32)
    for c in range(n_cells):
        w = rng.standard_normal((FIELD_N, FIELD_N))
        fields[c] = np.fft.ifft2(spec * np.fft.fft2(w)).real
    return fields


def shadow_fields(sim: SimConfig) -> np.ndarray:
    """(C, N, N) shadowing fields in dB on a 10 m grid centred on (0, 0)."""
    return _fields_cached(_key(sim, ("rings", "isd_m", "map_seed", "shadow_sigma_db", "shadow_decorr_m")))


def spatial_shadowing(pos: np.ndarray, sim: SimConfig) -> np.ndarray:
    """Bilinear interpolation of every cell's field at positions (..., 2) -> (..., C)."""
    f = shadow_fields(sim)
    g = pos / FIELD_RES_M + FIELD_N / 2
    i0 = np.floor(g).astype(int)
    fr = g - i0
    i0 %= FIELD_N
    i1 = (i0 + 1) % FIELD_N
    x0, y0, x1, y1 = i0[..., 0], i0[..., 1], i1[..., 0], i1[..., 1]
    fx, fy = fr[..., 0:1], fr[..., 1:2]
    v00 = np.moveaxis(f[:, x0, y0], 0, -1)
    v10 = np.moveaxis(f[:, x1, y0], 0, -1)
    v01 = np.moveaxis(f[:, x0, y1], 0, -1)
    v11 = np.moveaxis(f[:, x1, y1], 0, -1)
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy).astype(np.float64)
