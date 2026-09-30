"""Tests for the city model (road mobility, spatial shadowing) and the default simulator."""

import hashlib

import numpy as np
import pytest

from ai_ran_llm.city import FIELD_RES_M, ROUTE_TYPES, city_routes, shadow_fields, spatial_shadowing
from ai_ran_llm.config import ObsConfig, SimConfig
from ai_ran_llm.policies import A3Policy
from ai_ran_llm.simulator import generate_episode, run_policy

CITY = SimConfig(mobility="roads", shadowing="spatial")


def test_default_simulator_unchanged():
    """Fingerprint of the original simulator: the committed corpus and every published number
    depend on it. If this fails, the default random stream or channel model changed."""
    ep = generate_episode(4, 50, np.random.default_rng(0))
    assert ep.route_id is None
    assert hashlib.sha256(np.round(ep.rsrp_meas, 6).tobytes()).hexdigest()[:16] == "f6203ed6212b4944"


def test_spatial_field_statistics():
    f = shadow_fields(CITY)
    assert f.shape == (19, 256, 256)
    assert abs(f.std() - CITY.shadow_sigma_db) < 0.3
    lag = int(CITY.shadow_decorr_m / FIELD_RES_M)
    rho = np.mean([np.corrcoef(f[c, :, :-lag].ravel(), f[c, :, lag:].ravel())[0, 1] for c in range(3)])
    assert abs(rho - np.exp(-1)) < 0.08                     # exponential correlation at the decorr distance
    assert not np.allclose(shadow_fields(SimConfig(shadowing="spatial", map_seed=2))[0], f[0])


def test_shadowing_is_tied_to_places_across_drives():
    ep1 = generate_episode(3, 20, np.random.default_rng(1), SimConfig(shadowing="spatial"))
    ep2 = generate_episode(3, 20, np.random.default_rng(2), SimConfig(shadowing="spatial"))
    p = np.stack([ep1.pos[0, 5], ep2.pos[1, 7]])[None]       # any positions: lookups are deterministic
    np.testing.assert_allclose(spatial_shadowing(p, CITY), spatial_shadowing(p.copy(), CITY))
    # two UEs at the same place see the same shadowing, even in different drives
    same = np.array([[[100.0, 200.0], [100.0, 200.0]]])
    s = spatial_shadowing(same, CITY)
    np.testing.assert_allclose(s[0, 0], s[0, 1])


def test_road_mobility():
    routes = city_routes(CITY)
    assert len(routes) == CITY.n_routes and routes == city_routes(CITY)          # deterministic city
    assert abs(sum(r.weight for r in routes) - 1) < 1e-9
    ep = generate_episode(24, 400, np.random.default_rng(5), CITY)
    assert ep.route_id.shape == (24,) and ep.pos.shape == (24, 400, 2)
    sp = CITY.road_spacing_m
    for u in range(24):
        r = routes[ep.route_id[u]]
        lo, hi = ROUTE_TYPES[r.kind][0]
        assert lo <= ep.speed_kmh[u] <= hi
        if r.kind != "highway":                              # grid routes stay on street lines
            g = ep.pos[u] / sp
            assert (np.minimum(np.abs(g[:, 0] - np.round(g[:, 0])), np.abs(g[:, 1] - np.round(g[:, 1]))) < 1e-6).all()
    v = np.linalg.norm(np.diff(ep.pos, axis=1), axis=-1) / CITY.dt_s * 3.6
    assert (v <= ep.speed_kmh[:, None] + 1e-6).all()          # never faster than cruise speed
    assert (v < 0.1).any()                                   # some stops (lights, route ends)
    assert len(set(ep.route_id.tolist())) < 24                # popular routes are shared


def test_city_closed_loop_runs_and_rejects_unknown_models():
    ep = generate_episode(8, 100, np.random.default_rng(0), CITY)
    m, _ = run_policy(ep, A3Policy(), ObsConfig())
    assert m.samples == 800 and np.isfinite(m.summary()["mean_sinr_db"])
    with pytest.raises(ValueError):
        generate_episode(2, 5, np.random.default_rng(0), SimConfig(mobility="teleport"))
    with pytest.raises(ValueError):
        generate_episode(2, 5, np.random.default_rng(0), SimConfig(shadowing="none"))
