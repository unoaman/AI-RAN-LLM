"""Tests for location context: tokenizer layout, Location service, corpus, closed loop, report API."""

import json

import numpy as np
import pytest
import torch

from ai_ran_llm.config import ModelConfig, ObsConfig, SimConfig
from ai_ran_llm.dataset import generate_dataset, prompt_to_text
from ai_ran_llm.inference import HandoverLLM, report_to_observation, stack_observations
from ai_ran_llm.location import (LocationAwarePolicy, LocationConfig, LocationService, RadioMap, TransitionModel,
                                 compute_context, position_track, previous_cells)
from ai_ran_llm.model import HandoverGPT
from ai_ran_llm.simulator import build_observation, generate_episode, run_policy
from ai_ran_llm.tokenizer import HandoverTokenizer
from ai_ran_llm.train import train

CITY = SimConfig(mobility="roads", shadowing="spatial")
ALL = ObsConfig(use_position=True, use_radio_map=True, use_trajectory=True)


@pytest.fixture(scope="module")
def service():
    return LocationService.build(CITY, n_drives=3, n_ue=16, n_steps=200, seed=777)


def _tiny(obs_cfg):
    tok = HandoverTokenizer(obs_cfg)
    torch.manual_seed(0)
    model = HandoverGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=tok.prompt_len + 16, n_layer=1,
                                    n_head=2, n_embd=32))
    return HandoverLLM(model, tok)


def test_tokenizer_layouts():
    base = HandoverTokenizer()
    assert base.vocab_size == 349 and base.prompt_len == 41          # unchanged without context
    full = HandoverTokenizer(ALL)
    assert full.vocab_size == 447 and full.prompt_len == 63
    assert full.itos[:349] == base.itos                              # context tokens are appended only
    assert HandoverTokenizer(ObsConfig(use_trajectory=True)).prompt_len == 41 + 1 + 4
    ep = generate_episode(3, 30, np.random.default_rng(0))
    obs = build_observation(ep, 20, ep.rsrp_meas[:, 20].argmax(1), ObsConfig())
    p = full.encode_prompts(obs)                                    # no context -> all <unk>
    assert p.shape == (3, 63) and (p == full.UNK).sum() == 3 * (2 + 4 * 5)
    assert list(np.nonzero(p[0] == full.NBR)[0]) == [14, 26, 38, 50]
    assert "unknown" in prompt_to_text(full, p[0])


def test_quantisers_and_unknowns():
    t = HandoverTokenizer(ALL)
    assert t.itos[int(t.ratio(0.25))] == "L+2" and t.itos[int(t.radial(-9.0))] == "S-2"
    assert t.itos[int(t.mapgain(99))] == "M+20" and t.itos[int(t.prob(0.74))] == "P7"
    assert t.itos[int(t.count(0))] == "N0" and t.itos[int(t.count(1000))] == "N6"
    assert int(t.mapgain(np.nan)) == t.UNK


def test_radio_map_and_transitions(service):
    cfg = LocationConfig()
    ep = generate_episode(4, 200, np.random.default_rng(1), CITY)
    rm = RadioMap(ep.n_cells, cfg)
    rm.add(ep, ep.pos)                                              # exact positions
    cells = np.argmax(ep.rsrp_meas[:, 100], axis=1)[:, None]
    v = rm.lookup(cells, ep.pos[:, 100])
    assert np.isfinite(v).all() and np.abs(v[:, 0] - ep.rsrp_meas[np.arange(4), 100, cells[:, 0]]).max() < 10
    assert np.isnan(rm.lookup(cells[:1], np.array([[5000.0, 5000.0]]))).all()   # outside the map
    tm = TransitionModel(cfg)
    tm.add(np.array([[0, 0, 1, 1, 2, 2, 1, 1, 2], [3, 1, 1, 0, 0, 0, 0, 0, 0]]))   # 0>1>2>1>2 and 3>1>0
    p, n = tm.probs(np.array([-1, 0]), np.array([1, 1]), np.array([[2, 0], [2, 0]]))
    assert n[0] == 3 and np.allclose(p[0], [2 / 3, 1 / 3])        # no previous cell: P(next | current)
    assert n[1] == 3 and np.allclose(p[1], [2 / 3, 1 / 3])        # (0, 1) seen once < 5: back-off
    assert np.isnan(tm.probs(np.array([-1]), np.array([7]), np.array([[1, 2]]))[0]).all()
    assert service.radio_map.coverage() > 0 and service.transitions.t2


def test_service_roundtrip(service, tmp_path):
    service.save(str(tmp_path))
    back = LocationService.load(str(tmp_path))
    assert np.allclose(back.radio_map.cnt, service.radio_map.cnt) and back.sim == service.sim
    assert back.transitions.to_dict() == service.transitions.to_dict()


def test_context_values(service):
    ep = generate_episode(6, 120, np.random.default_rng(2), CITY)
    _, traj = run_policy(ep, LocationAwarePolicy(_tiny(ALL), service), ObsConfig())
    t = 60
    obs = build_observation(ep, t, traj[:, t], ObsConfig())
    exact = position_track(ep, np.random.default_rng(0), LocationConfig(pos_sigma_m=0))
    ctx = compute_context(ep, t, obs, exact, previous_cells(traj)[:, t], service, LocationConfig())
    d_s = np.linalg.norm(ep.pos[:, t] - ep.sites[obs.serving], axis=1) + 1
    d_n = np.linalg.norm(ep.pos[:, t] - ep.sites[obs.nbr_ids[:, 0]], axis=1) + 1
    assert np.allclose(ctx["dist_ratio"][:, 0], np.log10(d_s / d_n))
    assert np.abs(ctx["radial_speed"]).max() <= ep.speed_kmh.max() / 3.6 + 1e-6    # exact track: physical speeds
    assert set(ctx) == {"dist_ratio", "radial_speed", "map_gain_now", "map_gain_ahead", "next_prob", "next_count"}


def test_context_corpus_and_training(service, tmp_path):
    d = generate_dataset(1, n_ue=6, n_steps=120, seed=3, sim=CITY, obs_cfg=ALL, location=service, verbose=False)
    tok = HandoverTokenizer(ALL)
    assert d["tokens"].shape[1] >= tok.prompt_len + 12 and int(d["prompt_len"]) == 63
    assert json.loads(str(d["obs_cfg"]))["use_radio_map"] is True
    plain = generate_dataset(1, n_ue=6, n_steps=120, seed=3, sim=CITY, verbose=False)
    assert "obs_cfg" not in plain and len(plain["tokens"]) == len(d["tokens"])      # same drives and labels
    assert np.array_equal(d["tokens"][:, 63:75][:, 0], plain["tokens"][:, 41])       # same first answer token
    with pytest.raises(ValueError):
        generate_dataset(1, n_ue=2, n_steps=60, sim=CITY, obs_cfg=ALL, verbose=False)   # service missing
    path = tmp_path / "ctx.npz"
    np.savez(path, **d)
    _, stats = train(str(path), str(tmp_path / "m.pt"), epochs=1, batch_size=64,
                     model_cfg=ModelConfig(n_layer=1, n_head=2, n_embd=32), log_every=10**9)
    llm = HandoverLLM.load(str(tmp_path / "m.pt"))
    assert llm.tok.obs_cfg.use_trajectory and llm.tok.prompt_len == 63


def test_closed_loop_policy_and_report_api(service):
    llm = _tiny(ALL)
    ep = generate_episode(4, 80, np.random.default_rng(4), CITY)
    m, _ = run_policy(ep, LocationAwarePolicy(llm, service), ObsConfig())
    assert m.samples == 4 * 80
    report = {"serving_cell": 0, "serving_rsrp": [-95] * 5, "sinr_db": -2,
              "context": {"radial_speed": 6.0, "next_count": 40},
              "neighbors": [{"cell_id": 1, "rsrp": [-92] * 5,
                             "context": {"dist_ratio": 0.2, "radial_speed": -8.0, "map_gain_now": 3.0,
                                         "map_gain_ahead": 5.0, "next_prob": 0.8}},
                            {"cell_id": 2, "rsrp": [-99] * 5}]}
    obs = report_to_observation(report, ALL)
    assert obs.context["next_prob"][0, 0] == 0.8 and np.isnan(obs.context["next_prob"][0, 1])
    assert obs.context["radial_speed"][0, 0] == 6.0 and obs.context["radial_speed"][0, 1] == -8.0
    ans = llm.handle_report(report)
    assert ans["action"] in ("HANDOVER", "STAY")
    both = stack_observations([obs, report_to_observation({k: v for k, v in report.items() if k != "context"}, ALL)])
    assert both.context["next_prob"].shape == (2, 4)
