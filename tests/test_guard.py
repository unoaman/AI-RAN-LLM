"""Tests for model-side hysteresis: per-neighbour thresholds and ReturnGuard."""

import numpy as np
import torch

from ai_ran_llm.config import ModelConfig, ObsConfig
from ai_ran_llm.inference import HandoverLLM, LLMPolicy, ReturnGuard
from ai_ran_llm.model import HandoverGPT
from ai_ran_llm.simulator import Observation, build_observation, generate_episode, run_policy
from ai_ran_llm.tokenizer import HandoverTokenizer


def _tiny():
    tok = HandoverTokenizer()
    torch.manual_seed(0)
    return HandoverLLM(HandoverGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=tok.prompt_len + 16,
                                               n_layer=1, n_head=2, n_embd=32)), tok)


def _obs(serving, nbr_ids, serving_rsrp, nbr_rsrp, sinr):
    U, K = np.shape(nbr_ids)
    return Observation(serving=np.array(serving), serving_hist=np.tile(np.array(serving_rsrp, float)[:, None], 5),
                       nbr_ids=np.array(nbr_ids), nbr_hist=np.tile(np.array(nbr_rsrp, float)[:, :, None], 5),
                       sinr_db=np.array(sinr, float), speed_kmh=np.full(U, 30.0))


def test_per_neighbour_thresholds():
    llm = _tiny()
    ep = generate_episode(16, 40, np.random.default_rng(0))
    obs = build_observation(ep, 30, ep.rsrp_meas[:, 30].argmax(1), ObsConfig())
    _, p_ho = llm.score(llm.tok.encode_prompts(obs))
    thr = 0.5 * p_ho.max(axis=1).min()                 # every UE hands over at this threshold
    t_scalar, _ = llm.decide_batch(obs, thr)
    t_array, _ = llm.decide_batch(obs, np.full(obs.nbr_ids.shape, thr))
    assert np.array_equal(t_scalar, t_array) and (t_scalar >= 0).all()
    k = p_ho.argmax(axis=1)
    block = np.full(obs.nbr_ids.shape, thr)
    block[np.arange(16), k] = np.inf                   # best neighbour blocked -> next best that clears thr
    t_blocked, _ = llm.decide_batch(obs, block)
    second = np.where(block < np.inf, p_ho, -1).argmax(axis=1)
    expect = np.where(p_ho[np.arange(16), second] >= thr, obs.nbr_ids[np.arange(16), second], -1)
    assert np.array_equal(t_blocked, expect)


def test_return_guard_thresholds():
    g = ReturnGuard(window_s=2.0, threshold=0.8, margin_db=3.0, rescue_sinr_db=-6.0)
    g.reset()
    nbrs = [[0, 2], [0, 2], [0, 2], [0, 2]]
    g.thresholds(0.0, _obs([0, 0, 0, 1], nbrs, [-90] * 4, [[-95, -99]] * 4, [5] * 4), 0.5)
    # UEs 0-2 hand over 0 -> 1 at t = 0.5 s; UE 3 stays on 1 (never left cell 0)
    g.thresholds(0.5, _obs([1, 1, 1, 1], [[0, 2]] * 4, [-90] * 4, [[-95, -99]] * 4, [5] * 4), 0.5)
    thr = g.thresholds(1.0, _obs([1, 1, 1, 1], nbrs, [-90] * 4,
                                 [[-85, -99], [-89, -99], [-89, -99], [-85, -99]], [5, 5, -8, 5]), 0.5)
    assert thr[0].tolist() == [0.8, 0.5]               # return, margin 5 dB >= 3: stricter threshold
    assert thr[1].tolist() == [np.inf, 0.5]            # return, margin 1 dB < 3: blocked
    assert thr[2].tolist() == [0.5, 0.5]               # serving SINR below rescue level: no guard
    assert thr[3].tolist() == [0.5, 0.5]               # cell 0 is not UE 3's previous cell
    late = g.thresholds(3.0, _obs([1] * 4, nbrs, [-90] * 4, [[-89, -99]] * 4, [5] * 4), 0.5)
    assert (late == 0.5).all()                         # outside the 2 s window


def test_guard_in_closed_loop():
    llm = _tiny()
    ep = generate_episode(8, 120, np.random.default_rng(1))
    m0, _ = run_policy(ep, LLMPolicy(llm, 0.0), ObsConfig())          # hands over at every report
    never = ReturnGuard(window_s=5.0, threshold=1.1, rescue_sinr_db=-np.inf)      # no return within 5 s at all
    m1, _ = run_policy(ep, LLMPolicy(llm, 0.0, never), ObsConfig())
    assert m0.returns_5s > 0 and m1.returns_5s == 0 and m1.ping_pongs == 0 and m0.samples == m1.samples
    s = m0.summary()
    assert "return_5s_pct" in s and s["return_5s_pct"] >= s["ping_pong_pct"]
