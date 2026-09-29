import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import torch

from ai_ran_llm.cli import EXAMPLE_REPORT
from ai_ran_llm.config import ModelConfig, ObsConfig
from ai_ran_llm.dataset import export_jsonl, export_raw, generate_dataset
from ai_ran_llm.inference import HandoverLLM, LLMPolicy, report_to_observation
from ai_ran_llm.model import HandoverGPT
from ai_ran_llm.policies import A3Policy, OraclePolicy, label_decision, oracle_decision
from ai_ran_llm.serve import make_handler
from ai_ran_llm.simulator import build_observation, generate_episode, hex_sites, load_episode, run_policy
from ai_ran_llm.tokenizer import HandoverTokenizer
from ai_ran_llm.train import make_batch


def _episode(n_ue=8, n_steps=120, seed=0):
    return generate_episode(n_ue, n_steps, np.random.default_rng(seed))


def _tiny_llm():
    tok = HandoverTokenizer()
    torch.manual_seed(0)
    model = HandoverGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=63, n_layer=1, n_head=2, n_embd=32))
    return HandoverLLM(model, tok)


def test_hex_grid_has_19_cells():
    assert hex_sites(2, 500.0).shape == (19, 2)


def test_episode_shapes_and_ranges():
    ep = _episode()
    assert ep.rsrp_meas.shape == (8, 120, 19)
    assert -160 < ep.rsrp_true.min() and ep.rsrp_true.max() < -20
    sinr = ep.sinr_db(0, ep.rsrp_meas[:, 0].argmax(1))
    assert sinr.shape == (8,) and np.isfinite(sinr).all()


def test_observation_excludes_serving_and_is_sorted():
    ep, obs_cfg = _episode(), ObsConfig()
    serving = np.zeros(ep.n_ue, dtype=int)
    obs = build_observation(ep, 50, serving, obs_cfg)
    assert obs.nbr_ids.shape == (ep.n_ue, obs_cfg.n_neighbors)
    assert not (obs.nbr_ids == 0).any()
    last = obs.nbr_hist[:, :, -1]
    assert (np.diff(last, axis=1) <= 1e-9).all()
    np.testing.assert_allclose(obs.serving_hist[:, -1], ep.rsrp_meas[:, 50, 0])


def test_oracle_targets_are_reported_neighbours():
    ep, obs_cfg = _episode(), ObsConfig()
    obs = build_observation(ep, 30, ep.rsrp_meas[:, 30].argmin(1), obs_cfg)   # worst cell -> must hand over
    target, gain = oracle_decision(ep, 30, obs, obs_cfg)
    assert (target >= 0).all()
    assert all(t in row for t, row in zip(target, obs.nbr_ids))
    assert (gain > obs_cfg.oracle_margin_db).all()


def test_smoothed_labels():
    ep = _episode(32, 200, seed=3)
    raw_cfg = ObsConfig(label_window=0, label_confirm_horizon=0)
    serving = ep.rsrp_meas[:, 50].argmax(1)
    obs = build_observation(ep, 50, serving, raw_cfg)
    raw, _ = oracle_decision(ep, 50, obs, raw_cfg)
    same, _ = label_decision(ep, 50, obs, raw_cfg)
    np.testing.assert_array_equal(raw, same)                  # no smoothing == raw teacher
    confirmed, _ = label_decision(ep, 50, obs, ObsConfig(label_window=0, label_confirm_horizon=20))
    assert set(np.nonzero(confirmed >= 0)[0]) <= set(np.nonzero(raw >= 0)[0])   # confirm only removes HOs
    early, _ = label_decision(ep, 50, obs, ObsConfig(label_window=3, label_confirm_horizon=0))
    assert set(np.nonzero(raw >= 0)[0]) <= set(np.nonzero(early >= 0)[0])       # window only adds HOs
    assert (early[raw >= 0] == raw[raw >= 0]).all()


def test_closed_loop_metrics():
    ep, obs_cfg = _episode(16, 300), ObsConfig()
    m_a3, traj = run_policy(ep, A3Policy(), obs_cfg)
    m_or, _ = run_policy(ep, OraclePolicy(obs_cfg), obs_cfg)
    assert traj.shape == (16, 300)
    assert m_a3.samples == m_or.samples == 16 * 300
    assert m_or.summary()["mean_se_bps_hz"] >= m_a3.summary()["mean_se_bps_hz"] - 0.05


def test_tokenizer_layout_and_answer():
    tok = HandoverTokenizer()
    ep, obs_cfg = _episode(), ObsConfig()
    obs = build_observation(ep, 40, np.zeros(ep.n_ue, dtype=int), obs_cfg)
    prompts = tok.encode_prompts(obs)
    assert prompts.shape == (ep.n_ue, tok.prompt_len)
    assert (prompts[:, 0] == tok.BOS).all() and (prompts[:, -1] == tok.ANS).all()
    tgt = int(obs.nbr_ids[0, 1])
    ans = tok.encode_answer(tgt, 4.2, obs.serving_hist[0], obs.nbr_ids[0], obs.nbr_hist[0], -3.0)
    text = tok.decode(ans)
    assert text.startswith(f"<ho> C{tgt} <why> serving") and "G+4" in text and text.endswith("low_sinr <eos>")
    info = tok.explain(ans)
    assert info["action"] == "HANDOVER" and info["target_cell"] == tgt and "+4 dB" in info["rationale"]


def test_dataset_and_jsonl(tmp_path):
    d = generate_dataset(1, n_ue=6, n_steps=100, seed=1, verbose=False)
    tok = HandoverTokenizer()
    P = int(d["prompt_len"])
    assert d["tokens"].shape[1] == 64
    assert set(np.unique(d["tokens"][:, P])) <= {tok.STAY, tok.HO}
    n = export_jsonl(d["tokens"], P, str(tmp_path / "x.jsonl"), limit=5)
    rows = [json.loads(line) for line in open(tmp_path / "x.jsonl")]
    assert n == 5 and rows[0]["messages"][2]["role"] == "assistant"


def test_export_raw_matches_corpus(tmp_path):
    import gzip
    n = export_raw(str(tmp_path), n_episodes=2, n_ue=4, n_steps=80, seed=5, verbose=False)
    corpus = generate_dataset(2, n_ue=4, n_steps=80, seed=5, verbose=False)
    assert n["in_corpus"] == len(corpus["tokens"])            # same drives, same subsampling
    recs = [json.loads(line) for line in gzip.open(tmp_path / "reports.jsonl.gz", "rt")]
    assert len(recs) == n["reports"] and sum(r["in_corpus"] for r in recs) == n["in_corpus"]
    r = recs[0]
    assert len(r["serving_rsrp"]) == 5 and len(r["neighbors"]) == 4 and r["label"]["action"] in ("STAY", "HANDOVER")
    ep = load_episode(str(tmp_path / "drive_001.npz"))
    assert ep.rsrp_meas.shape == (4, 80, 19)
    assert json.load(open(tmp_path / "drives.json"))["drives"][1]["file"] == "drive_001.npz"
    obs = report_to_observation(r, ObsConfig())                 # reports are valid xApp requests
    assert obs.nbr_ids.shape == (1, 4)


def test_loss_masking_and_training_step():
    tok = HandoverTokenizer()
    d = generate_dataset(1, n_ue=6, n_steps=100, seed=2, verbose=False)
    tokens = torch.from_numpy(d["tokens"][:64])
    P = int(d["prompt_len"])
    x, y, w = make_batch(tokens, P, tok.PAD, 0.1)
    assert (w[:, : P - 1] == 0.1).all() and (w[y == -100] == 0).all()
    llm = _tiny_llm()
    model = llm.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        _, loss = model(x, y, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


def test_constrained_generation_only_targets_reported_cells():
    llm = _tiny_llm()
    obs = report_to_observation(EXAMPLE_REPORT, ObsConfig())
    prompt = llm.tok.encode_prompts(obs)[0]
    out = llm.generate(prompt)
    assert out[0] in (llm.tok.STAY, llm.tok.HO)
    if out[0] == llm.tok.HO:
        assert llm.tok.itos[out[1]] in {f"C{n['cell_id']}" for n in EXAMPLE_REPORT["neighbors"]}
    p_stay, p_ho = llm.score(llm.tok.encode_prompts(obs))
    assert np.isclose(p_stay[0] + p_ho[0].sum(), 1.0, atol=1e-5)


def test_handle_report_and_policy():
    llm = _tiny_llm()
    res = llm.handle_report(EXAMPLE_REPORT)
    assert res["action"] in ("HANDOVER", "STAY") and 0 <= res["confidence"] <= 1
    # an untrained model is unsure -> A3 fallback; cell 4 beats serving by >2 dB in the last 2 reports
    res = llm.handle_report(EXAMPLE_REPORT, min_confidence=0.99, a3_hyst_db=2.0, a3_ttt=2)
    assert res["source"] == "a3_fallback" and res["action"] == "HANDOVER" and res["target_cell"] == 4
    ep = _episode(4, 40)
    m, _ = run_policy(ep, LLMPolicy(llm), ObsConfig())
    assert m.samples == 160


def test_http_endpoint():
    llm = _tiny_llm()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(llm, 0.5))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/handover"
        req = urllib.request.Request(url, json.dumps(EXAMPLE_REPORT).encode(), {"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["ue_id"] == "ue-42" and body["action"] in ("HANDOVER", "STAY")
    finally:
        server.shutdown()
