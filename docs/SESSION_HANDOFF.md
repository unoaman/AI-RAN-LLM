# Session handoff: complete project knowledge

Everything needed to continue this project on a new machine, in a new session, by a human or an
AI agent. It collates the project state, key results, decisions, in-flight work, conventions and
the operational lessons from the build session. Detailed documents:

| Document | Contents |
|---|---|
| `README.md` | What it is, quick start, results, data, real-RAN summary |
| `docs/DESIGN.md` | Full design: decisions (D1–D20), algorithms, interfaces, file/function reference, complete development history (§21), invariants (§22) |
| `docs/RAN_INTEGRATION.md` | Real-RAN integration: protocol, RRC parsing, OCUDU / srsRAN and OAI setup, guard-rail results, rollout checklist |
| `docs/LOCATION_AWARE_HANDOVER.md` | Location, radio map, trajectory mining: concept, city simulator, learnability results, implementation, closed-loop results (§13) |
| `docs/SESSION_HANDOFF.md` | This file |

- **Repository:** `unoaman/AI-RAN-LLM`, branch `claude/llm-ai-ran-mobility-handover-sip5ax`
  (no PR opened yet).
- **Original session:** https://claude.ai/code/session_013cWuVVEiRkubHR8oXGvv5s. Open it while
  signed in to the same claude.ai account. The conversation persists, but the cloud machine does
  not: only pushed work survives.

---

## 1. Resume on a new machine

```bash
git clone -b claude/llm-ai-ran-mobility-handover-sip5ax https://github.com/unoaman/AI-RAN-LLM.git
cd AI-RAN-LLM
python3 -m venv .venv && source .venv/bin/activate          # Python 3.10+ (built with 3.11)
pip install -e ".[dev]"                                     # numpy, torch (CPU or CUDA), pytest
pytest -q                                                   # expect 42 passed, about 10 s
python -m ai_ran_llm evaluate --episodes 1                  # smoke test with the shipped model
```

* **Hardware:** a CPU is enough (built on 4 cores, 15 GB RAM, no GPU). A GPU is used
  automatically by `train` and makes it much faster.
* **No Git LFS:** all committed binaries are ordinary files (the largest is the 25 MB corpus).
* **Scripts outside the package** (`experiments/`, `integrations/`) need `pip install -e .` or
  `PYTHONPATH=.`.

---

## 2. The project in one page

A **small domain-specific language model for AI-RAN handover**. It reads a UE measurement report
as tokens and generates stay / hand over to cell X plus a rationale. It runs as a near-RT RIC
xApp.

```
simulator (19 cells, 3GPP-style channel, UE mobility)
   → look-ahead teacher labels (best reported neighbour over the next 1 s, 2 dB margin)
   → 349-token domain language, 41-token prompt
   → 0.85 M-param GPT (4 layers, 4 heads, 128-d), trained on CPU
   → decision: P(handover to best neighbour) ≥ 0.35, grammar-constrained
      (only reported cells), A3 fallback / override, guard rails
   → closed-loop benchmark vs 3GPP A3 on identical drives
   → real RAN: RRC reports in, E2SM-RC / OAI telnet / srsRAN-OCUDU console out
```

**Key results** (simulation only; full tables in the docs):

| Result | Numbers | Source |
|---|---|---|
| Shipped model vs A3, original simulator (5 × 64 UEs × 60 s) | SE 2.969 b/s/Hz vs 2.959 (A3 1 dB), 2.919 (A3 2 dB); outage 1.9 % vs 2.2 % / 3.8 %; HOF 0.53 vs 0.69 / 1.22 per UE-min; ping-pong 21 % vs 31 % / 13 % | README Results |
| Gap to the non-causal teacher | teacher SE 3.006, outage 0.56 %, ping-pong 5 % | README |
| Real-RAN path (fake gNB over TCP) | Reproduces the benchmark **exactly** (identical trajectories); realistic reporting (200 ms, 8 neighbours, RRC-quantised) costs SE −0.025 | `RAN_INTEGRATION.md` §6 |
| Guard rails | `--confirm 2`: ping-pong 21 → 8 %, but outage 1.9 → 3.7 %; `--hold-off-s 1` raises RLF 0.006 → 0.34. Defaults: confirm 1, hold-off 0, A3 override 6 dB (free in-distribution) | `RAN_INTEGRATION.md` §6 |
| Location learnability, city, 3 seeds (recall @ 50 % precision) | radio-map forecast **+78 %**, position + heading +27 %, trajectory prior (no location) +16 %, all +83 %; target accuracy 82 → 90 % | `LOCATION_AWARE_HANDOVER.md` §9 |
| City models, validation after 1 epoch (HO recall) | base 0.341, +position 0.406, +radio map 0.488, trajectory / all: see §4 | this file, §4 |
| Negative results | Label smoothing (window / confirm) did not help; delta tokens did not raise per-sample recall; per-sample accuracy is misleading (always-STAY ≈ 88–90 %) | `DESIGN.md` §21.8 |

---

## 3. Repository map

```
ai_ran_llm/
  config.py        SimConfig (+ city options), ObsConfig (+ location flags), ModelConfig
  simulator.py     channel, mobility, reports, run_policy (closed loop), KPIs, drive I/O
  city.py          opt-in road mobility + spatial shadowing (--mobility roads --shadowing spatial)
  policies.py      A3, look-ahead teacher, label smoothing, OraclePolicy
  dataset.py       corpus generation (+ location context), raw export, SFT JSONL, from real traces
  tokenizer.py     349 base tokens; +98 location-context tokens when enabled (447)
  model.py         HandoverGPT
  train.py         training (reads ObsConfig from context corpora), per-epoch checkpoints
  inference.py     HandoverLLM: score, decide_reports, constrained generation, report_to_observation
  location.py      position track, RadioMap, TransitionModel, LocationService, LocationAwarePolicy
  evaluate.py      closed-loop benchmark
  serve.py         HTTP endpoint
  cli.py           gen-data, train, evaluate, infer, serve, export-raw, export-jsonl, ran-xapp,
                   fake-gnb, ran-parse, build-location-service
  ran/             real-RAN integration: messages (ran-bridge protocol), rrc (TS 38.331/38.133
                   parsing), cells, tracker, controller (guard rails), actuators (E2SM-RC, OAI
                   telnet, console, command), sources (log tail), bridge, runtime, app, fake_gnb
integrations/      OCUDU/srsRAN + OAI example configs, O-RAN SC RIC xApp, bridge-agent template
experiments/       location learnability studies, closed-loop location study, logs/
tests/             test_pipeline (14), test_ran (16), test_city (5), test_location (7) = 42
checkpoints/       handover_llm.pt (shipped), city_base / city_position / city_radio_map.pt
data/              handover_corpus.npz (527 512 samples), raw/ (drive 0), city/location_service/
docs/              DESIGN, RAN_INTEGRATION, LOCATION_AWARE_HANDOVER, SESSION_HANDOFF
```

---

## 4. In-flight work: closed-loop location experiment

`experiments/location_closed_loop.py` builds a location service from 20 history drives (seed
777). It then builds 5 city corpora with identical drives and labels (seed 0, 40 drives × 32 UEs,
368 624 samples each), trains one model per corpus (1 epoch, same architecture), and benchmarks
them in the city (seed 10000, 5 × 64 UEs × 60 s) against A3, the shipped model and the teacher.

| Model | Prompt | Val decision acc | Val HO recall | Checkpoint |
|---|---:|---:|---:|---|
| base (report only) | 41 | 0.914 | 0.341 | `checkpoints/city_base.pt` (committed) |
| + position | 50 | 0.919 | 0.406 | `checkpoints/city_position.pt` (committed) |
| + radio map | 49 | 0.927 | 0.488 | `checkpoints/city_radio_map.pt` (committed) |
| + trajectory | 46 | in training at handoff | | not yet |
| all | 63 | pending | | not yet |

* **Committed so far:** the three finished checkpoints, the location service
  (`data/city/location_service/`) and the partial log
  (`experiments/logs/location_closed_loop.partial.log`).
* **Not committed:** the corpora (17–21 MB each), which are fully reproducible from the seeds.

**To finish on any machine** (the script skips the service and models that already exist, so
only the missing corpora, the trajectory and all models, and the benchmark run):

```bash
PYTHONPATH=. python experiments/location_closed_loop.py      # → data/city/closed_loop.json
```

Then:
1. Fill `CLOSED_LOOP_RESULTS` in `docs/LOCATION_AWARE_HANDOVER.md` §13 from
   `data/city/closed_loop.json` (use `ai_ran_llm.evaluate.format_table`).
2. Add the headline to README and to `docs/DESIGN.md` §21.15.
3. Commit the new checkpoints (`checkpoints/city_*.pt` is allowed by `.gitignore`).

A missing corpus is regenerated identically: same seed, same bytes.

---

## 5. Decisions worth knowing before changing anything

* **Small from-scratch GPT, not a fine-tuned general LLM:** latency (about 2 ms per report),
  on-prem, a numeric domain. A general-LLM path exists via `export-jsonl` (D1).
* **Imitation of a non-causal teacher:** the future is known in replay; a behaviour-policy mix
  (DAgger-like) covers off-optimal states (D3).
* **Neighbour RSRP as deltas to serving (D tokens); threshold 0.35, not argmax** (D5, D8). The
  threshold mattered more than any other change.
* **Evaluate in closed loop on identical drives;** per-sample accuracy misleads (D10).
* **Real RAN:** a vendor-neutral core with thin adapters; parse the real RRC
  MeasurementReport (E2SM-KPM lacks neighbour RSRP); shadow mode by default; keep the RAN's own
  A3 as a backstop; guard timers on the report clock (D15–D20).
* **Location:** location derived only from the same RSRP reports adds nothing. Gains need new
  measurements or cross-UE memory (radio map, route statistics). The original simulator could not
  show this, hence the opt-in city model.

---

## 6. Conventions and invariants (`docs/DESIGN.md` §22)

* **Opt-in only:** new simulator or model features must not change the defaults:
  * the default simulator fingerprint (`tests/test_city.py::test_default_simulator_unchanged`);
  * the default tokenizer (349 tokens / 41-token prompt);
  * the committed corpus (regenerates bit-for-bit with `gen-data`).
* **Tokenizer:** vocabulary order is a checkpoint contract; only append new families.
* **Real-RAN regression test:** the fake gNB with guard rails off must equal the offline
  benchmark (`test_fake_gnb_end_to_end_equals_offline_policy`).
* **Random streams:** do not reorder random-number use in `iter_labelled_drives`,
  `_label_steps` or `generate_episode`.
* **Commits:** after changes run `pytest -q` (and `evaluate` for model changes). Commit
  messages end with the attribution lines. Never put model identifiers in commits or docs.
* **Honesty rules:** document negative results; say when something is simulation-only or
  untested on real hardware; re-check claims against the numbers (two README claims had to be
  corrected during the session).

---

## 7. Operational lessons from the build session

| Lesson | Detail |
|---|---|
| Long jobs get killed | A backgrounded tool command hit a time limit; a container restart killed another run. Run long jobs detached (`setsid nohup … &`); save checkpoints every epoch (implemented); make experiment scripts resumable (implemented) |
| CPU speed | About 0.56 s per training step (batch 256) on 4 cores, so 1 epoch of 350k samples is about 13 min, and 2 epochs of 527k about 37 min. Contention from parallel jobs slows everything |
| `pkill -f pattern` | Also matched the shell running it and killed the command; use the PID instead |
| Mermaid diagrams | Validate before committing: `npm i @mermaid-js/mermaid-cli` (with `PUPPETEER_SKIP_DOWNLOAD=1`) and `mmdc -p puppeteer.json` pointing at a local Chromium. Quote labels with brackets or parentheses; no semicolons in sequence-diagram messages; avoid self-loops with long labels (a user-reported overlap) |
| Network | docs.srsran.com, srsran.com and docs.ocudu.org were blocked from the build environment; GitHub and raw.githubusercontent.com worked. OCUDU / srsRAN console and config facts partly come from search snippets and should be re-checked against current docs |
| Clean tree | The environment's stop hook requires committing and pushing before ending a turn |
| Scripts outside the package | Need `PYTHONPATH=.` or `pip install -e .` |

---

## 8. Open next steps (priority order)

1. **Finish §4** (closed-loop location results) and decide whether context tokens improve
   closed-loop KPIs: recall, outage and HOF without raising RLF or ping-pong.
2. **Location in the RAN path:** let `ai_ran_llm/ran/tracker.py` pass Location-xApp context into
   reports (`report_to_observation` already accepts it).
3. **Realistic positioning:** derive position from simulated AoA / TA; add a Kalman filter
   (radial speed is noisy today, about ±14 m/s at 10 m error).
4. **Model:** model-side hysteresis against ping-pong; iterative DAgger; RL fine-tuning; more
   epochs or data for the city models (1 epoch was used for speed).
5. **Real network:** shadow-mode trial on OCUDU / OAI (`docs/RAN_INTEGRATION.md` §10);
   retrain on real traces (`gen-data --from-drives`).
6. **Housekeeping:** open a PR when ready (none exists yet); pass `ObsConfig` explicitly through
   the CLI for non-default report shapes.

---

## 9. Conversation timeline (user requests → outcome)

1. Create an LLM for AI-RAN mobility handover → simulator, tokenizer, GPT, training, benchmark,
   xApp endpoint.
2. Training status / where files live → explained the cloud container; committed model and data
   later.
3. Less-noisy labels → tried smoothing (negative result); the threshold fix (0.35) was the real
   gain.
4. Commit training data; show the reports behind it → corpus, `export-raw`, readable reports,
   `--from-drives` for real traces.
5. Detailed design document with diagrams, using the chat history → `docs/DESIGN.md`; fixed an
   overlapping diagram.
6. Integrate with a real RAN (OCUDU / OAI) → `ai_ran_llm/ran/`, `integrations/`,
   `docs/RAN_INTEGRATION.md`; found and fixed clock, padding and out-of-distribution issues.
7. Would location / AoA help; a helper xApp? → concept + learnability study (no code changes, as
   asked) → `docs/LOCATION_AWARE_HANDOVER.md`.
8. Trajectory mining? → concept; then implemented the city simulator and measured it (radio map
   +78 %).
9. Make the model use location → context tokens, `location.py`, the closed-loop experiment (§4).
10. Save the session for another machine → this handoff; committed the finished experiment
    artifacts.
