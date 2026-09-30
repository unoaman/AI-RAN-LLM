# Session handoff: where the project stands

Use this file to resume work in a new session, on any machine, by a human or an AI agent. It
summarises the state at the end of the build session, what is in flight, and how to continue.
Background: `README.md` (usage), `docs/DESIGN.md` (full design and history),
`docs/RAN_INTEGRATION.md` (real RAN), `docs/LOCATION_AWARE_HANDOVER.md` (location / trajectory
work).

- **Branch:** `claude/llm-ai-ran-mobility-handover-sip5ax` (repo `unoaman/AI-RAN-LLM`)
- **Original session:** https://claude.ai/code/session_013cWuVVEiRkubHR8oXGvv5s (open it while
  signed in to the same claude.ai account; the conversation history persists, but the cloud
  machine does not, so only pushed work survives)

## 1. What exists (all pushed)

| Area | State | Where |
|---|---|---|
| HandoverLLM (0.85 M-param GPT) | Trained on the original simulator; beats all A3 settings on SE / outage / HOF at threshold 0.35 | `checkpoints/handover_llm.pt`, README Results |
| Training data | 527 512-sample corpus + raw drive 0; regenerates bit-for-bit | `data/handover_corpus.npz`, `data/raw/` |
| Real-RAN integration | ran-bridge protocol, RRC MeasurementReport parser, tracker, controller + guard rails + A3 override, actuators (E2SM-RC, OAI telnet, srsRAN/OCUDU console, command); fake gNB reproduces the benchmark exactly | `ai_ran_llm/ran/`, `integrations/`, `docs/RAN_INTEGRATION.md` |
| City simulator | Opt-in road mobility + location-tied shadowing; defaults unchanged (fingerprint test) | `ai_ran_llm/city.py` |
| Location context | Opt-in tokens for position, radio map and trajectory prior; Location service; `LocationAwarePolicy` | `ai_ran_llm/location.py`, `docs/LOCATION_AWARE_HANDOVER.md` §12 |
| Learnability evidence | City: radio-map forecast +78 % recall at 50 % precision, position +27 %, trajectory prior (no location) +16 % | `docs/LOCATION_AWARE_HANDOVER.md` §9, `experiments/` |
| Tests | 42 passing (`pytest -q`) | `tests/` |

## 2. In flight when this file was written

**Closed-loop location experiment** (`experiments/location_closed_loop.py`). It trains 5 models on
identical city drives (base, +position, +radio map, +trajectory, all) and benchmarks them in
closed loop. Its outputs (`data/city/`, `checkpoints/city_*.pt`) are git-ignored, so they are
lost if the machine is reclaimed.

Validation scores so far (1 epoch each, 368 624 samples, same drives and labels):

| Model | Prompt tokens | Val decision acc | Val HO recall (argmax) |
|---|---:|---:|---:|
| base (report only) | 41 | 0.914 | 0.341 |
| + position | 50 | 0.919 | 0.406 |
| + radio map | 49 | 0.927 | 0.488 |
| + trajectory | 46 | (training) | |
| all | 63 | (pending) | |

The closed-loop results table in `docs/LOCATION_AWARE_HANDOVER.md` §13 still says
`CLOSED_LOOP_RESULTS`. This is the placeholder to fill.

**To finish it** (about 1.5 h on 4 CPU cores from scratch; the script skips steps whose outputs
exist):

```bash
pip install -e .[dev]
PYTHONPATH=. python experiments/location_closed_loop.py      # writes data/city/closed_loop.json
```

Then:
* put the table into §13 of `docs/LOCATION_AWARE_HANDOVER.md`;
* update README / DESIGN (§21.15) with the headline;
* consider committing the best city checkpoint (3–4 MB).

## 3. Open next steps (from the docs)

1. Finish §13 above. Decide whether location / map / trajectory context improves closed-loop
   KPIs: recall, outage and HOF without raising RLF or ping-pong.
2. Let the RAN tracker (`ai_ran_llm/ran/tracker.py`) carry location context from a Location xApp
   into reports (`report_to_observation` already accepts it).
3. Derive positions from simulated AoA / TA instead of noisy true positions; add a tracking
   (Kalman) filter to reduce radial-speed noise.
4. Model-side hysteresis to cut ping-pong; iterative DAgger; RL fine-tuning.
5. Shadow-mode trial on an OCUDU / OAI testbed (`docs/RAN_INTEGRATION.md` §10).

## 4. Conventions to keep (details: `docs/DESIGN.md` §22)

* New simulator or model features are **opt-in**. The default simulator fingerprint, the default
  tokenizer (349 tokens / 41-token prompt) and the committed corpus must not change.
* The fake gNB with guard rails off must equal the offline benchmark exactly.
* Guard timers use the report clock.
* Run `pytest -q` and `python -m ai_ran_llm evaluate` after changes; commit messages end with the
  session attribution lines.
