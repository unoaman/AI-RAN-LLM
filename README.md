# AI-RAN-LLM: HandoverLLM

A small, domain-specific **language model for AI-RAN mobility management**. It reads
UE measurement reports as token sequences and generates a **handover decision**,
a **target cell** and a short **natural-language rationale**. It is meant to run as
a near-RT RIC xApp (or an AI-RAN inference service next to the gNB).

```
<bos> <spd> V9 <sinr> Q-1 <srv> C9 R-88 R-90 R-92 R-95 R-97
<nbr> C4 D-11 D-6 D-2 D+3 D+8  <nbr> C10 D-9 D-7 D-6 D-3 D-2  ...  <ans>
                                  ──▶  <ho> C4 <why> serving falling neighbor rising gain G+6 low_sinr <eos>
```

> "Hand over to cell 4: serving cell falling, neighbor cell rising, predicted RSRP gain +6 dB
> over the next second, serving SINR is low."

> **Full design and development record:** [`docs/DESIGN.md`](docs/DESIGN.md). It covers the
> architecture and interface diagrams, algorithms, assumptions, the reasoning behind each
> decision (and alternatives rejected), a per-function reference, the artifacts, and the
> complete experiment history.
>
> **Concept: location-aware handover** (position from AoA / RTT plus a radio-map "Location xApp"):
> [`docs/LOCATION_AWARE_HANDOVER.md`](docs/LOCATION_AWARE_HANDOVER.md). It includes simulator
> evidence (a radio-map forecast raises recall by about 78 % in the simulated city), trajectory
> mining, and a validation plan. The simulator's opt-in "city" model (`--mobility roads
> --shadowing spatial`) makes these measurable. The model can take location, radio-map and
> trajectory inputs as optional context tokens. In closed loop in the city, context cuts handover
> failures by about half and outage by a third versus the same model without it. Trained for 3
> epochs, the radio-map model also beats the shipped model and A3 on handover failures (−47 % and
> −85 % vs A3 2 dB), outage and SE (§13 and §13.1 of that document). With the ping-pong guard below, it beats A3 at
> 2 dB on every KPI in the city, including ping-pong (7.2 % vs 8.3 %; §13.2).

## Why an LLM for handover?

Classical 3GPP handover (event **A3**: neighbour > serving + hysteresis for time-to-trigger)
is reactive and has one static trade-off. Small hysteresis/TTT means **ping-pongs**. Large
hysteresis/TTT means late handovers, **radio link failures (RLF)** and **HO failures**.
HandoverLLM is trained to *anticipate* the radio conditions of the next second from the
RSRP trajectories in the report:

* **Proactive:** it imitates a non-causal look-ahead teacher that knows which cell will
  be best over the next 1 s.
* **Explainable:** every decision includes a rationale (signal trends, predicted gain,
  low-SINR flag), which operators can audit.
* **Safe by construction:** grammar-constrained decoding means it can only output
  `<stay>` or `<ho>` to a cell **that the UE actually reported**. If its confidence is
  below a threshold, the xApp falls back to a conservative A3 rule.
* **LLM-ecosystem friendly:** the same corpus exports as chat-format JSONL, so you can
  fine-tune any open LLM (Llama, Qwen, Mistral…) with standard SFT/LoRA tooling.

## Training data

All training data is **simulated**; no real network data is used. The measurement reports
are produced by the simulator, labelled by a look-ahead teacher and tokenised:

```
simulator.generate_episode      19-cell channel + UE mobility, every 100 ms
        │                       (RSRP from every cell to every UE)
        ▼
simulator.run_policy            a behaviour policy (teacher / delayed teacher / random A3)
        │                       decides the serving cell; one measurement report per UE per step
        ▼
policies.label_decision         teacher label: stay, or hand over to cell X
        │                       (looks 1 s into the simulated future)
        ▼
dataset.generate_dataset        tokenise report + label ─▶ data/handover_corpus.npz
```

Two copies of the data are committed.

### 1. `data/handover_corpus.npz`: what the model trains on (25 MB)

This is the exact corpus the shipped checkpoint was trained on.

* 527,512 samples from 60 drives × 32 UEs × 60 s, generated with seed 0. 12.5% of the samples
  are handover labels.
* `tokens`: int64 array of shape (527512, 64). Each row holds the 41 report tokens followed by the
  answer, padded with `<pad>`.
* `prompt_len`: 41.
* **Rounded values:** RSRP and SINR are rounded to 1 dB, and neighbour RSRP is stored relative
  to the serving cell.
* **Some reports are left out:** reports where every neighbour is more than 6 dB weaker are
  trivially STAY, and only 10% of those are kept.

`python -m ai_ran_llm gen-data` regenerates this file bit-for-bit.

### 2. `data/raw/`: the raw simulation behind the corpus, human-readable (5 MB sample)

This is **drive 0 of the corpus** (32 UEs × 60 s), exported with
`python -m ai_ran_llm export-raw`. It uses the same seed and random stream as `gen-data`,
so these are exactly the drives the corpus came from.

| File | Contents |
|---|---|
| `reports.jsonl.gz` | 18,624 measurement reports (every UE, every labelled step) with unrounded values, in the **same JSON format the xApp endpoint accepts**. Each report includes the teacher's `label` and `in_corpus` (whether it was kept in the corpus). The 8,641 reports marked `in_corpus` are exactly this drive's samples in `handover_corpus.npz`. |
| `drive_000.npz` | The full channel of the drive (float32). `pos` (U,T,2) is UE position in m. `speed_kmh` (U,). `rsrp_true` (U,T,C) is large-scale RSRP, the teacher's view. `rsrp_inst` (U,T,C) adds fast fading and drives SINR. `rsrp_meas` (U,T,C) is the L3-filtered UE measurement, which is what the reports contain. `serving` and `sinr_db` (U,T) are the serving cell and its SINR at each report, under the behaviour policy. `sites` (C,2). |
| `drives.json` | Seed, simulator and report configuration, and the behaviour policy of each drive. |
| `cells.csv` | Cell id and site coordinates (m). |

One report from `reports.jsonl.gz`:

```json
{"episode": 0, "ue_id": "ue-23", "t": 14, "time_s": 1.4, "serving_cell": 18, "speed_kmh": 10.8, "sinr_db": 5.4,
 "serving_rsrp": [-85.7, -88.2, -87.1, -85.9, -83.9],
 "neighbors": [{"cell_id": 14, "rsrp": [-86.4, -87.1, -89.7, -89.9, -88.2]},
               {"cell_id": 10, "rsrp": [-92.2, -92.9, -90.5, -87.8, -91.2]},
               {"cell_id": 9,  "rsrp": [-94.5, -93.5, -91.8, -93.4, -91.6]},
               {"cell_id": 5,  "rsrp": [-99.0, -98.8, -98.8, -98.9, -96.1]}],
 "label": {"action": "HANDOVER", "target_cell": 14, "gain_db": 2.1,
           "rationale": "Hand over to cell 14: serving cell rising, neighbor cell falling, predicted RSRP gain +2 dB over the next second."},
 "in_corpus": true}
```

This example also shows why the task is hard. Right now, cell 14 is weaker than the serving
cell and falling. The teacher still says "hand over" because it can see that cell 14 will be
2 dB better over the next second. The shipped model stays (0.99 confidence), which is the
reasonable call from the report alone.

Reading the data:

```python
import gzip, json, numpy as np
from ai_ran_llm.simulator import load_episode

reports = [json.loads(l) for l in gzip.open("data/raw/reports.jsonl.gz", "rt")]
drive = np.load("data/raw/drive_000.npz")        # raw arrays
ep = load_episode("data/raw/drive_000.npz")      # as a simulator Episode (replay any policy on it)
```

Export more drives with `export-raw --episodes N` (about 5 MB per drive). For fine-tuning a
general-purpose LLM, `export-jsonl` writes the corpus as chat-format text.

### Using real network data

`gen-data --from-drives` builds a corpus from drive files instead of simulating. The files can
be real traces (drive tests, RIC/E2 KPM logs, MDT) or `export-raw` output. Each file is one
`.npz` with these arrays:

| Array | Shape | Required | Meaning |
|---|---|---|---|
| `rsrp_meas` | (U, T, C) | **yes** | L3-filtered RSRP (dBm) of every cell for every UE, one row per 100 ms step. `NaN` means not measured. |
| `serving` | (U, T) | for `--serving logged` | Serving cell index at each report. |
| `sinr_db` | (U, T) | no | Serving SINR at each report. If absent, it is estimated from RSRP. |
| `rsrp_true` | (U, T, C) | no | What the teacher uses to look 1 s ahead. Defaults to `rsrp_meas`. |
| `rsrp_inst` | (U, T, C) | no | Used to compute SINR. Defaults to `rsrp_meas`. |
| `speed_kmh`, `pos`, `sites` | (U,), (U,T,2), (C,2) | no | UE speed, UE position, cell site positions. |

Cells are indexed `0..C-1` with C ≤ 64. Keep your own index → PCI / NR-CGI mapping.
`data/raw/drive_000.npz` is a complete example of the layout.

```bash
# build a corpus from traces, using the serving cells the network actually used
python -m ai_ran_llm gen-data --from-drives traces/*.npz --out data/my_corpus.npz

# or replay the simulated behaviour policies (teacher / delayed teacher / random A3) on the traces' RSRP
python -m ai_ran_llm gen-data --from-drives traces/*.npz --serving replay --out data/my_corpus.npz

python -m ai_ran_llm train --data data/my_corpus.npz --out checkpoints/my_model.pt
```

How it works:

* **Labels still come from the look-ahead teacher.** In a log the future is known, so each
  report is labelled with the best reported neighbour over the following 1 s.
* **`--serving logged` (default)** uses the states your network actually visited. This is the
  most realistic option, but it only covers situations your current handover settings produced.
* **`--serving replay`** needs no `serving` array. It explores more states: late, early and
  missed handovers.

This path was checked end to end. Rebuilding a corpus from `data/raw/drive_000.npz` with its
logged serving cells reproduces all 569 handover labels and all 7,099 non-trivial samples of
drive 0 bit-for-bit. Only the random 10% subsample of trivial STAY reports differs.

A few things to watch for with real data:

* **Report timing:** resample to the model's report timing, 100 ms steps with 5 readings 200 ms
  apart. Otherwise, retrain with an `ObsConfig` that matches your reporting.
* **Large-scale RSRP:** with only measured RSRP, the teacher's view includes measurement noise.
  If you can, provide a smoothed `rsrp_true`.
* **Real SINR:** interference in real networks differs from the simulator, so pass logged
  `sinr_db` if you have it.
* **Evaluation:** `evaluate` still benchmarks on simulated drives. You can replay policies on a
  trace with `load_episode` + `run_policy`, but the handover outcomes (RLF, HOF) then come from
  the simulator's radio model applied to your RSRP.

## Components

| Module | What it does |
|---|---|
| `ai_ran_llm/simulator.py` | Vectorised 19-cell hexagonal macro network (ISD 500 m): 3GPP path loss `128.1+37.6·log10(d)`, distance-correlated shadowing, fast fading, L3 filtering, Gauss-Markov UE mobility at 3–120 km/h. Closed-loop evaluation with T310-based RLF, HO failure, ping-pong (return within 1 s), HO interruption, SINR and spectral efficiency. |
| `ai_ran_llm/policies.py` | 3GPP A3 baseline and the look-ahead teacher (`oracle_decision`) used for labels. |
| `ai_ran_llm/tokenizer.py` | 349-token domain vocabulary: quantised RSRP/SINR/speed/gain tokens, neighbour-minus-serving deltas, cell IDs, structure and rationale words. |
| `ai_ran_llm/dataset.py` | Builds the corpus by driving simulations with a mix of behaviour policies (teacher, delayed teacher, randomised A3). This covers both good and off-optimal serving states, in the spirit of DAgger. Also exports JSONL for fine-tuning a general-purpose LLM, and the raw drives and readable reports (`export_raw`). |
| `ai_ran_llm/model.py` | Decoder-only GPT (pre-LN, causal SDPA, tied embeddings). Default: 4 layers, 4 heads, 128-d, ~0.85 M parameters. It is small enough for near-RT (<10 ms) CPU inference. |
| `ai_ran_llm/train.py` | Causal-LM training. Full weight on answer tokens, and 0.1 weight on report tokens as a light telemetry-forecasting objective. |
| `ai_ran_llm/inference.py` | Scoring, grammar-constrained generation, confidence gating + A3 fallback, closed-loop policy. |
| `ai_ran_llm/serve.py` | HTTP endpoint (`POST /v1/handover`) for a RIC/xApp integration. |
| `ai_ran_llm/evaluate.py` | Closed-loop benchmark on identical drives: HandoverLLM vs A3 settings vs teacher. |
| `ai_ran_llm/ran/` | Real-RAN integration: RRC MeasurementReport parsing (JSON/XER, TS 38.133), ran-bridge protocol, cell map, per-UE tracker, controller with guard rails, actuators (E2SM-RC, OAI telnet, srsRAN/OCUDU console, command), RRC log source, xApp runtime, fake gNB. See `docs/RAN_INTEGRATION.md`. |
| `ai_ran_llm/city.py` | Opt-in "city" model for the simulator: road-network mobility with popular, repeated routes and stops, and location-tied (spatial) shadowing shared by all UEs and drives. `--mobility roads --shadowing spatial`. Defaults keep the original simulator bit-for-bit. |
| `ai_ran_llm/location.py` | Optional location context for the model: position track, radio map, handover-sequence mining (Location xApp logic), `LocationAwarePolicy`. Enabled with `gen-data --use-position --use-radio-map --use-trajectory`. |
| `experiments/` | Location / radio-map / trajectory learnability studies (`docs/LOCATION_AWARE_HANDOVER.md`). |
| `integrations/` | Example cell maps and RAN configs for OCUDU/srsRAN and OAI, the O-RAN SC RIC xApp, and a RAN-side bridge agent template. |

## Quick start

```bash
pip install -e .[dev]

# The trained checkpoint (checkpoints/handover_llm.pt) and the corpus it was trained on
# (data/handover_corpus.npz) are committed. Skip to step 2 to retrain on the shipped data,
# or to step 3 to use the shipped model. Step 1 regenerates the corpus exactly (seed 0).

# 1. simulate 60 drives x 32 UEs x 60 s and build the corpus (~0.5 M samples, ~1 min)
python -m ai_ran_llm gen-data --episodes 60 --ues 32

# 2. train (CPU is fine: ~0.85 M params, ~35 min for 2 epochs on 4 cores)
python -m ai_ran_llm train --epochs 2

# 3. closed-loop benchmark on unseen drives
python -m ai_ran_llm evaluate --episodes 5

# 4. one-off decision on a JSON report ('-' uses a built-in example)
python -m ai_ran_llm infer -

# 5. serve as an xApp endpoint
python -m ai_ran_llm serve --port 8080
curl -s localhost:8080/v1/handover -d '{
  "ue_id": "ue-42", "serving_cell": 9, "speed_kmh": 90, "sinr_db": -1.0,
  "serving_rsrp": [-88, -90, -92, -95, -97],
  "neighbors": [{"cell_id": 4,  "rsrp": [-99, -96, -94, -92, -89]},
                {"cell_id": 10, "rsrp": [-97, -97, -98, -98, -99]}]}'

# 6. optional: export chat-format SFT data to fine-tune a general-purpose LLM
python -m ai_ran_llm export-jsonl --limit 100000

# 7. optional: export raw drives + readable measurement reports (see Training data)
python -m ai_ran_llm export-raw --episodes 1 --out data/raw

# 8. optional: build a corpus from your own traces (see Using real network data)
python -m ai_ran_llm gen-data --from-drives traces/*.npz --out data/my_corpus.npz

# 9. real RAN: see "Integrating with a real RAN" and docs/RAN_INTEGRATION.md
python -m ai_ran_llm ran-xapp --cells sim --actuator bridge --live   # xApp (shadow mode without --live)
python -m ai_ran_llm fake-gnb --ues 16                               # simulated gNB over the bridge

pytest -q
```

Example response for the built-in report (`infer -`) from the committed checkpoint:

```json
{
  "action": "HANDOVER",
  "target_cell": 4,
  "rationale": "Hand over to cell 4: serving cell falling, neighbor cell rising, predicted RSRP gain +4 dB over the next second, serving SINR is low.",
  "confidence": 0.3882,
  "p_stay": 0.5638,
  "p_handover": {
    "4": 0.3882,
    "10": 0.0107,
    "3": 0.0013,
    "8": 0.0361
  },
  "source": "llm"
}
```

Cell 4 is rising while the serving cell falls, and it is already 8 dB above serving in the
latest sample. P(handover to cell 4) = 0.39 clears the 0.35 threshold, so the xApp hands over.
The model then writes the rationale for that decision. The endpoint and the closed-loop
benchmark use the same rule:

* `--ho-threshold` sets the handover decision.
* `--min-confidence` (default 0.3) routes low-probability decisions to the A3 fallback.

## Results

Closed-loop benchmark: 5 unseen drives × 64 UEs × 60 s (5.3 UE-hours), with identical
channels for every policy. The model is the default 0.85 M-parameter configuration, trained
for 2 epochs on 527k samples (about 37 min on 4 CPU cores). Reproduce with
`python -m ai_ran_llm evaluate --episodes 5` (default `--ho-threshold 0.35`).

| Policy | HO/UE/min | Ping-pong % | RLF/UE/min | HOF/UE/min | SINR dB | SE b/s/Hz | Outage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3 (1 dB, 200 ms) | 17.86 | 31.1 | 0.000 | 0.69 | 6.98 | 2.959 | 2.22 |
| A3 (2 dB, 300 ms) | 10.36 | 13.2 | 0.016 | 1.22 | 6.59 | 2.919 | 3.76 |
| A3 (3 dB, 500 ms) | 5.67 | 2.9 | 0.700 | 1.25 | 5.99 | 2.849 | 6.16 |
| **HandoverLLM** (threshold 0.35) | 13.34 | 21.4 | 0.006 | 0.53 | 7.04 | 2.969 | 1.92 |
| Teacher (non-causal upper bound) | 8.01 | 5.3 | 0.000 | 0.00 | 7.41 | 3.006 | 0.56 |

How to read this:

* **HandoverLLM beats every A3 setting on throughput, SINR, outage and HO failures.** Its
  spectral efficiency (2.969) and outage (1.9%) beat even aggressive A3 at 1 dB (2.959, 2.2%).
  It does this with 25% fewer handovers and a third fewer ping-pongs than A3 at 1 dB, and 23%
  fewer HO failures.
* **Against A3 at 2 dB:** it makes more handovers (13.3 vs 10.4 per UE-min) and more ping-pongs
  (21% vs 13%), and in return gets much less outage and less than half the HO failures.
* **The decision threshold is the main tuning knob** (`--ho-threshold`). The model hands over
  when P(handover to its best neighbour) ≥ threshold. Handovers are rare, so the model's
  probabilities are conservative, and 0.35 works better than 0.5:

  | Threshold | HO/UE/min | Ping-pong % | HOF/UE/min | SE b/s/Hz | Outage % |
  |---:|---:|---:|---:|---:|---:|
  | 0.25 | 22.39 | 39.3 | 0.71 | 2.962 | 2.15 |
  | **0.35** | 13.34 | 21.4 | 0.53 | 2.969 | 1.92 |
  | 0.50 | 8.69 | 9.0 | 0.95 | 2.936 | 3.07 |

  Use 0.5 if handover signalling load matters more than throughput.
* **Ping-pong guard (`--return-guard`).** Model-side hysteresis: for 2 s after a handover, going
  back to the cell just left needs P ≥ 0.9 and a 5 dB margin, unless serving SINR is below −6 dB
  (never hold a UE on a failing link). On these drives most returns are genuine rescues, so the
  default only trims ping-pong (21.4 → 18.7 %); `--guard-rescue-sinr-db -8` gives 12.5 %,
  below A3 at 2 dB, for a little more outage (1.92 → 2.46 %) and HOF (0.53 → 0.74). In the city
  simulator the guard costs nothing (19.6 → 7.2 % for the radio-map model). See
  `docs/DESIGN.md` §21.17.
* **There is still a clear gap to the teacher.** Per-sample validation accuracy is 88% and
  handover recall at argmax is 13%. The teacher reacts to future shadowing, much of which
  cannot be predicted from a 1 s report history.

### Label-smoothing experiment (negative result)

To make the teacher's labels easier to learn, `policies.label_decision` adds two options:

* `ObsConfig.label_window`: label a handover if the teacher would hand over within the next
  W steps.
* `ObsConfig.label_confirm_horizon`: keep a handover label only if the target still beats the
  serving cell over a longer horizon.

Neither helped, so both are off by default:

* **Windowing makes the teacher itself worse in closed loop.** Handing over early joins a cell
  that is not yet better. At W=5 ping-pongs reach 40%.
* **A 2 s confirmation makes a cleaner teacher** (ping-pong 1.3% vs 5.7%, same SE), but it is no
  more predictable from the report. A small classifier on report features reaches the same
  handover AUC (≈0.83) for every variant.
* **The model trained on confirmed labels was slightly worse** in closed loop (SE 2.966, outage
  2.1% at threshold 0.35).

Better gains will more likely come from richer inputs, such as beam/CSI measurements,
position and heading, or longer history, than from relabelling.

## Integrating with a real RAN

`ai_ran_llm/ran/` connects the model to a live gNB. **Full guide: [`docs/RAN_INTEGRATION.md`](docs/RAN_INTEGRATION.md).**
It covers message formats and parsing, the step-by-step setup for OCUDU / srsRAN Project and
OpenAirInterface, and the rollout checklist.

```
gNB (OCUDU / srsRAN / OAI)                          HandoverLLM xApp (python -m ai_ran_llm ran-xapp)
  UE RRC MeasurementReport ─▶ CU-CP log (JSON/XER) ─▶ RrcLogSource ─┐
  or a RAN-side agent ── ran-bridge NDJSON/TCP ────▶ BridgeServer ──┴▶ tracker ▶ model ▶ guard rails
  E2SM-RC Handover Control ◀── near-RT RIC xApp ◀──┐                                       │
  OAI telnet ci trigger_f1_ho / trigger_n2_ho ◀────┼──────────── actuator ◀───────────────┘
  srsRAN/OCUDU console "ho pci rnti pci" ◀─────────┘            (shadow mode by default)
```

* **Getting measurements in:**
  * The xApp parses the real 3GPP RRC **MeasurementReport**, either as JSON (srsRAN/OCUDU CU-CP
    logs, tshark) or as asn1c XER (OAI).
  * It converts RSRP/RSRQ/SINR report indices with the TS 38.133 mappings.
  * Or a RAN-side agent streams simple JSON lines over the **ran-bridge** protocol. A template is
    in `integrations/bridge-agent/`.
* **Sending handovers:**
  * **E2SM-RC Control Style 3 / Action 1 (Handover Control)** through srsRAN's O-RAN SC RIC
    (`integrations/oran-sc-ric/`).
  * **OAI telnet:** `ci trigger_f1_ho` / `ci trigger_n2_ho`.
  * **srsRAN/OCUDU console:** `ho`.
  * Any command (e.g. FlexRIC's `xapp_rc_handover`), or the bridge.
* **Safety:**
  * Shadow mode by default (`--live` to actuate); every decision goes to a JSONL audit log.
  * Guard rails: pending command, failure backoff, hold-off, neighbour relation table,
    confirmation count, rate limit.
  * An optional A3 override for out-of-distribution inputs.
  * Keep the RAN's own A3 handover as a backstop (example configs in `integrations/`).
* **Tested without hardware:** a simulator-backed `fake-gnb` speaks the protocol. With guard
  rails off, the full network path reproduces the offline benchmark exactly. `ran-parse` checks
  that your gNB's logs parse before you connect anything.

```bash
python -m ai_ran_llm ran-parse /tmp/gnb.log --cells integrations/ocudu/cells.example.json   # check input
python -m ai_ran_llm ran-xapp --cells integrations/ocudu/cells.example.json --rrc-log /tmp/gnb.log  # shadow
python -m ai_ran_llm ran-xapp --cells sim --actuator bridge --live & python -m ai_ran_llm fake-gnb  # no radio
```

Measured on the benchmark drives through the fake gNB (details and more settings in the guide):

| Setting | HO/UE/min | Ping-pong % | RLF/UE/min | HOF/UE/min | SE b/s/Hz | Outage % |
|---|---:|---:|---:|---:|---:|---:|
| Offline benchmark (`evaluate`) | 13.34 | 21.4 | 0.006 | 0.53 | 2.969 | 1.92 |
| Real-RAN path, defaults (A3 override 6 dB) | 13.35 | 21.4 | 0.003 | 0.53 | 2.969 | 1.92 |
| Real-RAN path, defaults, realistic reports (200 ms, 8 nbrs, RRC-quantised) | 11.03 | 16.4 | 0.016 | 0.85 | 2.944 | 2.79 |
| Real-RAN path, `--confirm 2` | 8.21 | 8.4 | 0.034 | 1.18 | 2.919 | 3.70 |

**Status.** The integration code is tested against fakes and the simulator only. OAI/srsRAN
command syntax and E2 calls follow their documentation and example xApps. It has not been run
against a live gNB here. Start in shadow mode, and retrain on your own traces
(`gen-data --from-drives`, see *Using real network data*) before trusting decisions on real
radio.

## Limitations

The model is trained on a synthetic channel model: a single-layer macro grid,
omnidirectional cells, no beam management, and no load or slice awareness. The
teacher optimises a 1 s look-ahead of large-scale RSRP. Treat the numbers above as a
simulation benchmark, not a field result.
