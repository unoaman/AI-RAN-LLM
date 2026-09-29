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

`data/handover_corpus.npz` (25 MB) is the exact corpus the shipped model was trained on:

* 527,512 samples from 60 simulated drives × 32 UEs × 60 s (seed 0), 12.5% of them handover
  labels.
* `tokens`: an int64 array of shape (527512, 64). Each row is the 41 report tokens, then the
  answer, padded with `<pad>`.
* `prompt_len`: 41.

Decode a row with `HandoverTokenizer().decode(row)`. To export the corpus as chat-format JSONL
for fine-tuning a general-purpose LLM, run `python -m ai_ran_llm export-jsonl`.

## Components

| Module | What it does |
|---|---|
| `ai_ran_llm/simulator.py` | Vectorised 19-cell hexagonal macro network (ISD 500 m): 3GPP path loss `128.1+37.6·log10(d)`, distance-correlated shadowing, fast fading, L3 filtering, Gauss-Markov UE mobility at 3–120 km/h. Closed-loop evaluation with T310-based RLF, HO failure, ping-pong (return within 1 s), HO interruption, SINR and spectral efficiency. |
| `ai_ran_llm/policies.py` | 3GPP A3 baseline and the look-ahead teacher (`oracle_decision`) used for labels. |
| `ai_ran_llm/tokenizer.py` | 349-token domain vocabulary: quantised RSRP/SINR/speed/gain tokens, neighbour-minus-serving deltas, cell IDs, structure and rationale words. |
| `ai_ran_llm/dataset.py` | Builds the corpus by driving simulations with a mix of behaviour policies (teacher, delayed teacher, randomised A3). This covers both good and off-optimal serving states, in the spirit of DAgger. Also exports JSONL for fine-tuning a general-purpose LLM. |
| `ai_ran_llm/model.py` | Decoder-only GPT (pre-LN, causal SDPA, tied embeddings). Default: 4 layers, 4 heads, 128-d, ~0.85 M parameters. It is small enough for near-RT (<10 ms) CPU inference. |
| `ai_ran_llm/train.py` | Causal-LM training. Full weight on answer tokens, and 0.1 weight on report tokens as a light telemetry-forecasting objective. |
| `ai_ran_llm/inference.py` | Scoring, grammar-constrained generation, confidence gating + A3 fallback, closed-loop policy. |
| `ai_ran_llm/serve.py` | HTTP endpoint (`POST /v1/handover`) for a RIC/xApp integration. |
| `ai_ran_llm/evaluate.py` | Closed-loop benchmark on identical drives: HandoverLLM vs A3 settings vs teacher. |

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

* **Inputs** map directly to the E2SM-KPM / RRC measurement report content: serving and
  neighbour RSRP (L3-filtered), serving SINR (from CQI) and UE speed (from Doppler or
  positioning).
* **Outputs** map to an E2SM-RC control message (handover to the target PCI/NR-CGI). You
  can also use them in "advisory" mode, tuning CIO or hysteresis per cell pair.
* **Before a live deployment**, retrain on your own traces. You can replay drive-test or
  RIC logs through `oracle_decision`, because the future RSRP is known offline. Tune
  `ObsConfig` to your reporting interval and neighbour list size. Keep the confidence gate
  and A3 fallback enabled.

## Limitations

The model is trained on a synthetic channel model: a single-layer macro grid,
omnidirectional cells, no beam management, and no load or slice awareness. The
teacher optimises a 1 s look-ahead of large-scale RSRP. Treat the numbers above as a
simulation benchmark, not a field result.
