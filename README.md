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

Example response for the built-in report (`infer -`) from the trained checkpoint:

```json
{
  "action": "STAY",
  "target_cell": null,
  "rationale": "Stay on serving cell: serving cell falling, predicted RSRP gain -1 dB over the next second, serving SINR is low.",
  "confidence": 0.5638,
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

In this example the model is arguably wrong. In the latest sample cell 4 is already 8 dB above the (falling) serving cell, and A3 would hand over. The model stays with only 56% confidence and gives cell 4 most of the rest. This is the model's known weakness: it is reluctant to hand over (13% HO recall; see Results). Raising `--min-confidence` (e.g. to 0.6) routes such uncertain cases to the A3 fallback, which would hand over to cell 4 here.

A report carries up to `n_neighbors` (4) neighbours and `hist_len` (5) L3-filtered RSRP
samples per cell, taken every 200 ms (oldest first). Shorter histories are left-padded.

## Results

Closed-loop benchmark: 5 unseen drives × 64 UEs × 60 s (5.3 UE-hours), with identical
channels for every policy. The model is the default 0.85 M-parameter configuration, trained
for 2 epochs on 527k samples (about 37 min on 4 CPU cores). Reproduce with
`python -m ai_ran_llm evaluate --episodes 5`.

| Policy | HO/UE/min | Ping-pong % | RLF/UE/min | HOF/UE/min | SINR dB | SE b/s/Hz | Outage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3 (1 dB, 200 ms) | 17.86 | 31.1 | 0.000 | 0.69 | 6.98 | 2.959 | 2.22 |
| A3 (2 dB, 300 ms) | 10.36 | 13.2 | 0.016 | 1.22 | 6.59 | 2.919 | 3.76 |
| A3 (3 dB, 500 ms) | 5.67 | 2.9 | 0.700 | 1.25 | 5.99 | 2.849 | 6.16 |
| **HandoverLLM** (conf 0.5) | 8.69 | 9.0 | 0.025 | 0.95 | 6.74 | 2.936 | 3.07 |
| Teacher (non-causal upper bound) | 8.01 | 5.3 | 0.000 | 0.00 | 7.41 | 3.006 | 0.56 |

How to read this:

* **HandoverLLM beats the mid-range A3 setting (2 dB / 300 ms) on every KPI.** It makes 16% fewer
  handovers, has fewer ping-pongs (9.0% vs 13.2%), 22% fewer HO failures, higher SINR and
  spectral efficiency, and less outage.
* **Against the aggressive A3 setting (1 dB):** it uses half the handovers and has less than a
  third of the ping-pongs, but gives up a little SE (-0.8%) and outage.
* **Against the conservative A3 setting (3 dB):** it avoids that setting's radio link failures and
  most of its HO failures.
* **There is still a clear gap to the teacher.** Per-sample validation accuracy is 88% and
  handover recall is only 13%. The teacher reacts to future shadowing, much of which cannot be
  predicted from a 1 s report history, so many of its labels cannot be learned. Smoothing the
  labels, adding richer inputs (beam and CSI measurements, position) and using larger models
  are the obvious next steps.

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
