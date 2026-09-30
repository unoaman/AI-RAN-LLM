# HandoverLLM — Design, Implementation and Development Record

This document explains the `AI-RAN-LLM` repository end to end: what it does, why it
is built the way it is, which alternatives were rejected and why, every algorithm and
assumption, every file, function, interface and artifact, and the full history of how
it was developed (including the experiments that failed).

It is written so that a human engineer **or another AI model** can pick the project up
cold, understand every design choice, reproduce every number, and change it safely.

> **Status.** Research prototype. Everything is trained and evaluated in a simulator.
> The real-RAN integration (§16.6, `docs/RAN_INTEGRATION.md`) is tested against fakes
> and the simulator, but not against a live gNB. See [§20 Limitations](#20-limitations-known-issues-and-risks).

---

## Table of contents

1. [How to read this document](#1-how-to-read-this-document)
2. [Problem statement and background](#2-problem-statement-and-background)
3. [Goals and non-goals](#3-goals-and-non-goals)
4. [Solution overview (big picture)](#4-solution-overview-big-picture)
5. [Key design decisions — why, and why not](#5-key-design-decisions--why-and-why-not)
6. [Assumptions](#6-assumptions)
7. [Radio and mobility simulator](#7-radio-and-mobility-simulator)
8. [Measurement reports (observations)](#8-measurement-reports-observations)
9. [Policies: A3 baseline and the look-ahead teacher](#9-policies-a3-baseline-and-the-look-ahead-teacher)
10. [Closed-loop evaluation and KPIs](#10-closed-loop-evaluation-and-kpis)
11. [Training-data pipeline](#11-training-data-pipeline)
12. [Tokenizer and the "language" of handover](#12-tokenizer-and-the-language-of-handover)
13. [Model architecture](#13-model-architecture)
14. [Training](#14-training)
15. [Inference, decision rule and safety](#15-inference-decision-rule-and-safety)
16. [Interfaces (CLI, HTTP, Python, files)](#16-interfaces-cli-http-python-files)
17. [File structure and function reference](#17-file-structure-and-function-reference)
18. [Artifacts](#18-artifacts)
19. [Results](#19-results)
20. [Limitations, known issues and risks](#20-limitations-known-issues-and-risks)
21. [Development history (from the build conversation)](#21-development-history-from-the-build-conversation)
22. [Extension guide and invariants (for humans and AI agents)](#22-extension-guide-and-invariants-for-humans-and-ai-agents)
23. [Testing](#23-testing)
24. [Glossary](#24-glossary)
25. [References](#25-references)

---

## 1. How to read this document

| If you want to… | Read |
|---|---|
| Understand the idea in 5 minutes | §2, §4, §19 |
| Know *why* something is done a certain way | §5 (decision records), §21 (history) |
| Understand an algorithm precisely | §7–§15 (formulas and pseudo-code) |
| Call the system | §16 (CLI, HTTP API, Python API, file formats) |
| Connect it to a real RAN (OCUDU / srsRAN, OAI) | §16.6 and `docs/RAN_INTEGRATION.md` |
| Find a function | §17 |
| Change the code without breaking reproducibility | §22 |
| Know what not to trust | §20 |

Conventions used below:

* `U` = number of UEs, `T` = number of time steps, `C` = number of cells,
  `K` = reported neighbours (4), `H` = RSRP history length per cell (5).
* One simulation step is **100 ms** (`dt_s = 0.1`).
* Code references are written as `module.function` (e.g. `simulator.run_policy`),
  all under `ai_ran_llm/`.

---

## 2. Problem statement and background

### 2.1 Mobility handover in cellular networks

A UE (phone, car modem) is served by one cell. As it moves, the signal of the
serving cell weakens and a neighbour's strengthens. The network must **hand over**
(HO) the UE to a better cell at the right moment:

* **Too late** → the serving link degrades below what the control channel can decode.
  The UE declares a **radio link failure (RLF)** (after timer **T310** expires) or the
  handover command itself cannot be delivered (**handover failure, HOF**). Either way,
  the UE loses service until it re-establishes (≈ hundreds of ms to seconds).
* **Too early / too often** → the UE is moved to a cell that is not yet (or only
  momentarily) better and quickly returns: a **ping-pong**. Each HO costs signalling
  and a short user-plane interruption.

### 2.2 How it is done today: event A3

In 3GPP NR/LTE the UE measures **RSRP** (reference-signal received power, dBm) of the
serving and neighbour cells, smooths it with a **layer-3 (L3) filter**, and reports when
**event A3** holds:

```
Mn + Ofn + Ocn − Hys  >  Mp + Ofp + Ocp + Off        for TimeToTrigger (TTT)
```

i.e. a neighbour is better than the serving cell by an offset/hysteresis for a
time-to-trigger. The network then commands the HO. A3 is **reactive** and has a single
static trade-off: small hysteresis/TTT ⇒ ping-pongs, large ⇒ late HOs, RLFs, HOFs.
Mobility Robustness Optimisation (MRO) tunes these parameters slowly per cell pair,
but cannot react to an individual UE's trajectory.

### 2.3 AI-RAN and the near-RT RIC

**AI-RAN** means running AI inside the radio access network. In the O-RAN
architecture, the **near-real-time RAN Intelligent Controller (near-RT RIC)** hosts
**xApps** that receive RAN telemetry over the **E2** interface (e.g. E2SM-KPM
measurements) and send control actions back (e.g. E2SM-RC handover control) within a
**10 ms – 1 s** control loop. A handover xApp is the natural home for this project.

### 2.4 The request

The project started from a single request:

> *"Can you create a LLM for AI-RAN which helps in mobility handover"*

The repository was empty. Everything here was designed and built from scratch in one
working session (see §21).

---

## 3. Goals and non-goals

**Goals**

1. A language model that reads a UE measurement report and outputs
   **(a)** stay / hand over, **(b)** the target cell, **(c)** a short human-readable
   rationale.
2. **Proactive**: learn to anticipate the next second instead of reacting like A3.
3. **Safe by construction**: never command a HO to a cell the UE did not report; have a
   deterministic fallback.
4. **Measurable**: a closed-loop benchmark against A3 on identical radio conditions with
   standard mobility KPIs.
5. **Deployable shape**: CPU inference in the near-RT RIC latency budget, an HTTP
   endpoint whose request format mirrors a measurement report.
6. **Reproducible**: every dataset, model and number regenerable from seeds.
7. **Bridge to real data and to general LLMs**: a documented trace format and a JSONL
   export for fine-tuning open LLMs.
8. **Real-RAN integration**: read real RRC measurement reports and command handovers on
   OCUDU / srsRAN Project and OpenAirInterface, safely (shadow mode, guard rails, backstop).

**Non-goals (explicitly out of scope)**

* A hardened, certified xApp product. The integration layer (§16.6) is a working reference:
  E2SM-RC via srsRAN's O-RAN SC RIC framework, OAI telnet, console and a generic bridge. It
  has not been validated on live RAN hardware.
* Beam management, carrier aggregation, conditional handover (CHO), dual connectivity,
  load balancing, slicing.
* A 3GPP-conformant system-level simulator (the simulator is a deliberately simple
  macro model, §7).
* Claims about field performance.

---

## 4. Solution overview (big picture)

### 4.1 One-paragraph summary

A **vectorised simulator** generates UE drives through a 19-cell network. A
**non-causal teacher** that can see 1 s into the future labels every measurement report
with the best action. Reports and labels are serialised into a **349-token domain
language** and a **0.85 M-parameter GPT** is trained to generate the answer
(`<ho> C4 <why> serving falling neighbor rising gain G+4 <eos>`). At inference,
**grammar-constrained decoding** restricts the output to reported cells, a
**probability threshold** (0.35) turns the model's calibrated-but-cautious
probabilities into decisions, and a **conservative A3 fallback** handles low
confidence. A **closed-loop benchmark** replays unseen drives with each policy and
measures HO rate, ping-pong, RLF, HOF, SINR, spectral efficiency and outage.

### 4.2 System context

```mermaid
flowchart LR
    subgraph RAN["RAN (gNB / DU / CU)"]
        UE[UE measurements<br/>RSRP, SINR, speed]
    end
    subgraph RIC["Near-RT RIC"]
        X[Handover xApp<br/>HandoverLLM]
    end
    UE -- "E2SM-KPM report<br/>(here: JSON over HTTP)" --> X
    X -- "E2SM-RC HO control<br/>(here: JSON response)" --> RAN
    X -. "rationale, probabilities, fallback flag" .-> OPS[Operator / MRO / logs]
```

In this repository the E2 leg is replaced by `POST /v1/handover` (§16.2).

### 4.3 Component view

```mermaid
flowchart TB
    subgraph Offline["Offline: data + training"]
        SIM[simulator.generate_episode<br/>channel + mobility] --> RP[simulator.run_policy<br/>behaviour policy in the loop]
        TR[(real traces .npz)] --> LE[simulator.load_episode]
        LE --> RP
        LE --> LOG[dataset._logged_reports<br/>logged serving cells]
        RP --> LAB[policies.label_decision<br/>look-ahead teacher]
        LOG --> LAB
        LAB --> TOK[tokenizer.HandoverTokenizer]
        TOK --> CORPUS[(data/handover_corpus.npz)]
        LAB --> RAW[("data/raw: reports.jsonl.gz,<br/>drive_XXX.npz")]
        CORPUS --> TRAIN[train.train]
        TRAIN --> CKPT[(checkpoints/handover_llm.pt)]
        CORPUS --> SFT[(SFT JSONL for general LLMs)]
    end
    subgraph Online["Online: inference"]
        CKPT --> LLM[inference.HandoverLLM]
        REQ[JSON report] --> R2O[inference.report_to_observation]
        R2O --> LLM
        LLM --> DEC{"P(ho to c) ≥ 0.35?"}
        DEC -- "yes: force ho + best cell" --> GEN["constrained generation<br/>of rationale"]
        DEC -- "no: force stay" --> GEN
        GEN --> CONF{"confidence ≥ 0.3?"}
        CONF -- no --> A3F[A3 fallback]
        CONF -- yes --> RESP[JSON decision]
        A3F --> RESP
    end
    subgraph Eval["Evaluation"]
        CKPT --> POL[inference.LLMPolicy]
        POL --> BENCH[evaluate.benchmark<br/>vs A3 + teacher]
        SIM --> BENCH
    end
```

The real-RAN side (sources, tracker, controller, actuators) is shown in §16.6.

### 4.4 The learning problem in one picture

```
 time ─────────────────────────────────────────────────────────────▶
          report window (what the model sees)          teacher look-ahead
   t-8   t-6   t-4   t-2    t        t+1 ........................ t+10
   │     │     │     │      │        │◀──── mean large-scale RSRP ───▶│
   R     R     R     R      R        best reported neighbour vs serving
   5 L3-filtered RSRP samples per cell,       gain > 2 dB  ⇒  label <ho> C*
   200 ms apart (+ SINR, speed)               else         ⇒  label <stay>
```

The model must infer from the past 800 ms what the teacher knows about the next
1000 ms. Part of that future (new shadowing) is unpredictable; this bounds achievable
accuracy (§19.3).

---

## 5. Key design decisions — why, and why not

Each entry: **decision**, **why**, **alternatives rejected and why not**.

### D1. A small domain-specific GPT trained from scratch

* **Why.** Near-RT RIC loops need ≤ tens of ms per decision for many UEs at ~10 Hz;
  a 0.85 M-parameter model scores one report in ~1.7 ms on CPU (§15.6). It runs on-prem
  (operator data stays local), deterministically, at negligible cost per decision, and
  its vocabulary is exactly the radio quantities involved.
* **Why an LLM at all (vs. a classifier).** The generative formulation gives, in one
  model: the decision, the target *chosen among variable reported cells*, and a
  rationale; it uses the same serialised interface a general-purpose LLM would use
  (hence the JSONL export), and it is extensible by adding tokens (new measurements)
  without new heads.
* **Why not** fine-tune a general LLM (Llama/Qwen/etc.) as the primary model: 100–1000×
  more parameters for a problem with ~40 numeric inputs, latency too high for the
  control loop, and nothing in its pretraining knows this simulator's radio behaviour.
  The path remains open via `export-jsonl` (§16.4).
* **Honest note.** A gradient-boosted tree or MLP on hand-made features would likely
  reach similar decision quality (a small MLP reached AUC ≈ 0.83 in §21.8). The LLM
  form is chosen for the interface, rationale and extensibility, not because the
  decision itself requires a transformer.

### D2. Simulation first, real data later

* **Why.** No real traces were available; a simulator gives unlimited labelled data,
  the ability to compute a non-causal teacher, and a closed-loop benchmark on
  *identical* channels for all policies.
* **Why not** public datasets: none provides per-UE RSRP of all cells at 100 ms
  together with a controllable serving cell for closed-loop evaluation.
* **Mitigation.** A documented trace format and `gen-data --from-drives` (§11.5)
  so real logs can replace the simulator for training.

### D3. Imitation of a look-ahead teacher (not reinforcement learning)

* **Why.** In a replayable drive the future is known, so the "right" action can be
  computed directly: the best reported neighbour over the next second. Supervised
  imitation is stable, fast and simple.
* **Why not** RL: reward design (trading RLF vs ping-pong vs throughput), sample
  efficiency, and stability would dominate the project; RL is a natural *next* step
  once the supervised policy exists (it can initialise an RL policy).
* **Distribution shift.** A pure imitation learner only sees states the teacher visits.
  The corpus therefore mixes behaviour policies (teacher, delayed teacher, randomised
  A3) so the model sees late, early and missed HO states — the idea behind DAgger
  (Ross et al., 2011), though without iterative re-collection (§11.2).

### D4. Teacher = best reported neighbour over 1 s, 2 dB margin

* **Why.** Chosen by closed-loop sweep (§21.3): margin 2 dB / horizon 10 steps gave
  low ping-pong (≈ 5 %) with near-zero RLF/HOF. Restricting candidates to the *reported*
  neighbours keeps labels achievable from the report.
* **Why not** horizon 20: fewer ping-pongs but more RLF (it waits too long);
  margin 1 dB: ~10 % ping-pong.

### D5. Tokenisation: quantised numeric tokens, fixed layout

* **Why.** Each numeric value becomes one token at 1 dB (RSRP, SINR, gain, delta) or
  10 km/h (speed) resolution — small vocabulary, every token has radio meaning, the
  prompt has a **fixed 41-token layout** (no padding inside prompts, positions carry
  meaning, batched scoring is trivial).
* **Neighbour RSRP as deltas to serving (D tokens).** The HO decision hinges on
  *neighbour − serving*. With absolute tokens the model has to learn subtraction
  between two learned embeddings. Deltas expose the key quantity directly (§21.5).
* **Numeric embedding initialisation.** Numeric tokens get a sinusoidal encoding of
  their value added to the random init, so `R-90` starts near `R-91` (ordinal prior).
* **Why not** float inputs through a linear projection: loses the "everything is a
  token" LLM interface and the JSONL-compatible text form; possible future hybrid.
* **Why not** BPE/text tokens (e.g. "-92.4 dBm"): 3–6 tokens per number, longer
  sequences, digit-level arithmetic burden.

### D6. Loss on answer tokens (weight 1) + report tokens (weight 0.1)

* **Why.** The answer is what matters; a small weight on report tokens adds a
  telemetry-forecasting auxiliary objective (predict how RSRP evolves within the
  history), which is cheap regularisation.

### D7. Grammar-constrained decoding

* **Why.** A free LLM can emit an invalid token or a cell that was never measured.
  The decoder masks logits so that only a valid answer can be produced:
  `<stay>|<ho> Ck` with `Ck` ∈ reported neighbours, then rationale vocabulary.
  This makes "HO to an unreported cell" impossible *by construction*.

### D8. Threshold decision rule `P(ho→c) ≥ 0.35` (not argmax)

* **Why.** HO labels are rare (12.5 %) and partly unpredictable, so the model's HO
  probabilities are conservative. Argmax (≡ threshold 0.5) hands over too late.
  A closed-loop sweep (§21.7) showed 0.35 dominates A3 (1 dB) on every KPI except a
  small RLF difference. The threshold is the main operator knob (`--ho-threshold`).
* **Why not** retrain with class weights: equivalent in effect, less transparent,
  and needs retraining for every operating point.

### D9. A3 fallback below a confidence floor

* **Why.** When the chosen action's probability is below `min_confidence` (0.3), a
  deterministic, explainable 3GPP-style rule decides instead. Operators can raise the
  floor to route more decisions to A3.

### D10. Evaluate in closed loop, not only per-sample accuracy

* **Why.** Per-sample accuracy is misleading here: always-STAY scores ~88–90 %.
  Decisions change future states (serving cell), so only closed-loop KPIs on identical
  drives reflect quality (§10).

### D11. HOF judged on large-scale SINR; Qout −8 dB, HOF −10 dB

* **Why.** With instantaneous (fast-faded) SINR, ~40 % of A3 handovers "failed" —
  unrealistic. The HO command is delivered over several TTIs, so large-scale SINR is
  the right quantity; thresholds were calibrated so A3 shows realistic
  failure/ping-pong trade-offs (§21.2).

### D12. Label smoothing kept as an option, default off

* **Why.** Tried "HO within W steps" and "confirm over 2 s" to make labels more
  learnable. Neither increased learnability and windowing made the teacher itself
  worse; the retrained model was slightly worse (§21.8). Kept as documented options
  for future work with richer inputs.

### D13. Standard library HTTP server

* **Why.** Zero dependencies, easy to test, shows the interface. **Why not** FastAPI /
  gRPC / E2 SDK: out of scope for a prototype; the handler is 30 lines to port.

### D14. Commit model + corpus + a raw sample

* **Why.** Users asked to commit them; they are small (3.4 MB, 25 MB, 5 MB) and make
  the repo usable immediately. Everything is also regenerable bit-for-bit from seeds.

### D15. Real-RAN integration: vendor-neutral core, thin adapters

* **Why.** OCUDU/srsRAN and OAI expose different mechanisms: logs, telnet, console, E2 with
  different SM versions, FlexRIC or the O-RAN SC RIC. All of them reduce to two canonical
  messages, `MeasReport` in and `HandoverCommand` out. The model, tracker and controller only
  see canonical messages, and each RAN needs only a small adapter.
* **Why not** write the xApp directly against one RIC SDK: that ties the decision logic to one
  RIC and one SM version and makes it untestable without that RIC.

### D16. Parse the real RRC MeasurementReport; do not rely on E2SM-KPM

* **Why.** The model needs per-UE RSRP of serving **and neighbour** cells. That is exactly the
  content of the RRC MeasurementReport (TS 38.331). Standard E2SM-KPM measurements do not
  include per-UE neighbour RSRP. The parser works on the ASN.1 field names, so one parser
  handles srsRAN/OCUDU JSON logs, tshark exports and asn1c XER (OAI).
* **Why not** proprietary metric streams: they are version-specific and usually lack
  neighbour measurements.

### D17. ran-bridge: NDJSON over TCP

* **Why.** Any RAN-side code (C, C++, Python) can emit JSON lines with no dependencies. The
  protocol is easy to debug with `nc`, versioned (`ai-ran-llm/ran-bridge/1`), and doubles as
  the fake gNB's transport, which makes the whole path testable.
* **Why not** gRPC/protobuf or E2AP between agent and xApp: heavier toolchains for an
  internal hop. E2 remains the actuation path where the RAN supports it.

### D18. Safety model: shadow by default, guard rails, A3 override, RAN backstop

* **Shadow mode** (`dry_run=True`) is the default, and actuation needs `--live`. Every decision
  goes to an audit log.
* **Guard rails** (pending, failure backoff, hold-off, neighbour relation table, confirmation
  count, rate limit) bound the damage of a bad model or bad input.
* **A3 override** (optional): the low-confidence fallback cannot catch *confident* mistakes on
  out-of-distribution inputs. A synthetic report with an 18 dB-stronger neighbour and an
  inconsistent SINR gave P(stay) = 0.998 (§21.13). The override hands over when a neighbour
  leads by a large margin anyway.
* **RAN backstop:** the example configs keep the RAN's own A3 handover at a conservative offset.
  An xApp outage then degrades to classical behaviour instead of "no handovers". The first
  draft disabled it; that was reversed once the failure mode was considered (§21.13).

### D19. Guard timers use the report clock

* **Why.** Reports carry RAN timestamps: simulation time, log time, or a RAN epoch. Outcomes may
  come from a different clock. Mixing them made "time since last handover" negative and
  blocked every handover (bug found by the end-to-end test, §21.13). All timers now use the
  UE's latest report time.

### D20. Lockstep fake gNB as the integration test

* **Why.** A simulator-backed gNB that speaks the real protocol, waits for a `decision` per
  report, and is scored by `run_policy` gives the same KPIs as the offline benchmark. With
  guard rails off it must reproduce the benchmark **exactly**, which is a strong regression
  test for parsing, resampling, routing and actuation. Timing does not affect results, because
  of the lockstep.

---

## 6. Assumptions

| Area | Assumption | Value / where |
|---|---|---|
| Deployment | Single-layer macro grid, omni cells, one carrier | 19 sites, ISD 500 m (`SimConfig.rings=2, isd_m=500`) |
| Propagation | 3GPP macro path loss | `128.1 + 37.6·log10(d_km)`, d ≥ 10 m |
| Shadowing | Log-normal, spatially correlated (Gudmundson), independent per cell | σ = 6 dB, decorrelation 50 m |
| Fast fading | i.i.d. Gaussian in dB per step (averaged over RBs) | σ = 2 dB |
| Measurement | UE error + L3 filter | σ = 1.5 dB, α = 0.5 (≈ filterCoefficient k=4 at 100 ms) |
| Tx / noise | Per-RE RS power, per-RE noise incl. NF | 15 dBm, −125 dBm |
| Interference | All other cells, scaled by load | load = 0.7 |
| Mobility | Constant speed per UE, Gauss-Markov heading, soft boundary | 3–120 km/h, heading noise 0.05 rad/step |
| City model (opt-in) | Road mobility with repeated routes and location-tied shadowing (§7.6) | `mobility="roads"`, `shadowing="spatial"`, `map_seed` 1 |
| Time | Decision and measurement every step | 100 ms |
| RLF | T310-like: N consecutive out-of-sync steps | Qout = −8 dB, 5 steps (500 ms) |
| HOF | Large-scale SINR at HO command below threshold | −10 dB |
| Recovery | Outage after RLF/HOF (re-establishment) | 1 s |
| Ping-pong | HO back to previous cell within | 1 s (10 steps) |
| HO cost | User-plane interruption | 50 ms (half a step of SE) |
| Report | Top-K neighbours by current filtered RSRP, H samples | K = 4, H = 5, stride 200 ms |
| Teacher | Best reported neighbour over next 1 s of large-scale RSRP | margin 2 dB, horizon 10 steps |
| Cells | Max cells per network | 64 (tokenizer limit) |
| Value ranges | Token clipping | RSRP [−140, −40], SINR [−20, 40], delta [−30, 30], gain [−15, 15] dB; speed bins 0–120+ km/h |

---

## 7. Radio and mobility simulator

File: `ai_ran_llm/simulator.py`. Key idea: **the channel does not depend on which cell
serves the UE**, so an episode is generated once (`generate_episode`), then any number
of policies can be replayed on it (`run_policy`). This guarantees identical conditions
for all compared policies.

### 7.1 Network layout — `hex_sites(rings, isd)`

Axial hex coordinates `(q, r)` with `max(|q|, |r|, |q+r|) ≤ rings`, mapped to
`x = isd·(q + r/2)`, `y = isd·r·√3/2`, sorted by (y, x). `rings=2` ⇒ 19 cells.
Cell index = row in this array (see `data/raw/cells.csv`).

```
        C16  C17  C18
     C12  C13  C14  C15
  C7   C8   C9   C10  C11        (index layout, ISD 500 m, C9 at origin)
     C3   C4   C5   C6
        C0   C1   C2
```

### 7.2 Mobility — Gauss-Markov heading

* Start uniformly in a disc of radius `R = 0.85·rings·isd` (850 m).
* Speed `v ~ U(3, 120) km/h`, constant per UE; step length `v/3.6·dt`.
* Each step: `heading += N(0, 0.05)`; if outside `R`, heading is reset towards the
  centre `+ N(0, 0.5)` (soft reflecting boundary).

### 7.3 Channel

For UE `u`, time `t`, cell `c` with distance `d` (clamped ≥ 10 m):

```
PL(d)              = 128.1 + 37.6·log10(d_km)                                 [dB]
S[t+1]             = a·S[t] + sqrt(1−a²)·N(0, σ_sh²),  a = exp(−Δd / d_corr)  (Gudmundson AR(1))
rsrp_true[u,t,c]   = P_tx − PL − S                         large-scale RSRP (teacher's view)
rsrp_inst[u,t,c]   = rsrp_true + N(0, σ_ff²)               with fast fading (drives SINR)
raw                = rsrp_inst + N(0, σ_meas²)             UE measurement error
rsrp_meas[t]       = (1−α)·rsrp_meas[t−1] + α·raw[t]       L3 filter (what reports contain)
```

### 7.4 SINR — `Episode.sinr_db(t, serving, large_scale=False)`

```
S = 10^(rsrp[serving]/10),   I = load · Σ_{c≠serving} 10^(rsrp[c]/10),   N = 10^(−125/10)
SINR_dB = 10·log10( S / (I + N) )
```

`large_scale=True` uses `rsrp_true` (used for HOF), otherwise `rsrp_inst` (used for RLM,
throughput, reports).

### 7.5 Episode containers

`Episode` holds `sites, pos, speed_kmh, rsrp_true, rsrp_inst, rsrp_meas, sim` and,
for loaded traces, `logged_serving, logged_sinr_db`. `save_episode` writes float32
`.npz`; `load_episode` reads the same layout or a real trace (NaN → −160 dBm =
"unmeasured", 64-cell limit, validation of `serving`) — see §16.5.

---

### 7.6 Opt-in city model — `city.py`

`SimConfig.mobility = "roads"` moves UEs along popular routes on a road network (street grid,
highways, walkways; Zipf popularity; stops at intersections). `SimConfig.shadowing = "spatial"`
replaces the per-path AR(1) shadowing with one Gudmundson-correlated field per cell, tied to
places and shared by all UEs and drives. The city is drawn from `map_seed` with its own random
generator. Both default off, and the original simulator is pinned by
`tests/test_city.py::test_default_simulator_unchanged`. With the city model, location, radio maps
and trajectory mining become measurable (`docs/LOCATION_AWARE_HANDOVER.md` §8–§9).

## 8. Measurement reports (observations)

`simulator.build_observation(ep, t, serving, obs_cfg)` produces an `Observation`
(batched over UEs) — the exact information an xApp would receive:

| Field | Shape | Content |
|---|---|---|
| `serving` | (U,) | serving cell index |
| `serving_hist` | (U, H) | serving L3-filtered RSRP at `t−8, t−6, t−4, t−2, t` (oldest first; indices clipped at 0) |
| `nbr_ids` | (U, K) | K strongest *other* cells by current filtered RSRP, strongest first |
| `nbr_hist` | (U, K, H) | their RSRP histories at the same instants |
| `sinr_db` | (U,) | current serving SINR (instantaneous) |
| `speed_kmh` | (U,) | UE speed |

`history_steps(t, obs)` computes `t − stride·[H−1 … 0]`.

The HTTP request format (§16.2) is the same information in JSON;
`inference.report_to_observation` converts it (sorting neighbours by latest RSRP,
left-padding short histories, padding missing neighbours with a −140 dBm copy).

---

## 9. Policies: A3 baseline and the look-ahead teacher

File: `ai_ran_llm/policies.py`. All policies implement a duck-typed protocol:

```python
class Policy:
    def reset(self, ep, obs_cfg): ...          # optional, called at episode start
    def decide(self, ep, t, obs) -> np.ndarray # (U,) target cell or -1 (stay)
    def on_handover(self, mask): ...           # optional, called with executed-HO mask
```

### 9.1 A3 — `A3Policy(hyst_db, ttt_steps)`

Per UE and cell, a counter increments while `meas[c] > meas[serving] + hyst` and resets
otherwise. When any counter reaches `ttt_steps`, hand over to the strongest such cell;
counters of UEs that handed over are reset. Uses **all** cells (not only the top-K),
which is favourable to A3. Benchmarked settings: (1 dB, 2 steps), (2 dB, 3), (3 dB, 5).

### 9.2 Teacher — `oracle_decision(ep, t, obs, obs_cfg)`

```
future[c] = mean(rsrp_true[u, t+1 : t+1+horizon, c])          # next 1 s, large scale
k*        = argmax over reported neighbours of future
gain      = future[k*] − future[serving]
target    = k* if gain > margin (2 dB) else −1
```

Returns `(target, gain)`. **Non-causal** — it uses the future. That is intentional: it
defines the behaviour the model must learn to anticipate. Also used as the upper-bound
row in benchmarks ("Oracle (non-causal)").

### 9.3 Smoothed labels — `label_decision` (options, default off)

* `label_window = W`: if the teacher would hand over at any of the next W steps
  (serving fixed), label the earliest such HO now.
* `label_confirm_horizon = L`: keep a HO label only if the target beats serving on
  average over the next L steps.
* `W = L = 0` ⇒ identical to `oracle_decision` (tested). See §21.8 for why off.

### 9.4 `OraclePolicy(obs_cfg, exec_prob=1.0, seed, smoothed=False)`

Runs the teacher in the loop. With `exec_prob < 1` each teacher HO is executed only
with that probability per step → **delayed HOs** → off-optimal states for the corpus.

---

## 10. Closed-loop evaluation and KPIs

### 10.1 `run_policy(ep, policy, obs_cfg, record=False)` — per-step algorithm

```mermaid
flowchart TD
    A["step t: compute SINR on serving"] --> B{"blocked?"}
    B -- no --> C["RLM: out-of-sync counter<br/>SINR below Qout = -8 dB"]
    B -- yes --> D
    C --> RLF{"counter ≥ 5?"}
    RLF -- yes --> R1["RLF: re-establish on strongest cell,<br/>blocked 1 s"]
    RLF -- no --> D
    R1 --> D["build Observation;<br/>record serving cell in traj"]
    D --> E["policy.decide"]
    E --> F{"target ≠ serving<br/>and not blocked?"}
    F -- no --> M["accumulate KPIs"]
    F -- yes --> G["HO executed: count HO;<br/>ping-pong if back to previous cell within 1 s;<br/>HOF if large-scale SINR below -10 dB, then blocked 1 s;<br/>SE × 0.5 this step"]
    G --> M
    M --> N["next step"]
```

Serving cell at t=0 = strongest measured cell. `traj[:, t]` records the serving cell
**when the step-t report was taken** (before the step-t decision) — this convention
makes exported drives rebuild the same reports (§11.5).

### 10.2 KPI definitions — `Metrics.summary()`

| KPI | Definition |
|---|---|
| `ho_per_ue_min` | executed HOs per UE-minute |
| `ping_pong_pct` | % of HOs returning to the previous cell within 1 s of the last HO |
| `rlf_per_ue_min` | RLFs per UE-minute |
| `hof_per_ue_min` | HO failures per UE-minute |
| `mean_sinr_db` | mean SINR over UE-steps; blocked steps count as Qout − 10 = −18 dB |
| `mean_se_bps_hz` | mean `log2(1+SINR)`; 0 while blocked; × (1 − 0.05/0.1) on HO steps |
| `outage_pct` | % of UE-steps blocked or with SINR < Qout |

### 10.3 Benchmark — `evaluate.benchmark`

Generates `n_episodes` drives from a fixed seed (default 10 000, disjoint from training
seed 0) and replays every policy on each identical drive; default 5 × 64 UEs × 60 s =
5.3 UE-hours.

---

## 11. Training-data pipeline

File: `ai_ran_llm/dataset.py`.

### 11.1 Flow

```mermaid
flowchart LR
    RNG[("rng = default_rng(seed)")] --> GE["generate_episode"]
    GE --> BP["_behaviour_policy<br/>draws from rng"]
    BP --> RP["run_policy record=True<br/>(t, Observation) for all t"]
    RP --> LS["_label_steps:<br/>teacher label + keep mask<br/>(rng for easy subsample)"]
    LS --> ENC["encode prompt + answer<br/>to a 64-token row"]
    ENC --> SH["rng.shuffle rows"]
    SH --> OUT[("handover_corpus.npz")]
```

### 11.2 Behaviour-policy mix — `_behaviour_policy(rng, obs_cfg)`

| Probability | Policy | Purpose |
|---|---|---|
| 35 % | teacher (`OraclePolicy`) | on-policy states |
| 25 % | delayed teacher, `exec_prob ~ U(0.1, 0.5)` | late-HO states |
| 40 % | A3, `hyst ~ U(0, 5) dB`, `ttt ~ {1..7}` | early/late/ping-pong states of classical control |

Labels are **always** from the teacher, whatever policy drove the serving cell.

### 11.3 Sample selection — `_label_steps`

For each labelled step (`t ≥ stride·(H−1)` so the history is full, and
`t + lookahead < T` so the teacher's window is complete):

* keep if teacher says HO, **or** best neighbour is within 6 dB of serving
  (a "hard" sample), **or** with probability 0.1 otherwise ("easy" STAY subsample).

Easy STAYs are uninformative; subsampling them keeps the corpus focused without
distorting borderline cases.

### 11.4 Reproducibility

`generate_dataset` and `export_raw` both consume `iter_labelled_drives` with the **same
random stream**. Consequences (all verified):

* `gen-data` with seed 0 regenerates `data/handover_corpus.npz` **bit-for-bit**.
* `export-raw` with seed 0 exports exactly drive 0 of the corpus; its `in_corpus` flags
  match that drive's 8 641 samples.

### 11.5 Drives from files (real traces) — `gen-data --from-drives`

`iter_labelled_drives(..., drives=[paths], serving="logged"|"replay")`:

* `logged` (default): reports are built along the file's `serving` array
  (`_logged_reports`), optionally with logged `sinr_db` — the states the real network
  visited. No policy in the loop.
* `replay`: runs the behaviour-policy mix on the file's RSRP (needs no `serving`).

Verified: rebuilding from `data/raw/drive_000.npz` (logged) reproduces all 569 HO
labels and all 7 099 non-trivial samples of drive 0 bit-for-bit; only the random 10 %
easy-STAY subsample differs.

### 11.6 Corpus statistics (shipped)

60 drives × 32 UEs × 60 s, seed 0 → **527 512 samples**, **65 798 HO labels (12.5 %)**,
shape (527 512, 64) int64, `prompt_len = 41`.

---

## 12. Tokenizer and the "language" of handover

File: `ai_ran_llm/tokenizer.py`, class `HandoverTokenizer`.

### 12.1 Vocabulary (349 tokens, fixed order)

| Id range | Family | Tokens | Resolution |
|---|---|---|---|
| 0–10 | specials | `<pad> <bos> <eos> <spd> <sinr> <srv> <nbr> <ans> <ho> <stay> <why>` | — |
| 11–17 | rationale words | `serving neighbor rising falling stable gain low_sinr` | — |
| 18–81 | cells | `C0 … C63` | index |
| 82–182 | serving RSRP | `R-140 … R-40` | 1 dB |
| 183–243 | SINR | `Q-20 … Q40` | 1 dB |
| 244–256 | speed | `V0 … V12` | 10 km/h bins (V12 = 120+) |
| 257–287 | predicted gain | `G-15 … G+15` | 1 dB |
| 288–348 | neighbour − serving | `D-30 … D+30` | 1 dB |

Values outside a range are clipped. **The order is part of the checkpoint contract**
(embedding rows) — see §22.

### 12.2 Prompt layout (41 tokens, fixed positions)

| Pos | Token | Meaning |
|---|---|---|
| 0 | `<bos>` | |
| 1–2 | `<spd> V*` | speed bin |
| 3–4 | `<sinr> Q*` | serving SINR |
| 5–6 | `<srv> C*` | serving cell |
| 7–11 | `R* ×5` | serving RSRP history (oldest → newest) |
| 12–18 | `<nbr> C* D*×5` | neighbour 1 (strongest): id + (nbr − serving) history |
| 19–25 | same | neighbour 2 |
| 26–32 | same | neighbour 3 |
| 33–39 | same | neighbour 4 |
| 40 | `<ans>` | answer starts at position 41 |

`prompt_len = 7 + H + K·(2 + H) + 1`.

### 12.3 Answer grammar

```
answer    := decision "<why>" rationale "<eos>"
decision  := "<stay>" | "<ho>" CELL          ; CELL ∈ reported neighbours
rationale := "serving" TREND [ "neighbor" TREND ] "gain" GAIN [ "low_sinr" ]
TREND     := "rising" | "falling" | "stable"
GAIN      := "G-15" … "G+15"
```

Label construction (`encode_answer`):

* `TREND` of a cell = last − first sample of its RSRP history: `> +1.5 dB` rising,
  `< −1.5 dB` falling, else stable (observable from the report).
* `neighbor TREND` only for HO (the target's trend).
* `GAIN` = teacher's gain of the best reported neighbour (future quantity — for STAY
  it may be positive but ≤ margin, or negative).
* `low_sinr` if serving SINR < 0 dB.

`explain()` renders tokens into JSON + a sentence, e.g.
*"Hand over to cell 4: serving cell falling, neighbor cell rising, predicted RSRP gain
+4 dB over the next second, serving SINR is low."*

### 12.4 Example (from the shipped corpus)

```
<bos> <spd> V8 <sinr> Q-3 <srv> C9 R-85 R-91 R-96 R-95 R-94
<nbr> C4 D-11 D-6 D+0 D-1 D-1  <nbr> C2 D-16 D-13 D-4 D-2 D-5
<nbr> C10 D-20 D-13 D-4 D-5 D-7  <nbr> C3 D-19 D-14 D-7 D-6 D-7
<ans> <stay> <why> serving falling gain G+1 low_sinr <eos>
```

### 12.5 Numeric features — `numeric_features(dim)`

For each numeric family, value `v` → `[sin(v·s·f_i), cos(v·s·f_i)]` with
`f_i = 100^(−i/(dim/2−1))`, `s = 1` (speed bins `s = 5`). Added ×0.05 to the random
embedding init in `train.train` (weights are tied with the output head).

---

### 12.6 Optional location-context tokens

`ObsConfig.use_position` / `use_radio_map` / `use_trajectory` (default off) append `<unk>` and the
families L (distance ratio), S (radial speed), M (radio-map gain), P (next-cell probability) and
N (route-history confidence) to the vocabulary, giving 447 tokens. Context tokens are inserted
after the serving block and after each neighbour block, and the prompt grows to at most 63 tokens.
Values come from `ai_ran_llm/location.py`; unknown values become `<unk>`. With the flags off,
nothing changes. Full description: `docs/LOCATION_AWARE_HANDOVER.md` §12.

## 13. Model architecture

File: `ai_ran_llm/model.py`, class `HandoverGPT` (nanoGPT-style decoder).

```mermaid
flowchart TB
    IN["token ids (B, T ≤ 63)"] --> E["token embedding 349×128<br/>+ learned position embedding 63×128"]
    E --> D["dropout 0.1"]
    D --> B1["Block ×4:<br/>x + CausalSelfAttn(LN(x))<br/>x + MLP(LN(x)), 128→512→128, GELU"]
    B1 --> LN["final LayerNorm"]
    LN --> H["LM head 128→349<br/>(weights tied to token embedding)"]
    H --> OUT["logits (B, T, 349)"]
```

| Component | Setting |
|---|---|
| Layers / heads / width | 4 / 4 / 128 (head dim 32) |
| Attention | `F.scaled_dot_product_attention(is_causal=True)` |
| Norm | pre-LayerNorm |
| Context | 63 (64-token rows shifted by one) |
| Init | N(0, 0.02); residual projections N(0, 0.02/√(2·layers)) |
| Parameters | **846 080** = tok emb 44 672 + pos emb 8 064 + 4 × 198 272 + final LN 256 |
| Loss | per-token cross-entropy with per-token weights, `ignore_index = −100` |

---

## 14. Training

File: `ai_ran_llm/train.py`.

### 14.1 Objective — `make_batch`

```
x = tokens[:, :-1],  y = tokens[:, 1:],  y[pad] = −100
w = 0.1 for report positions (< prompt_len−1), 1.0 for answer positions, 0 for pad
loss = Σ w·CE / Σ w
```

### 14.2 Optimisation

| Item | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, betas (0.9, 0.95), weight decay 0.1 (applied to all params) |
| Schedule | linear warm-up `min(500, total/10)` steps, cosine decay to 5 % of lr |
| Batch | 256 sequences |
| Epochs | 2 (CLI default) → 3 916 steps |
| Grad clip | 1.0 |
| Validation | first 5 % of the (shuffled) corpus |
| Hardware used | 4 CPU cores, no GPU: ≈ 0.56 s/step, ≈ 37 min for 2 epochs |

### 14.3 Validation metrics — `evaluate_split`

Teacher-forced: `answer_loss`; `decision_acc` = argmax over {STAY, HO} at the answer
position **and** (for HO) argmax cell correct; `ho_recall` = fraction of HO labels
predicted correctly with argmax. (Argmax ≠ deployed threshold rule — closed-loop KPIs
are the real metric.)

### 14.4 Checkpoint format (`.pt`, `torch.save` dict)

| Key | Content |
|---|---|
| `model_cfg` | `ModelConfig` dict (vocab 349, block 63, 4/4/128, dropout) |
| `obs_cfg` | `ObsConfig` dict used by the tokenizer |
| `sim_cfg` | `SimConfig` dict |
| `state_dict` | weights |
| `val` | last validation metrics |
| `epoch` | epoch number (checkpoints written after the per-epoch-save change) |

A checkpoint is written **after every epoch** (added after a run was lost, §21.9).

---

## 15. Inference, decision rule and safety

File: `ai_ran_llm/inference.py`, class `HandoverLLM`.

### 15.1 Scoring — one forward pass — `score(prompts)`

Append `<ho>` to each prompt (length P+1) and run the model once:

```
p(stay), p(ho)  = softmax over {<stay>, <ho>} logits at position P−1
p(c_k | ho)     = softmax over the K *reported* neighbour cell tokens at position P
p_ho[k]         = p(ho) · p(c_k | ho)                      (U, K)
```

Probabilities are renormalised over the allowed tokens only (constrained scoring).

### 15.2 Decision rule — `decide_batch(obs, ho_threshold)`

```
k* = argmax_k p_ho[k]
HO to nbr_ids[k*]  if p_ho[k*] ≥ ho_threshold (default 0.35)   else STAY
```

### 15.3 Constrained generation — `generate(prompt, prefix=None)`

```mermaid
flowchart TD
    S(["start: prompt ends with the ans token"]) --> D{"step 1: decision token<br/>allowed: stay, ho"}
    D -- "ho" --> C["step 2: target cell token<br/>allowed: only the 4 reported neighbours"]
    D -- "stay" --> W["why token<br/>allowed: why only"]
    C --> W
    W --> R["rationale token<br/>allowed: rationale words, gain tokens, eos"]
    R --> E{"eos emitted, or<br/>12 answer tokens reached?"}
    E -- "no" --> R
    E -- "yes" --> F(["end of answer"])
```

Greedy argmax within the allowed set at each state. `prefix` forces the decision
(e.g. `[<ho>, C4]`) so the model writes the rationale **for the decision actually
taken** by the threshold rule.

### 15.4 xApp entry point — `handle_report(report, ho_threshold=0.35, min_confidence=0.3, a3_hyst_db=3, a3_ttt=3)`

```mermaid
sequenceDiagram
    participant C as Client (RIC / RAN)
    participant S as serve.py handler
    participant L as HandoverLLM
    C->>S: POST /v1/handover {report}
    S->>L: handle_report(report)
    L->>L: report_to_observation → encode_prompts
    L->>L: score → p_stay, p_ho[k]
    alt p_ho[k*] ≥ ho_threshold
        L->>L: force ho + best cell, conf = p_ho[k*]
    else otherwise
        L->>L: force stay, conf = p_stay
    end
    L->>L: generate rationale (constrained)
    alt conf < min_confidence
        L->>L: A3 fallback on the same report<br/>(nbr − serving > hyst for last ttt samples)
    end
    L-->>S: {action, target_cell, rationale, confidence, p_stay, p_handover, source}
    S-->>C: 200 JSON
```

**Batched form.** `decide_reports(reports, ...)` does the same for many reports with one
forward pass. `explain=False` skips rationale generation (≈ 10× faster). `handle_report` is
`decide_reports([report])[0]`, and the RAN controller uses the batched form.

### 15.5 Safety properties

1. **No HO to an unreported cell** — scoring and generation are restricted to reported
   neighbour tokens.
2. **Well-formed output** — the grammar makes malformed answers impossible.
3. **Short reports do not dilute the decision** — when a report has fewer than K neighbours,
   `report_to_observation` pads with copies; `score` masks duplicate cell tokens so the real
   neighbour keeps all of P(ho → cell) (bug found during integration, §21.13).
4. **Deterministic fallback** — A3 on the same report below the confidence floor,
   flagged `source: "a3_fallback"`.
5. **Transparency** — full probability vector and rationale returned.

### 15.6 Measured latency (4-core CPU, PyTorch 2.14, shipped checkpoint)

| Call | Latency |
|---|---|
| `score` for 1 report | ≈ 1.7 ms |
| `handle_report` (score + rationale generation) | ≈ 17 ms |
| `decide_batch` for 64 UEs | ≈ 29 ms |

Decision-only scoring fits comfortably in a near-RT RIC loop; rationale generation
could be made asynchronous or cached (KV cache not implemented).

---

## 16. Interfaces (CLI, HTTP, Python, files)

### 16.1 CLI — `python -m ai_ran_llm <command>` (or `ai-ran-llm` after `pip install -e .`)

| Command | Purpose | Main options (default) |
|---|---|---|
| `gen-data` | build a tokenised corpus | `--episodes 60 --ues 32 --steps 600 --seed 0 --out data/handover_corpus.npz`; `--from-drives NPZ…` build from drive files; `--serving logged\|replay`; city flags `--mobility --shadowing --map-seed`; context `--use-position --use-radio-map --use-trajectory --location-service DIR` |
| `train` | train HandoverGPT | `--data … --out checkpoints/handover_llm.pt --epochs 2 --batch-size 256 --lr 1e-3 --layers 4 --heads 4 --embd 128` |
| `evaluate` | closed-loop benchmark vs A3 + teacher | `--ckpt … --episodes 5 --ues 64 --seed 10000 --ho-threshold 0.35 --json FILE`; city flags; `--location-service DIR` for context models |
| `infer` | decide for one JSON report | `REPORT.json` or `-` (built-in example), `--ckpt`, `--ho-threshold 0.35` |
| `serve` | HTTP endpoint | `--host 0.0.0.0 --port 8080 --ho-threshold 0.35 --min-confidence 0.3` |
| `export-raw` | raw drives + readable reports | `--episodes 1 --ues 32 --steps 600 --seed 0 --out data/raw` |
| `build-location-service` | radio map + handover statistics from history drives | `--out DIR --drives 20 --ues 32 --seed 777 --pos-sigma 10` + city flags |
| `export-jsonl` | chat-format SFT data | `--data … --out data/handover_sft.jsonl --limit N` |
| `ran-xapp` | handover xApp against a real RAN | `--cells FILE\|sim`, sources `--bridge HOST:PORT\|off`, `--rrc-log FILE`, `--id-regex`, `--learn-id`, `--pci-regex`; `--actuator log\|bridge\|oai-telnet\|console\|command`; `--live`; `--ho-threshold 0.35 --min-confidence 0.3 --confirm 1 --hold-off-s 0 --a3-override-db 6`; `--audit ran_audit.jsonl` |
| `fake-gnb` | simulated gNB on the bridge | `--xapp HOST:PORT --ues 16 --steps 600 --seed 10000 --realistic` |
| `ran-parse` | check a gNB log offline | `LOG --cells FILE --id-regex --learn-id --pci-regex` |

### 16.2 HTTP API — `serve.py`

`GET /healthz` → `{"status": "ok"}`

`POST /v1/handover` — body: one report object or a list of them.

Request:

```json
{
  "ue_id": "ue-42",
  "serving_cell": 9,
  "speed_kmh": 90,
  "sinr_db": -1.0,
  "serving_rsrp": [-88, -90, -92, -95, -97],
  "neighbors": [
    {"cell_id": 4,  "rsrp": [-99, -96, -94, -92, -89]},
    {"cell_id": 10, "rsrp": [-97, -97, -98, -98, -99]}
  ]
}
```

| Field | Required | Notes |
|---|---|---|
| `serving_cell` | yes | cell index 0–63 |
| `serving_rsrp` | yes | L3-filtered dBm, oldest first, 200 ms apart; shorter lists are left-padded, longer truncated to 5 |
| `neighbors[]` | yes, ≥ 1 | `cell_id`, `rsrp` list as above; sorted by latest RSRP, top 4 used, missing padded at −140 dBm |
| `sinr_db` | no (0) | serving SINR |
| `speed_kmh` | no (30) | |
| `ue_id` | no | echoed back |

Response (200):

```json
{
  "action": "HANDOVER",
  "target_cell": 4,
  "rationale": "Hand over to cell 4: serving cell falling, neighbor cell rising, predicted RSRP gain +4 dB over the next second, serving SINR is low.",
  "tokens": "<ho> C4 <why> serving falling neighbor rising gain G+4 low_sinr <eos>",
  "confidence": 0.3882,
  "p_stay": 0.5638,
  "p_handover": {"4": 0.3882, "10": 0.0107, "3": 0.0013, "8": 0.0361},
  "source": "llm",
  "ue_id": "ue-42"
}
```

`source` is `"llm"` or `"a3_fallback"`. Errors: 400 `{"error": "bad report: …"}` for
missing keys / bad types; 404 for other paths. No authentication or TLS (prototype).

### 16.3 Python API (main entry points)

```python
from ai_ran_llm.config import SimConfig, ObsConfig, ModelConfig
from ai_ran_llm.simulator import generate_episode, load_episode, save_episode, run_policy, build_observation
from ai_ran_llm.policies import A3Policy, OraclePolicy, oracle_decision, label_decision
from ai_ran_llm.dataset import generate_dataset, export_raw, export_jsonl, iter_labelled_drives
from ai_ran_llm.tokenizer import HandoverTokenizer
from ai_ran_llm.model import HandoverGPT
from ai_ran_llm.train import train
from ai_ran_llm.inference import HandoverLLM, LLMPolicy, report_to_observation
from ai_ran_llm.evaluate import benchmark, default_policies, format_table

llm = HandoverLLM.load("checkpoints/handover_llm.pt")
llm.handle_report(report_dict)                      # one decision + rationale
ep = generate_episode(64, 600, np.random.default_rng(1))
metrics, traj = run_policy(ep, LLMPolicy(llm, 0.35), ObsConfig())
```

### 16.4 SFT JSONL (for general-purpose LLMs) — `export-jsonl`

One line per corpus row:

```json
{"messages": [
  {"role": "system", "content": "You are a near-RT RIC mobility xApp. ..."},
  {"role": "user", "content": "UE speed 110-119 km/h, serving SINR -10 dB.\nServing cell 9, L3-filtered RSRP history (dBm, oldest first): -85, -83, -83, -83, -88.\nNeighbour cell 13, RSRP relative to serving (dB): +1, +2, +2, +1, +6.\n..."},
  {"role": "assistant", "content": "Hand over to cell 13: serving cell falling, neighbor cell rising, predicted RSRP gain +8 dB over the next second, serving SINR is low."}
]}
```

### 16.5 File formats

**Corpus** `data/handover_corpus.npz`: `tokens` int64 (N, 64), `prompt_len` int64 (41).

**Drive** `drive_XXX.npz` (export) / real trace (input to `--from-drives`):

| Array | Shape | Req. for traces | Meaning |
|---|---|---|---|
| `rsrp_meas` | (U, T, C) float | **yes** | L3-filtered RSRP dBm at 100 ms; NaN = not measured |
| `serving` | (U, T) int | for `--serving logged` | serving cell at each report |
| `sinr_db` | (U, T) | no | serving SINR at each report |
| `rsrp_true` | (U, T, C) | no (→ `rsrp_meas`) | teacher's look-ahead signal |
| `rsrp_inst` | (U, T, C) | no (→ `rsrp_meas`) | SINR computation |
| `speed_kmh`, `pos`, `sites` | (U,), (U,T,2), (C,2) | no | |

**Reports** `reports.jsonl.gz`: one JSON per UE per labelled step = HTTP request format
+ `episode, ue_id, t, time_s, label{action, target_cell, gain_db, rationale}, in_corpus`.

**Manifest** `drives.json`: `seed, n_ue, n_steps, dt_s, sim_config, obs_config,
drives[{episode, file, behaviour_policy}]`. **Cells** `cells.csv`: `cell_id,x_m,y_m`.

### 16.6 Real-RAN integration (`ai_ran_llm/ran/`)

The full guide, with protocol spec, per-RAN setup and troubleshooting, is
`docs/RAN_INTEGRATION.md`. Summary:

```mermaid
flowchart LR
    subgraph IN["measurement sources"]
        LOGS["RrcLogSource<br/>tail CU-CP log, RRC JSON / XER"]
        BR["BridgeServer<br/>ran-bridge NDJSON/TCP"]
    end
    subgraph CORE["decision core"]
        TRK["UEMeasurementTracker<br/>sample-and-hold to 5 × 200 ms"]
        CTL["HandoverController<br/>decide_reports + A3 override<br/>+ guard rails"]
    end
    subgraph OUT["actuators"]
        E2["OranScRicActuator<br/>E2SM-RC Style 3 / Action 1"]
        OAI["OAITelnetActuator<br/>ci trigger_f1_ho / n2_ho"]
        CON["ConsoleActuator<br/>ho pci rnti pci"]
        CMD["CommandActuator"]
        BRO["BridgeServer<br/>ho_command"]
    end
    LOGS --> TRK
    BR --> TRK
    TRK --> CTL
    CTL --> E2
    CTL --> OAI
    CTL --> CON
    CTL --> CMD
    CTL --> BRO
    CTL --> AUD[("audit JSONL")]
    CM[("cell map JSON<br/>PCI / NCI / gNB / E2 node ↔ index")] --> TRK
    CM --> CTL
```

* **Canonical messages** (`ran/messages.py`): `CellMeas`, `MeasReport`, `CellRef`,
  `HandoverCommand`, `HandoverOutcome`, `UERelease`, `Hello`, `Decision`. On the wire they
  follow the ran-bridge protocol `ai-ran-llm/ran-bridge/1`: NDJSON, with message types `hello`,
  `meas_report`, `ho_outcome`, `ue_release` (RAN → xApp) and `ho_command`, `decision`, `error`
  (xApp → RAN).
* **RRC parsing** (`ran/rrc.py`):
  * TS 38.133 index → value mappings: RSRP `n−157` dBm, RSRQ `(n−87)/2` dB, SINR `(n−47)/2` dB.
  * `parse_measurement_report` finds `measResults` at any depth.
  * `xer_to_dict` converts asn1c XER.
  * `iter_asn1_blocks` extracts JSON/XER blocks from logs, together with their header lines.
* **Cell map** (`ran/cells.py`): model index 0..63 ↔ PCI / NCI / PLMN / gNB / E2 node, plus the
  neighbour relation table.
* **Tracker** (`ran/tracker.py`): per-UE, per-cell time series, resampled at
  `t_last − {800, 600, 400, 200, 0}` ms by sample-and-hold. It left-pads, drops neighbours not
  seen for 2 s, and uses report timestamps only.
* **Controller** (`ran/controller.py`), per batch:
  * tracker update, then `decide_reports` (threshold + low-confidence A3 fallback), then the
    optional A3 override;
  * guard rails in order: pending → failure_backoff → hold_off → not_neighbour → confirm →
    rate_limit;
  * a `HandoverCommand`, marked `dry_run` in shadow mode.
* **Actuators** (`ran/actuators.py`): E2SM-RC through srsRAN's O-RAN SC RIC framework
  (`control_handover(e2_node_id, amf_ue_ngap_id, gnb_cu_ue_f1ap_id, plmn, target_nci)`), OAI
  telnet, srsRAN/OCUDU console via FIFO, a generic command (argv, no shell), and a log-only
  actuator.
* **Runtime** (`ran/runtime.py`):
  * processes events in order and batches consecutive reports;
  * turns actuation errors into `failure` outcomes;
  * echoes decisions to bridge agents and writes the audit JSONL.
* **Fake gNB** (`ran/fake_gnb.py`): a simulator-backed bridge client, used as the `run_policy`
  policy (D20).

---

## 17. File structure and function reference

```
AI-RAN-LLM/
├── README.md                     user guide, quick start, results
├── pyproject.toml                package metadata; deps numpy, torch; entry point ai-ran-llm
├── .gitignore                    ignores data/* and checkpoints/* except shipped artifacts
├── docs/
│   ├── DESIGN.md                 this document
│   ├── LOCATION_AWARE_HANDOVER.md location, radio map, trajectory mining: concept + findings
│   └── RAN_INTEGRATION.md        real-RAN integration guide (OCUDU / srsRAN, OAI)
├── ai_ran_llm/
│   ├── __init__.py               exports configs, HandoverGPT, HandoverTokenizer; __version__
│   ├── __main__.py               python -m ai_ran_llm → cli.main()
│   ├── config.py                 SimConfig, ObsConfig, ModelConfig
│   ├── simulator.py              network, channel, mobility, reports, closed loop, KPIs, drive I/O
│   ├── city.py                   opt-in city model: road mobility, spatial shadowing
│   ├── location.py               location context: position track, radio map, handover-sequence mining
│   ├── policies.py               A3, teacher, smoothed labels, OraclePolicy
│   ├── dataset.py                labelled drives, corpus, raw export, SFT export
│   ├── tokenizer.py              domain vocabulary, encode/decode/explain
│   ├── model.py                  HandoverGPT
│   ├── train.py                  training loop, validation, checkpoints
│   ├── inference.py              HandoverLLM, constrained decoding, xApp logic, LLMPolicy
│   ├── evaluate.py               benchmark harness and table formatting
│   ├── serve.py                  HTTP endpoint
│   ├── cli.py                    command line + EXAMPLE_REPORT
│   └── ran/                      real-RAN integration (§16.6)
│       ├── messages.py           canonical messages + ran-bridge protocol
│       ├── rrc.py                RRC MeasurementReport parsing, TS 38.133 mappings, log blocks
│       ├── cells.py              cell map (RAN ids ↔ model index), neighbour relations
│       ├── tracker.py            per-UE history, resampling to the model input
│       ├── controller.py         decisions, A3 override, guard rails, commands
│       ├── actuators.py          E2SM-RC, OAI telnet, console, command, log
│       ├── sources.py            RRC log tailing + UE id learning
│       ├── bridge.py             NDJSON/TCP server and client
│       ├── runtime.py            xApp event loop + audit log
│       ├── app.py                build an xApp from options (CLI, RIC wrapper)
│       └── fake_gnb.py           simulator-backed gNB over the bridge
├── tests/
│   ├── test_pipeline.py          14 model / data / inference tests
│   ├── test_ran.py               16 real-RAN integration tests
│   ├── test_city.py              5 city-model tests (incl. default-simulator fingerprint)
│   └── test_location.py          7 location-context tests
├── experiments/                  location / radio-map / trajectory learnability studies
├── integrations/
│   ├── ocudu/                    cells.example.json, mobility.example.yml (OCUDU / srsRAN)
│   ├── oai/                      cells.example.json, measurement.example.conf
│   ├── oran-sc-ric/              handover_llm_xapp.py (E2SM-RC xApp for srsRAN's RIC)
│   └── bridge-agent/             example_agent.py (RAN-side agent template)
├── checkpoints/
│   └── handover_llm.pt           shipped model (3.4 MB)
└── data/
    ├── handover_corpus.npz       shipped training corpus (25 MB)
    └── raw/                      drive 0 of the corpus, readable (5 MB)
        ├── reports.jsonl.gz
        ├── drive_000.npz
        ├── drives.json
        └── cells.csv
```

### 17.1 `config.py`

| Name | Description |
|---|---|
| `SimConfig` | radio/mobility parameters (§6): `rings, isd_m, dt_s, tx_power_dbm, noise_dbm, load, shadow_sigma_db, shadow_decorr_m, fading_sigma_db, meas_sigma_db, l3_alpha, min/max_speed_kmh, q_out_db, t310_steps, hof_sinr_db, ping_pong_steps, ho_interruption_s` |
| `ObsConfig` | report + teacher: `n_neighbors=4, hist_len=5, hist_stride=2, oracle_horizon=10, oracle_margin_db=2.0, label_window=0, label_confirm_horizon=0` |
| `ModelConfig` | `vocab_size (from tokenizer), block_size, n_layer, n_head, n_embd, dropout`; `to_dict()` |

### 17.2 `simulator.py`

| Name | Description |
|---|---|
| `hex_sites(rings, isd)` | (C, 2) hex-grid site coordinates |
| `Episode` | dataclass holding one drive; props `n_ue, n_steps, n_cells`; `sinr_db(t, serving, large_scale)` |
| `generate_episode(n_ue, n_steps, rng, sim)` | mobility + channel + L3 measurements (§7) |
| `save_episode(path, ep, **extra)` | float32 compressed `.npz` |
| `UNMEASURED_DBM` | −160, replaces NaN in traces |
| `load_episode(path, sim)` | load export or real trace; validates; fills defaults |
| `Observation` | batched report (§8) |
| `history_steps(t, obs)` | report sample indices |
| `build_observation(ep, t, serving, obs)` | top-K neighbours + histories + SINR + speed |
| `Metrics` | KPI accumulators; `merge()`, `summary()` |
| `run_policy(ep, policy, obs_cfg, record)` | closed-loop replay (§10); returns metrics, serving trajectory, optionally `(t, obs)` list |

**`city.py`:** `ROUTE_TYPES`, `Route`, `city_routes(sim)`, `road_mobility(n_ue, n_steps, rng, sim)`
→ (pos, speed, route_id), `shadow_fields(sim)`, `spatial_shadowing(pos, sim)`. `Episode.route_id`
is set in road mode.

**`location.py`:** `LocationConfig`, `position_track(ep, rng, cfg)`, `previous_cells(traj)`,
`RadioMap.add/lookup/coverage`, `TransitionModel.add/probs/to_dict/from_dict`,
`LocationService.build/save/load`, `compute_context(ep, t, obs, est, prev, service, cfg)`,
`drop_context`, `LocationAwarePolicy`. `Observation.context` holds the arrays;
`inference.empty_context`, `stack_observations` and `report_to_observation` carry them.

### 17.3 `policies.py`

| Name | Description |
|---|---|
| `A3Policy(hyst_db, ttt_steps)` | `reset`, `decide`, `on_handover` (§9.1) |
| `oracle_decision(ep, t, obs, obs_cfg)` | teacher target + gain (§9.2) |
| `_future_mean(ep, start, length)` | mean large-scale RSRP over a future window |
| `label_decision(ep, t, obs, obs_cfg)` | teacher with optional window/confirm smoothing (§9.3) |
| `OraclePolicy(obs_cfg, exec_prob, seed, smoothed)` | teacher in the loop, optionally delayed |

### 17.4 `dataset.py`

| Name | Description |
|---|---|
| `_behaviour_policy(rng, obs_cfg)` | draws the rollout policy (§11.2) |
| `describe_policy(policy)` | human-readable name for manifests |
| `LabelledStep` | `t, obs, target, gain, keep` for all UEs at one step |
| `_label_steps(ep, seen, rng, obs_cfg, easy_keep_prob)` | teacher labels + keep mask (§11.3) |
| `_logged_reports(ep, obs_cfg)` | reports along logged serving cells |
| `iter_labelled_drives(...)` | generator over simulated or file drives (§11.1, §11.5) |
| `generate_dataset(...)` | corpus dict `{tokens, prompt_len}` |
| `export_raw(out_dir, ...)` | raw drives + readable reports (§16.5) |
| `prompt_to_text(tok, ids)` | English rendering of a prompt |
| `export_jsonl(tokens, prompt_len, path, limit)` | chat-format SFT file |

### 17.5 `tokenizer.py`

| Name | Description |
|---|---|
| constants | `SPECIALS, WORDS, MAX_CELLS=64, RSRP_RANGE, SINR_RANGE, SPEED_BINS=13, GAIN_RANGE, DELTA_RANGE, TREND_DB=1.5` |
| `_trend_word(delta)` | rising / falling / stable |
| `HandoverTokenizer(obs_cfg)` | builds vocab; attributes `PAD, BOS, …, W_GAIN, W_LOW_SINR`, family bases |
| `.vocab_size`, `.prompt_len` | 349, 41 |
| `.cell/.rsrp/.sinr/.speed/.gain/.delta(v)` | value → token id (clipped, vectorised) |
| `.numeric_features(dim)` | sinusoidal value encodings (§12.5) |
| `.is_cell(tok)` | cell-token test |
| `.encode_prompts(obs)` | (U, 41) prompt ids, vectorised |
| `.encode_answer(target, gain, serving_hist, nbr_ids, nbr_hist, sinr_db)` | label tokens (§12.3) |
| `.decode(ids)` | tokens → string |
| `.explain(answer_ids)` | tokens → `{action, target_cell, rationale}` |

### 17.6 `model.py`

| Name | Description |
|---|---|
| `CausalSelfAttention` | fused QKV, SDPA causal |
| `Block` | pre-LN attention + MLP |
| `HandoverGPT(cfg)` | `forward(idx, targets=None, loss_weight=None) → (logits, loss)`, `num_params()` |

### 17.7 `train.py`

| Name | Description |
|---|---|
| `make_batch(tokens, prompt_len, pad_id, prompt_weight)` | inputs, targets, weights (§14.1) |
| `evaluate_split(model, tokens, prompt_len, tok)` | validation metrics (§14.3) |
| `train(data_path, out_path, epochs, batch_size, lr, prompt_weight, val_frac, seed, model_cfg, device, log_every)` | full loop, per-epoch checkpoint |

### 17.8 `inference.py`

| Name | Description |
|---|---|
| `HandoverLLM(model, tok, device)` / `.load(path)` | wrapper |
| `.score(prompts)` | `p_stay (U,)`, `p_ho (U, K)` (§15.1) |
| `.decide_batch(obs, ho_threshold)` | targets + confidence (§15.2) |
| `.generate(prompt, max_new, prefix)` | constrained greedy decoding (§15.3) |
| `.handle_report(report, ho_threshold, min_confidence, a3_hyst_db, a3_ttt)` | xApp logic (§15.4) |
| `.decide_reports(reports, ..., explain)` | batched `handle_report`, one forward pass |
| `stack_observations(list)` | concatenate `Observation`s along the UE axis |
| `.explain_answer(ids)` | `explain` + raw tokens |
| `report_to_observation(report, obs_cfg)` | JSON → `Observation` |
| `LLMPolicy(llm, ho_threshold)` | policy adapter for `run_policy` |

### 17.9 `evaluate.py`, `serve.py`, `cli.py`

| Name | Description |
|---|---|
| `evaluate.COLUMNS` | KPI keys and table headers |
| `evaluate.benchmark(policies, n_episodes, n_ue, n_steps, seed, sim, obs_cfg)` | identical drives for all policies |
| `evaluate.default_policies(llm, obs_cfg, ho_threshold=0.35)` | A3 ×3, HandoverLLM, teacher |
| `evaluate.format_table(results)` | fixed-width table |
| `serve.make_handler(llm, ho_threshold, min_confidence)` | request handler class |
| `serve.serve(checkpoint, host, port, ho_threshold, min_confidence)` | start `ThreadingHTTPServer` |
| `cli.main(argv)` | argparse dispatcher (§16.1) |
| `cli.EXAMPLE_REPORT` | built-in example (serving 9 falling, cell 4 rising) |

### 17.10 `ran/` (real-RAN integration)

| Module | Names |
|---|---|
| `messages.py` | `PROTOCOL`, `ProtocolError`, `CellMeas`, `MeasReport`, `CellRef`, `HandoverCommand`, `HandoverOutcome` (`OUTCOMES`), `UERelease`, `Hello`, `Decision`, `encode(msg)`, `decode(line)` |
| `rrc.py` | `rsrp_dbm/rsrq_db/sinr_db(index)`, `rsrp_index/sinr_index(value)`, `ParsedMeasReport`, `parse_measurement_report(dict)`, `xer_to_dict(xml)`, `LogBlock`, `iter_asn1_blocks(lines, header_filter, on_line)` |
| `cells.py` | `CellMap.from_dict/load/to_dict/resolve/by_pci/is_neighbour/for_simulator` |
| `tracker.py` | `UEMeasurementTracker.update/ready/history/build_report/serving/ue_ids/forget/gc` |
| `controller.py` | `ControllerConfig` (threshold, fallback, `a3_override_db`, guard parameters, `dry_run`, `explain`), `HandoverController.on_reports/on_outcome/on_release`, `ControllerStats` |
| `actuators.py` | `command_fields(cmd)`, `LogActuator`, `OAITelnetActuator(host, port, mode)`, `ConsoleActuator(path, template)`, `CommandActuator(template)`, `OranScRicActuator(xapp)`, `ActuationError` |
| `sources.py` | `DEFAULT_ID_PATTERNS`, `follow_lines(path)`, `RrcLogSource(path, ..., learn_patterns)` (`reports()`, `to_report()`, `learn()`, `start(events)`) |
| `bridge.py` | `BridgeServer(host, port, events)` (`start`, `send`, `send_handover`, `close`), `BridgeClient(host, port, node)` (`send`, `send_many`, `recv`) |
| `runtime.py` | `XAppRuntime(controller, actuator, events, audit_path)` (`run`, `process`, `stop`, `close`) |
| `app.py` | `XAppOptions`, `build_xapp(options, actuator=None)`, `load_cells(spec)` |
| `fake_gnb.py` | `FakeGnb(ep, host, port, cell_map, report_every, report_all_cells, max_neighbours, quantize_rrc)` (`decide`, `on_handover`, `run`) |

---

## 18. Artifacts

| Artifact | Size | Produced by | Reproducible | Notes |
|---|---|---|---|---|
| `checkpoints/handover_llm.pt` | 3.4 MB | `train --epochs 2` on the shipped corpus | yes, up to PyTorch nondeterminism | the "v1" model: delta encoding, raw teacher labels; val acc 0.880, HO recall (argmax) 0.135 |
| `data/handover_corpus.npz` | 25 MB | `gen-data` (seed 0, 60 × 32 × 600) | **bit-for-bit** | 527 512 rows, 12.5 % HO |
| `data/raw/reports.jsonl.gz` | 1.3 MB | `export-raw` (seed 0) | yes (gzip header timestamp differs) | 18 624 reports of drive 0, 8 641 `in_corpus` |
| `data/raw/drive_000.npz` | 4.0 MB | `export-raw` | yes | full channel of drive 0 |
| `data/raw/drives.json`, `cells.csv` | < 1 kB | `export-raw` | yes | drive 0 behaviour policy: delayed teacher p = 0.37 |

Integration examples (hand-written, `integrations/`): `ocudu/cells.example.json`,
`ocudu/mobility.example.yml`, `oai/cells.example.json`, `oai/measurement.example.conf`,
`oran-sc-ric/handover_llm_xapp.py`, `bridge-agent/example_agent.py`. The config files follow
upstream key names but must be adapted to your deployment and version.

Not committed (regenerable): SFT JSONL (hundreds of MB for the full corpus), additional
raw drives (~5 MB each), experiment checkpoints.

---

## 19. Results

### 19.1 Closed-loop benchmark (shipped model)

5 unseen drives × 64 UEs × 60 s (seed 10 000); identical channels for every policy;
reproduce with `python -m ai_ran_llm evaluate --episodes 5`.

| Policy | HO/UE/min | Ping-pong % | RLF/UE/min | HOF/UE/min | SINR dB | SE b/s/Hz | Outage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3 (1 dB, 200 ms) | 17.86 | 31.1 | 0.000 | 0.69 | 6.98 | 2.959 | 2.22 |
| A3 (2 dB, 300 ms) | 10.36 | 13.2 | 0.016 | 1.22 | 6.59 | 2.919 | 3.76 |
| A3 (3 dB, 500 ms) | 5.67 | 2.9 | 0.700 | 1.25 | 5.99 | 2.849 | 6.16 |
| **HandoverLLM (0.35)** | 13.34 | 21.4 | 0.006 | 0.53 | 7.04 | 2.969 | 1.92 |
| Teacher (non-causal) | 8.01 | 5.3 | 0.000 | 0.00 | 7.41 | 3.006 | 0.56 |

**Reading.** HandoverLLM beats every A3 setting on SE, SINR, outage and HOF. Against
aggressive A3 (1 dB) it uses 25 % fewer HOs with a third fewer ping-pongs (small RLF
difference: 0.006 vs 0). Against A3 (2 dB) it trades more HOs/ping-pongs for much less
outage and < half the HOFs. A large gap to the teacher remains.

### 19.2 Threshold sweep (same model)

| Threshold | HO/UE/min | Ping-pong % | RLF | HOF | SE | Outage % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 60.08 | 65.0 | 0.075 | 3.17 | 2.781 | 7.23 |
| 0.25 | 22.39 | 39.3 | 0.022 | 0.71 | 2.962 | 2.15 |
| **0.35** | 13.34 | 21.4 | 0.006 | 0.53 | 2.969 | 1.92 |
| 0.50 | 8.69 | 9.0 | 0.025 | 0.95 | 2.936 | 3.07 |

### 19.3 Why there is a ceiling

The teacher reacts to shadowing that has not happened yet. A small MLP on report
features reaches only AUC ≈ 0.83 for "teacher hands over to the strongest neighbour now"
(§21.8), and the per-sample HO recall of simple threshold rules on the current gap is
18–45 % (§21.6). Example from `data/raw` (UE 23, t = 1.4 s): the teacher labels
"hand over to cell 14" although cell 14 is currently weaker and falling, because it
*will* be 2 dB better over the next second; the model stays with 0.99 confidence — the
reasonable call from the report alone.

### 19.4 Through the real-RAN path (fake gNB), guard rails and reporting

Same 5 drives × 64 UEs × 60 s, run through bridge → tracker → controller → actuator → fake gNB:

| Setting | HO/UE/min | Ping-pong % | RLF/UE/min | HOF/UE/min | SE b/s/Hz | Outage % |
|---|---:|---:|---:|---:|---:|---:|
| A3 (2 dB, 300 ms), for reference | 10.36 | 13.2 | 0.016 | 1.22 | 2.919 | 3.76 |
| A: no guard rails, override off (= benchmark `evaluate`) | 13.34 | 21.4 | 0.006 | 0.53 | 2.969 | 1.92 |
| I: **defaults**: no guard rails, A3 override 6 dB | 13.35 | 21.4 | 0.003 | 0.53 | 2.969 | 1.92 |
| B: confirm 2 | 8.21 | 8.4 | 0.034 | 1.18 | 2.919 | 3.70 |
| C: hold-off 1 s | 10.95 | 10.0 | 0.341 | 0.76 | 2.926 | 3.56 |
| H: confirm 2, realistic reports (200 ms, 8 neighbours, RRC-quantised) | 6.21 | 4.2 | 0.263 | 1.45 | 2.869 | 5.46 |
| G: defaults, realistic reports (200 ms, 8 neighbours, RRC-quantised) | 11.03 | 16.4 | 0.016 | 0.85 | 2.944 | 2.79 |
| J: A3 override 8 dB instead of 6 dB | 13.35 | 21.4 | 0.003 | 0.53 | 2.969 | 1.92 |
| K: confirm 2 + A3 override 8 dB | 8.22 | 8.4 | 0.028 | 1.18 | 2.919 | 3.68 |
| D: confirm 2 + hold-off 1 s | 7.67 | 4.9 | 0.175 | 1.14 | 2.906 | 4.12 |
| F: confirm 2 + hold-off 1 s, realistic reports | 6.03 | 3.0 | 0.322 | 1.41 | 2.865 | 5.59 |

Guard rails trade ping-pong for outage/RLF; the A3 override is free in-distribution. Defaults: confirm 1, hold-off 0, override 6 dB (see `docs/RAN_INTEGRATION.md` §6).

---

## 20. Limitations, known issues and risks

**Modelling**

* Simulator simplifications: omni cells, single carrier, no beams, no load dynamics,
  no RACH/CHO/DAPS, fading i.i.d. per step in dB, constant UE speed, simplified
  T310/RLF/HOF, HOF still attaches the UE to the target after recovery.
* Teacher uses large-scale RSRP only (ignores interference/load); maximises RSRP, not
  throughput.
* Model sees only the top-4 neighbours and 5 samples; A3 sees all cells.
* Decisions every 100 ms with no model-side hysteresis/TTT → ping-pong 21 % at
  threshold 0.35.

**Engineering**

* `train.train` builds the tokenizer with the **default** `ObsConfig`; a corpus built
  with a different `ObsConfig` fails the `prompt_len` assertion — changing the report
  shape requires passing the config through (not yet a CLI option).
* `evaluate_split` metrics use argmax, not the deployed threshold.
* `report_to_observation` defaults `sinr_db=0`, `speed_kmh=30` when missing, and pads
  missing neighbours by duplicating the last one at −140 dBm (duplicated cells share
  probability mass in `score`).
* Rationale generation has no KV cache (~17 ms per report).
* Cell ids limited to 0–63; real PCI/NR-CGI must be mapped externally.
* HTTP server: no auth, TLS, rate limiting, or batching across requests.
* Weight decay applies to all parameters (embeddings, LayerNorm) — simplification.

**Location context**

* Positions are simulated (true position + correlated error), not derived from AoA/TA.
  Velocities are unfiltered differences of noisy positions.
* The RAN integration's tracker does not yet fill context fields.
* Radio map and route statistics come from simulated A3 history drives in the same simulated
  city.

**Real-RAN integration**

* Not run against a live gNB here. The OAI/srsRAN commands, config keys and E2 call follow
  upstream docs and example code, and may differ in your version.
* Log header formats vary. UE ids and the serving PCI are extracted with configurable regexes,
  which you must check with `ran-parse`.
* The OAI telnet F1 trigger picks the target DU itself, so it matches the model's target only
  with two DUs.
* The model sees at most 4 neighbours, and reports with fewer are padded. That input was never
  seen in training.
* Real SINR and speed may be missing: the defaults are SINR 0 dB and 30 km/h.
* The `rate_limit` guard uses wall-clock time, so disable it for faster-than-real-time replays.
* The in-RIC xApp (`integrations/oran-sc-ric`) is untested without a RIC.

**Evaluation**

* All results are simulation-only; `evaluate` always uses simulated drives; replaying
  real traces scores RLF/HOF with the simulator's radio model.
* Results are from single training runs (no seeds-variance study).

**Risk statement.** Do not deploy to a live network without retraining on real data,
shadow-mode evaluation, and the confidence gate/A3 fallback enabled.

---

## 21. Development history (from the build conversation)

This section records how the project was actually built in one conversation with the
user, including requests, decisions, mistakes and their fixes. Commit hashes refer to
branch `claude/llm-ai-ran-mobility-handover-sip5ax`.

### 21.1 Request and initial build — `72d07de`

**User:** *"Can you create a LLM for AI-RAN which helps in mobility handover."*
The repository was empty. The environment had Python 3.11 and no ML libraries;
PyTorch and NumPy were installed (CPU only, 4 cores, 15 GB RAM, no GPU).

Plan chosen: simulator → teacher labels → domain tokens → small GPT → constrained
inference → closed-loop benchmark vs A3 → HTTP xApp endpoint → tests. First version
used **absolute** RSRP tokens for neighbours.

### 21.2 Simulator calibration (before training)

| Setting | A3 (2 dB) HOF/UE/min | Observation |
|---|---:|---|
| Qout −6 dB, HOF −7 dB on **instantaneous** SINR | 3.47 (≈ 40 % of HOs) | unrealistic: fast fading dominates |
| same thresholds, HOF on **large-scale** SINR | 2.72 | still too harsh for this interference level |
| **Qout −8 dB, HOF −10 dB, large-scale** (kept) | 0.90 | realistic trade-offs; teacher HOF ≈ 0.01 |

### 21.3 Teacher tuning (closed-loop, 3 drives × 64 UEs)

| Margin / horizon | HO/UE/min | Ping-pong % | RLF | Outage % |
|---|---:|---:|---:|---:|
| 1 dB / 1 s | 8.41 | 10.5 | 0.005 | 0.57 |
| **2 dB / 1 s (kept)** | 6.77 | 4.9 | 0.000 | 0.48 |
| 2 dB / 2 s | 5.09 | 0.3 | 0.073 | 1.43 |
| 3 dB / 1 s | 5.68 | 2.8 | 0.000 | 0.53 |
| 3 dB / 2 s | 4.39 | 0.0 | 0.057 | 1.31 |

### 21.4 First training — v0 (absolute tokens)

* Corpus 527 512 samples in ~1 min. First attempt at 3 epochs was estimated at ~2 h
  on CPU (1.3 s/step under contention) and restarted with 1 epoch (~18 min at
  0.56 s/step).
* v0 validation: decision accuracy 0.877, HO recall 0.176.
* Closed loop (argmax rule): HO 10.72, ping-pong 14.4 %, RLF 0.031, HOF 0.725,
  SE 2.952, outage 2.53 % → better than A3 (2 dB) on HOF/SE but only on par with A3 (1 dB).

### 21.5 Neighbour deltas + numeric init — v1 — `bebff69`

Hypothesis: absolute neighbour tokens force the model to learn subtraction. Changed
neighbour histories to `D` tokens (neighbour − serving) and added sinusoidal numeric
initialisation; trained 2 epochs (~37 min).

* v1 validation: accuracy 0.880, HO recall 0.135 (per-sample metrics did not improve).
* Closed loop (argmax/0.5 rule): HO 8.69, ping-pong 9.0 %, RLF 0.025, HOF 0.95,
  SE 2.936, outage 3.07 %. Committed results in `3ea00e9`.

User questions during this phase: *"how is the training going?"* (reported progress and
the weak epoch-1 numbers honestly) and *"where are the generated files located and
where are you running the training?"* (explained the ephemeral cloud container, that
`data/` and `checkpoints/` were git-ignored, and offered to commit the checkpoint).

README corrections: the first README example response was an invented "HANDOVER, 0.97"
output; it was replaced by the real output, and two subsequent wrong descriptions of
that example (what A3 would do) were corrected (`0ef22dd`, `c4ab21f`).

### 21.6 Why per-sample metrics are weak — label predictability check

Rule "HO to strongest neighbour if its last delta > x (and trend > y)" on a fresh
sample: accuracy 0.78–0.90, recall 0.18–0.45; always-STAY ≈ 0.90. Conclusion:
per-sample accuracy is dominated by the class prior and the labels contain
unpredictable future; judge in closed loop.

### 21.7 The decision-rule fix — `4579585`

**User:** *"commit the code and do the next step for less noisy label."*

Finding: the earlier threshold sweep had no effect because the rule required
`p_ho > p_stay` **and** `≥ threshold`. Changed to `p_ho(best) ≥ threshold`. Sweep on
the v1 model (§19.2) showed **0.35** beats all A3 settings on SE/outage/HOF → default.

### 21.8 Label-smoothing experiment (negative result)

Implemented `label_decision` (window / confirm). Evaluated two ways before training:

*Teacher quality in closed loop (3 drives × 64 UEs):*

| window, confirm | HO/UE/min | Ping-pong % | SE | Outage % |
|---|---:|---:|---:|---:|
| 0, 0 (raw) | 7.85 | 5.7 | 3.023 | 0.54 |
| 3, 0 | 8.05 | 7.4 | 2.988 | 1.54 |
| 5, 0 | 12.53 | 40.3 | 2.957 | 2.95 |
| **0, 20** | 7.01 | 1.3 | 3.022 | 0.61 |
| 3, 20 | 7.23 | 2.2 | 2.991 | 1.45 |
| 3, 30 | 6.13 | 0.8 | 2.991 | 1.46 |

*Learnability (MLP on report features, HO-to-strongest-neighbour, AUC):*
(0,0) 0.834 · (0,20) 0.830 · (0,30) 0.828 · (1,20) 0.814 · (2,20) 0.803 · (3,20) 0.790.

Windowing makes the teacher worse (early HO joins a not-yet-better cell); confirmation
cleans the teacher but is not more predictable. A model trained on (0, 20) labels
(1 epoch, the 2nd epoch was lost — §21.9) scored at threshold 0.35: HO 14.81,
ping-pong 23.7 %, RLF 0.019, HOF 0.62, SE 2.966, outage 2.11 % — slightly worse than
v1. **Decision:** keep options, default off, ship v1 (`601ba5b`), document as a
negative result in README.

### 21.9 Operational incidents

* A background training job hit the session's background time limit at step
  3 000/3 828 and no checkpoint had been written (only saved at the end). **Fix:** save
  after every epoch (`b235c89`) and run training as a detached process.
* The container restarted mid-epoch-2 of the retry; the epoch-1 checkpoint survived and
  was evaluated (§21.8).

### 21.10 Aligning the xApp with the benchmark — `601ba5b`

The HTTP path used greedy argmax generation (answered STAY for the example) while the
benchmarked policy used the 0.35 threshold (would hand over). `handle_report` now makes
the decision with the threshold rule and generates the rationale for that decision
(`prefix`); `--ho-threshold` / `--min-confidence` added to `serve` and `infer`.
The trained checkpoint was committed as requested.

### 21.11 Committing data — `b7d30cf`, `4caeaed`, `3134fc2`

* **User:** *"can you commit the training data as well."* The corpus on disk was the
  smoothed-label experiment corpus, so the raw-label corpus was regenerated from seed 0
  and verified identical to the original (same 527 512 rows and byte-identical file size)
  before committing.
* **User:** *"what about the reports used for simulating where i can find the data used
  for training"* → explained that reports existed only in memory/tokenised form; added
  `export-raw` (readable reports in the xApp request format, full drive channel,
  manifest) using the same random stream, and committed drive 0 (5 MB).
* **User:** *"yes and update the readme"*, then *"yes go ahead"* → added
  `gen-data --from-drives` for real traces (NaN handling, logged serving, replay mode),
  changed `run_policy`'s trajectory convention to "serving cell at report time", and
  verified the exported drive rebuilds all 569 HO labels bit-for-bit.

### 21.12 This document

**User:** *"create a detailed md file documenting everything … so either a human or
some other AI model can understand every bit … why, how, why not … design … all the
interfaces with diagrams"*, then *"use this chat history as well"* → `docs/DESIGN.md`.

### 21.13 Real-RAN integration (OCUDU / srsRAN, OAI)

**User:** *"can you add more details for 'Integrating with a real RAN' … add new code for
interacting with real ran … message structure, parsing … update the design doc … integrate
with real ran like ocudu / oai ran"*

**Research first.** Upstream sources were checked before writing code (§25):
* OAI: telnet `ci trigger_f1_ho [cu-ue-id]` and `ci trigger_n2_ho <pci>,<rrc-ue-id>`,
  `nr_measurement_configuration`, `nrRRC_stats.log`, FlexRIC `xapp_rc_handover`, and E2SM-RC
  "Handover Control" (RC v1.03).
* srsRAN's `oran-sc-ric`: `simple_rc_ho_xapp.py` → `e2sm_rc.control_handover(e2_node_id,
  amf_ue_ngap_id, gnb_cu_ue_f1ap_id, plmn, target_nr_cell_id)`.
* srsRAN `configs/mobility.yml` and the `ho <pci> <rnti> <pci>` console command.
* OCUDU is the Linux Foundation continuation of srsRAN.
* TS 38.133 report mappings.
* Some documentation sites (docs.srsran.com, docs.ocudu.org) were blocked from this
  environment, so those facts come from search snippets and GitHub sources.

**Built.** The `ai_ran_llm/ran/` package (§16.6), the CLI commands `ran-xapp`, `fake-gnb` and
`ran-parse`, the `integrations/` examples, `docs/RAN_INTEGRATION.md`, and 16 tests. Also a
refactor: `decide_reports` is now shared by the benchmark path and the xApp.

**Found and fixed along the way:**
1. **Clock mixing.** Outcomes carried wall-clock time and reports carried RAN time. The
   hold-off guard blocked 276 of 290 recommendations in the first end-to-end run. Fixed by
   running all timers on the report clock (D19). After the fix, the network path reproduced the
   benchmark exactly: identical trajectories on 16 UEs × 20 s, and identical KPIs on all 5
   benchmark drives.
2. **Padding split the handover probability.** With one reported neighbour (a 2-cell lab), the
   3 padded copies split P(target) four ways (0.03 each instead of 0.14), so no handover ever
   passed the threshold. Fixed by masking duplicate cell tokens in `score`.
3. **Confident out-of-distribution mistakes.** In the synthetic agent demo, SINR was fixed at
   +5 dB while the neighbour was 18 dB stronger, a combination never seen in training. The
   model stayed with P = 0.998, and the low-confidence fallback cannot catch that. Added the
   optional A3 override (D18) and made the demo's SINR physically consistent.
4. **No backstop.** The first OCUDU example config disabled the CU-CP's own handover, so an
   xApp outage would mean no handovers at all. Changed to keep A3 at 8 dB / 480 ms as a
   backstop.
5. **Operator readability.** Rationales named model cell indices ("cell 1"). The controller now
   appends the PCI ("cell 1 (PCI 2)").
6. **Test hygiene.** An unjoined daemon thread printed a C++ "terminate called" at interpreter
   exit; the tests now join the xApp thread.

**Measured.** Guard rails trade ping-pong for outage (§19.4). The model tolerates realistic
reporting well (200 ms, 8 neighbours, RRC-quantised).

### 21.14 Location, trajectory mining and the city model

**User:** *"what do you think if the measurement reports are accompanied with location info
(triangulation, angle of arrival) … a helper xApp in the RIC?"*, then *"dont change the code lets
finalize the concept"*, *"create another md file to capture this idea and findings"*, *"can this
help … predictive analytics / trajectory mining"*, and *"yes and can you implement [the]
simulator upgrades"*.

1. **Location study** (no code changes). A learnability test in the original simulator: position
   with 10–30 m error catches about 10–13 % more teacher handovers at equal precision, and 100 m
   error gives about 4 %. Captured in `docs/LOCATION_AWARE_HANDOVER.md`.
2. **Key argument.** Location derived only from the same RSRP reports adds almost nothing. Gains
   need new measurements (AoA, TA/RTT, beams, GNSS) or memory across UEs (a radio map, route
   statistics). The original simulator cannot show the latter: it has random movement and
   per-path shadowing.
3. **City model.** `city.py` adds road mobility with repeated routes and location-tied shadowing,
   both opt-in. The default simulator is unchanged: the committed corpus regenerates bit-for-bit,
   and a fingerprint test was added.
4. **Measured** (`experiments/location_trajectory_learnability.py`, 3 seeds, maps learned only
   from training drives). In the city: radio-map forecast +78 % recall (target accuracy 82 →
   89 %), position +27 %, handover-sequence prior with no location +16 %, all combined +83 %. In
   the original simulator, the radio map and trajectory prior add little, as predicted.

### 21.15 Location context in the model

**User:** *"can you do changes for [the next step]: the model doesn't use location yet"*

* **Implemented:**
  * opt-in context tokens (§12.6) and `location.py`: position track, radio map, handover-sequence
    mining, a service built from separate history drives, `LocationAwarePolicy`;
  * context-aware corpus generation (10 % context dropout, `obs_cfg` stored in the corpus),
    training that reads it, report-JSON context, CLI (`build-location-service`, `gen-data`
    flags, `evaluate --location-service`), 7 tests.
* **Checked:** the default corpus is still bit-identical; the shipped checkpoint is unaffected.
* **Experiment:** `experiments/location_closed_loop.py` trains 5 models on identical city drives
  (base, +position, +radio map, +trajectory, all; 1 epoch each) and benchmarks them in closed loop
  (`docs/LOCATION_AWARE_HANDOVER.md` §13).
* **Result, "all" vs base at the same budget:**
  * threshold 0.35: HOF −62 %, outage −41 %, handovers −24 %, RLF −45 %;
  * threshold 0.5: HOF −52 %, outage −34 %, RLF −51 %;
  * the radio map carries most of the gain.
* **Not a clean sweep:** the longer-trained shipped model remains competitive in the city (lower
  RLF and ping-pong). The city models need more training and per-model thresholds.
* **Follow-up, 3 epochs** (`docs/LOCATION_AWARE_HANDOVER.md` §13.1):
  * Offline, the base model catches up (recall 0.34 → 0.58), but the closed-loop gain holds. At
    0.5, radio map vs base: HOF −64 %, outage −43 %, RLF −94 %.
  * The radio-map model at 0.5 now beats the shipped model: 26 % fewer handovers, 47 % fewer
    HOFs, 28 % less outage, equal RLF, higher SE.
  * Ping-pong (about 19 % vs 8 % for A3 at 2 dB) is the remaining gap.
  * Lesson: judge context by closed-loop KPIs, not offline recall. Run parallel benchmarks with
    limited torch threads, or run them one at a time.

### 21.16 Lessons learned

1. Calibrate the simulator against a classical baseline *before* training anything.
2. Per-sample accuracy is misleading for rare, partly unpredictable events; evaluate in
   closed loop on identical drives.
3. The decision rule (threshold) mattered more than the input encoding or label
   smoothing.
4. Measure learnability cheaply (small classifier) before paying for a training run.
5. Save checkpoints per epoch; long jobs in ephemeral environments get killed.
6. Keep the deployed path (xApp) and the evaluated path (benchmark) on the same code.
7. Make data generation bit-reproducible; it made "commit the data" and "export the raw
   data" verifiable.
8. An end-to-end test that must reproduce a known result exactly (fake gNB = benchmark) finds
   integration bugs (clocks, routing, padding) that unit tests miss.
9. Handle *confident* errors separately from low-confidence ones. An ML controller in a live
   network needs an independent, simple safety net and a RAN-side backstop.
10. A simulator can only reveal benefits it models. Check what the simulator *cannot* express
    (here: place-tied shadowing, routes) before concluding an idea does not help.

---

## 22. Extension guide and invariants (for humans and AI agents)

### 22.1 Invariants — do not break silently

| Invariant | Why | If you must change it |
|---|---|---|
| Tokenizer vocabulary **order** | checkpoint embedding rows map to token ids | append new tokens at the end *and* resize/retrain the embedding; old checkpoints become incompatible otherwise |
| `prompt_len = 7 + H + K(2+H) + 1` and fixed positions | `score` finds neighbour cells by `<nbr>` positions; checkpoint was trained on this layout | retrain; pass the new `ObsConfig` into `train` |
| Order of random-number consumption in `iter_labelled_drives` / `_label_steps` / `generate_episode` | bit-for-bit corpus regeneration and raw-export alignment | re-generate and re-commit the corpus and `data/raw`; update §11.6/§18 numbers |
| `run_policy` trajectory = serving at report time | exported drives must rebuild identical reports | update `_logged_reports` and re-export |
| HTTP request format = `reports.jsonl` format | exported reports can be sent straight to the endpoint | change both + README |
| ran-bridge protocol `ai-ran-llm/ran-bridge/1` | RAN-side agents are written against it | add fields compatibly (optional); bump the version string for breaking changes |
| Guard timers use report timestamps | mixed clocks silently block or allow handovers | keep every timer on `MeasReport.timestamp_s` |
| Context tokens are appended and off by default | the shipped checkpoint (349 tokens, 41-token prompt) and corpus must keep working | add new input families at the end of the vocabulary, behind a flag; test that `HandoverTokenizer()` stays 349 / 41 |
| Default simulator fingerprint (`test_default_simulator_unchanged`) | corpus reproducibility and every published number | new simulator features must be opt-in and must not consume the default random stream |
| Fake gNB with guards off ≡ offline benchmark | the integration regression test (`test_fake_gnb_end_to_end_equals_offline_policy`) | if you change decision logic, change it in `decide_reports` so both paths share it |

### 22.2 Common tasks

* **More neighbours / longer history:** change `ObsConfig`, regenerate corpus, make
  `train` use the same `ObsConfig` (currently default), retrain; `block_size` must be
  ≥ 41-token prompt + ~12 answer tokens.
* **Add an input (e.g. RSRQ, CQI, beam id):** add a token family at the *end* of the
  vocabulary, extend `encode_prompts`, `prompt_len`, `report_to_observation`,
  `prompt_to_text`, the drive format, and retrain.
* **Train on real traces:** convert logs to the drive format (§16.5), then
  `gen-data --from-drives traces/*.npz` → `train --data …`.
* **Fine-tune a general LLM:** `export-jsonl`, then any SFT/LoRA tool; keep constrained
  decoding/validation on the output side.
* **Change the operating point:** `--ho-threshold` (≥ 0.5 fewer HOs, ≤ 0.3 more
  aggressive); `--min-confidence` routes uncertain cases to A3.
* **Add a baseline policy:** implement `decide(ep, t, obs)` (+ optional `reset`,
  `on_handover`) and add it to `evaluate.default_policies`.
* **Add a RAN / actuator:** implement `send_handover(cmd)` (raise on failure) using
  `command_fields(cmd)`; add a measurement source that yields `MeasReport`s into the runtime
  queue, or write a bridge agent. Test against fakes as `tests/test_ran.py` does.
* **Always run** `pytest -q` and `python -m ai_ran_llm evaluate` after changes.

### 22.3 Promising next steps

1. Model-side hysteresis (require the HO decision for 2 consecutive reports) to cut
   ping-pong.
2. Richer inputs (RSRQ/SINR per neighbour, beam measurements, heading/position).
3. Iterative DAgger: roll out the model, relabel its states with the teacher, retrain.
4. RL fine-tuning on closed-loop KPIs starting from the supervised model.
5. Pass `ObsConfig` through `train`/CLI; KV cache for rationale generation.
7. Location-aware handover and trajectory mining (`docs/LOCATION_AWARE_HANDOVER.md`).
   Evidence so far, in the simulated city (learnability, 3 seeds): a radio-map forecast +78 %
   recall at equal precision, position and heading +27 %, a handover-sequence prior with no
   location +16 %. Next: closed-loop training with location / map / trajectory tokens.
6. Shadow-mode evaluation on a live OCUDU/OAI testbed; a native FlexRIC actuator and an
   E2SM-RC REPORT (message copy) measurement source.

---

## 23. Testing

`pytest -q` — 42 tests (≈ 10 s): 14 in `tests/test_pipeline.py`, 16 in `tests/test_ran.py`,
5 in `tests/test_city.py` (default-simulator fingerprint, spatial field statistics, place-tied
shadowing across drives, road mobility, unknown-model errors), and 7 in `tests/test_location.py`
(token layouts and quantisers, radio map and transition model, service save/load, context values,
context corpus + training, closed-loop policy and report API).

`tests/test_pipeline.py`:

| Test | Checks |
|---|---|
| `test_hex_grid_has_19_cells` | layout |
| `test_episode_shapes_and_ranges` | channel shapes, plausible RSRP, finite SINR |
| `test_observation_excludes_serving_and_is_sorted` | report construction |
| `test_oracle_targets_are_reported_neighbours` | teacher only picks reported cells, gain > margin |
| `test_smoothed_labels` | (0,0) ≡ raw teacher; confirm only removes HOs; window only adds HOs |
| `test_closed_loop_metrics` | `run_policy` bookkeeping; teacher SE ≥ A3 |
| `test_tokenizer_layout_and_answer` | prompt layout, answer tokens, `explain` |
| `test_dataset_and_jsonl` | corpus shape/answers, JSONL export |
| `test_export_raw_matches_corpus` | raw export ↔ corpus alignment, file formats, reports are valid requests |
| `test_corpus_from_drive_files` | `--from-drives` reproduces HO labels; NaN traces; missing-serving error; replay mode |
| `test_loss_masking_and_training_step` | loss weights, loss decreases |
| `test_constrained_generation_only_targets_reported_cells` | decoding constraints; probabilities sum to 1 |
| `test_handle_report_and_policy` | xApp answer, A3 fallback, closed-loop LLM policy |
| `test_http_endpoint` | HTTP round trip |

`tests/test_ran.py`:

| Test | Checks |
|---|---|
| `test_38133_mappings` | RSRP/RSRQ/SINR index ↔ value |
| `test_parse_measurement_report[json/xer]` | the same report parsed from srsRAN-style JSON and asn1c XER |
| `test_log_blocks_and_source` | JSON/XER blocks in a log, header filter, UE ids, `--learn-id` |
| `test_protocol_roundtrip_and_errors` | NDJSON round trip; malformed messages rejected |
| `test_cell_map` | NCI/PCI resolution, duplicate PCI error, neighbour table, auto-add |
| `test_tracker_resamples_sample_and_hold` | resampling, unknown serving cell, stale neighbours |
| `test_controller_confirm_and_pending_and_hold_off` | guard order and timers |
| `test_controller_neighbour_backoff_dry_run_and_inferred_success` | NRT, failure backoff, shadow mode, success inferred from reports |
| `test_a3_override_catches_confident_stay` | override fires only when a neighbour leads; PCI in rationale |
| `test_short_reports_do_not_split_probability` | padding fix |
| `test_oai_telnet_actuator` | F1/N2 command lines, fake telnet server, error reply |
| `test_console_command_and_e2_actuators` | console FIFO line, no argument injection, exit codes, E2SM-RC call arguments |
| `test_bridge_errors_and_routing` | error reply, routing of commands to the reporting agent |
| `test_runtime_actuation_failure_backs_off` | actuation error → failure outcome → backoff; audit log |
| `test_fake_gnb_end_to_end_equals_offline_policy` | full network path ≡ `run_policy(LLMPolicy)` |

Tests use tiny randomly initialised models (or a stub) so they do not depend on the checkpoint.

---

## 24. Glossary

| Term | Meaning |
|---|---|
| **A3** | 3GPP measurement event: neighbour better than serving by an offset for TTT |
| **AI-RAN** | AI-native radio access network |
| **CHO** | conditional handover |
| **DAgger** | dataset aggregation: train on states visited by the learner, labelled by an expert |
| **E2 / E2SM-KPM / E2SM-RC** | O-RAN interface / service models for measurements and RAN control |
| **HO / HOF** | handover / handover failure |
| **Hysteresis (Hys)** | margin a neighbour must exceed |
| **L3 filter** | exponential smoothing of measurements configured by RRC `filterCoefficient` |
| **MRO** | mobility robustness optimisation (SON function tuning HO parameters) |
| **Near-RT RIC / xApp** | O-RAN controller with 10 ms–1 s loop / application running on it |
| **Ping-pong** | HO back to the previous cell within a short time (here 1 s) |
| **Qout** | SINR level corresponding to out-of-sync (≈ 10 % PDCCH BLER) |
| **RLF / T310** | radio link failure / timer started on out-of-sync, RLF on expiry |
| **RSRP** | reference signal received power (dBm) |
| **SE** | spectral efficiency, here `log2(1+SINR)` b/s/Hz |
| **SINR** | signal to interference plus noise ratio |
| **Teacher / oracle** | non-causal labeller that sees the future |
| **TTT** | time-to-trigger |
| **UE** | user equipment |
| **AMF UE NGAP ID / gNB-CU UE F1AP ID** | per-UE identifiers on NGAP / F1AP; E2SM-RC handover control on srsRAN/OCUDU addresses the UE with them |
| **CU-CP** | central unit control plane: runs RRC, decides handovers |
| **FlexRIC** | near-RT RIC and SDK from the OAI ecosystem |
| **NCI / NR-CGI** | NR Cell Identity (36 bit) / NR Cell Global Identity = PLMN + NCI |
| **NDJSON** | newline-delimited JSON, one object per line (ran-bridge framing) |
| **NRT** | neighbour relation table (allowed handover targets per cell) |
| **OCUDU** | Open Centralized Unit / Distributed Unit: the Linux Foundation continuation of srsRAN Project |
| **PCI** | physical cell id (0..1007), reused across the network |
| **RNTI (C-RNTI)** | cell radio network temporary identifier of a UE in a cell |
| **Shadow mode** | the xApp decides and logs but does not actuate (`dry_run`) |
| **XER** | XML encoding rules for ASN.1 (asn1c's `xer_fprint`) |

---

## 25. References

Standards and specifications (background for the models and parameter choices):

1. 3GPP TS 38.331, *NR; Radio Resource Control (RRC) protocol specification* —
   measurement events (A3), layer-3 filtering (`filterCoefficient`), T310/RLF.
2. 3GPP TS 36.331, *E-UTRA; RRC protocol specification* — LTE counterparts.
3. 3GPP TS 38.133, *NR; Requirements for support of radio resource management* —
   radio link monitoring (Qout/Qin).
4. 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects* —
   macro-cell path loss `128.1 + 37.6·log10(R[km])` and system-simulation assumptions.
5. 3GPP TR 36.839, *Mobility enhancements in heterogeneous networks* — handover
   failure / ping-pong definitions (minimum time of stay 1 s).
6. 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz* —
   modern channel models (not used directly; the simulator is simpler).
7. O-RAN Alliance WG3: near-RT RIC architecture, E2 application protocol,
   E2SM-KPM and E2SM-RC service models.
8. AI-RAN Alliance (2024–), AI-for-RAN / AI-on-RAN concepts.

Models and methods:

9. M. Gudmundson, "Correlation model for shadow fading in mobile radio systems,"
   *Electronics Letters*, 1991 — exponential shadowing correlation.
10. B. Liang and Z. J. Haas, "Predictive distance-based mobility management for PCS
    networks," *IEEE INFOCOM*, 1999 — Gauss-Markov mobility.
11. S. Ross, G. Gordon, J. A. Bagnell, "A reduction of imitation learning and structured
    prediction to no-regret online learning (DAgger)," *AISTATS*, 2011.
12. A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.
13. A. Radford et al., "Language models are unsupervised multitask learners" (GPT-2),
    2019.
14. A. Karpathy, *nanoGPT* (GitHub) — reference structure for the decoder.
15. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization (AdamW),"
    *ICLR*, 2019.

Real-RAN integration (verified sources for §16.6 and `docs/RAN_INTEGRATION.md`):

16. OpenAirInterface, `doc/handover-tutorial.md` (openairinterface5g, develop): F1/N2
    handover, telnet `ci trigger_f1_ho`, `ci trigger_n2_ho <pci>,<rrc-ue-id>`,
    `neighbour_list`, `nr_measurement_configuration`, FlexRIC `xapp_rc_handover`.
17. OpenAirInterface E2 agent merge request "Implement E2SM-RC On Demand and Handover Control"
    (RC v1.03: report style 5, control style 3 handover).
18. srsRAN, `oran-sc-ric` (github.com/srsran/oran-sc-ric): `xApps/python/simple_rc_ho_xapp.py`,
    `lib/e2sm_rc_module.py` (`control_handover` = control style 3, action 1).
19. srsRAN Project, `configs/mobility.yml`; gNB handover tutorial (`ho` console command);
    discussion #803 (neighbour measurement reports, RRC JSON logs).
20. OCUDU project (ocudu.org, docs.ocudu.org): the continuation of srsRAN Project under the Linux
    Foundation, release 26.04.
21. 3GPP TS 38.133 §10.1: SS-RSRP / SS-RSRQ / SS-SINR measurement report mapping.

The parameter values in §6 are typical textbook/system-simulation choices informed by
these sources, not values copied from a specific table unless stated.
