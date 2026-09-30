# HandoverLLM on OAI + FlexRIC: end-to-end testbed

Everything in Docker on one Linux host:

* **5G core, RAN and UE:** OAI's 5G core, a CU plus two DUs over the RF simulator, and an nrUE.
* **RIC:** the FlexRIC nearRT-RIC.
* **E2 side of HandoverLLM:** the `llm_bridge` xApp (E2SM-RC).
* **The model:** `ran-xapp` (HandoverLLM).
* **Synthetic radio:** `ran-lab-drive`, because OAI's software UE sends no MeasurementReports.

The model's handover decisions go out as E2SM-RC Handover Control, and the OAI CU executes
them as real F1 handovers. The full description is in
[`docs/RAN_INTEGRATION.md` §9.7](../../../docs/RAN_INTEGRATION.md).

```bash
sudo modprobe sctp          # OAI (F1, NGAP) and FlexRIC (E2) need kernel SCTP
./run.sh all                # fetch OAI configs, build, start in order, drive 240 s, check every hop
./run.sh logs               # follow bridge / lab drive / CU handover lines
./run.sh down
```

| File | What |
|---|---|
| `docker-compose.yml` | OAI F1 RF-simulator CI deployment, plus nearRT-RIC, `llm-bridge`, `ran-xapp` and `lab-drive` |
| `Dockerfile.flexric` | FlexRIC `d7a71285` (the commit OAI pins) + `llm_bridge`; runs the bridge's offline tests |
| `Dockerfile.ai_ran_llm` | `ran-xapp` / `ran-lab-drive` with the shipped model (CPU torch) |
| `fetch_oai_conf.sh` | OAI CU/DU/UE/core configs at a pinned OAI commit; adds `e2_agent` to the CU |
| `flexric.conf` | nearRT-RIC address for the RIC and the xApps |
| `run.sh` | `up`, `test`, `all`, `logs`, `down` |
| `out/` | logs, audit log, lab-drive summary (created by `run.sh`) |

Settings: `DURATION_S` (240), `SPEED_KMH` (30), `PLOSS_MAX_DB` (45; 55+ loses sync), `LIVE=` for
shadow mode, `OAI_TAG` (2026.w39).

Requirements: Docker with Compose v2, about 8 GB RAM and 20 GB disk. The first build takes
about 15 minutes.

**Status:** not yet run end to end. It was written on a machine whose kernel lacks SCTP, where
the compose file validates, both images build, and the bridge's tests pass inside the image.
Please report what `./run.sh test` prints.
