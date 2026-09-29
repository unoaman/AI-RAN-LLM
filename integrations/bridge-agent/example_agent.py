#!/usr/bin/env python3
"""Template for a RAN-side ran-bridge agent.

Use this when neither the RRC-log source nor E2 fits your RAN: run it next to
(or inside) the gNB / CU-CP, feed it measurements from wherever your RAN exposes
them (a hook in the RRC code, a metrics socket, E2SM-RC "message copy"
reports decoded by your RIC, ...), and let it execute the handover commands.

Protocol: docs/RAN_INTEGRATION.md §4 (NDJSON over TCP). This file only uses the
standard library + ai_ran_llm.ran (BridgeClient, message classes); a C/C++ agent
just writes the same JSON lines.

Try it against the xApp:
    python -m ai_ran_llm ran-xapp --cells integrations/ocudu/cells.example.json \\
        --actuator bridge --live --confirm 1 --hold-off-s 0
    python integrations/bridge-agent/example_agent.py --demo
"""

import argparse
import math
import threading
import time

from ai_ran_llm.ran.bridge import BridgeClient
from ai_ran_llm.ran.messages import CellMeas, HandoverCommand, HandoverOutcome, MeasReport, UERelease


SERVING = {}          # demo state: UE -> serving PCI


def execute_handover(cmd: HandoverCommand) -> tuple[str, str]:
    """Replace with your RAN's handover trigger. Return (status, detail);
    status is one of success / failure / rejected / timeout."""
    SERVING[cmd.ue_id] = cmd.target_cell.pci
    print(f"[agent] handover {cmd.ue_id}: PCI {cmd.source_cell.pci} -> PCI {cmd.target_cell.pci} "
          f"(NCI {cmd.target_cell.nci:#x}, conf {cmd.confidence:.2f}): {cmd.rationale}")
    return "success", ""


def listen(client: BridgeClient):
    while True:
        msg = client.recv()
        if msg is None:
            return
        if isinstance(msg, HandoverCommand):
            status, detail = execute_handover(msg)
            client.send(HandoverOutcome(msg.command_id, msg.ue_id, status, detail))
        elif isinstance(msg, dict) and msg.get("type") == "error":
            print("[agent] xApp reported a protocol error:", msg.get("detail"))


def demo(client: BridgeClient):
    """One UE driving from PCI 1 towards PCI 2: serving falls, neighbour rises."""
    ue_ids = {"rnti": 0x4601, "amf_ue_ngap_id": 1, "gnb_cu_ue_f1ap_id": 1, "cu_ue_id": 1}
    SERVING["ue-1"] = 1
    for k in range(40):
        serving = SERVING["ue-1"]
        a, b = -85.0 - 0.6 * k, -105.0 + 0.6 * k
        rsrp = {1: a, 2: b}
        other = b if serving == 1 else a
        sinr = 10 * math.log10(10 ** (rsrp[serving] / 10) / (0.7 * 10 ** (other / 10) + 10 ** (-12.5)))
        client.send(MeasReport(ue_id="ue-1", serving=CellMeas(pci=serving, rsrp_dbm=rsrp[serving], sinr_db=sinr),
                               neighbours=[CellMeas(pci=p, rsrp_dbm=v) for p, v in rsrp.items() if p != serving],
                               timestamp_s=k * 0.12, ue_ids=ue_ids, speed_kmh=60.0, seq=k))
        time.sleep(0.12)
    client.send(UERelease("ue-1"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xapp", default="127.0.0.1:7000")
    p.add_argument("--demo", action="store_true", help="send a synthetic UE trajectory")
    a = p.parse_args()
    host, _, port = a.xapp.rpartition(":")
    client = BridgeClient(host, int(port), node="example-agent", timeout=None)
    threading.Thread(target=listen, args=(client,), daemon=True).start()
    if a.demo:
        demo(client)
        time.sleep(0.5)
    else:
        print("connected; wire your RAN's measurements into client.send(MeasReport(...))")
        threading.Event().wait()


if __name__ == "__main__":
    main()
