"""Lab drive (OAI + FlexRIC testbed relay) against the real ran-xapp runtime and a fake llm_bridge."""

import json
import socket
import threading
import time

from ai_ran_llm.ran.app import XAppOptions, build_xapp
from ai_ran_llm.ran.cells import CellMap
from ai_ran_llm.ran.controller import ControllerConfig
from ai_ran_llm.ran.labdrive import LabDrive, LabDriveConfig

CELLS = {"plmn": "00101", "cells": [
    {"index": 0, "pci": 0, "nci": "0x12345678", "gnb": "oai-cu", "e2_node_id": "001-01/3584", "neighbours": [1]},
    {"index": 1, "pci": 1, "nci": "0x11111111", "gnb": "oai-cu", "e2_node_id": "001-01/3584", "neighbours": [0]}]}


class FakeBridge:
    """Stands in for llm_bridge + OAI: reports the UE's serving cell, executes ho_commands."""

    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.rfile = self.sock.makefile("rb")
        self.serving = 0x12345678
        self.commands = []
        self.stop = threading.Event()
        self.send({"type": "hello", "node": "fake-llm_bridge"})
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._ctx, daemon=True).start()

    def send(self, m):
        self.sock.sendall((json.dumps(m) + "\n").encode())

    def _ctx(self):
        while not self.stop.is_set():
            self.send({"type": "ue_context", "ue_id": "amf=1", "ue_ids": {"amf_ue_ngap_id": 1, "ran_ue_id": 1},
                       "serving": {"nci": self.serving}})
            time.sleep(0.05)

    def _rx(self):
        for line in self.rfile:
            m = json.loads(line)
            if m["type"] == "ho_command":
                self.commands.append(m)
                self.serving = m["target_cell"]["nci"]      # OAI executes the F1 handover


def test_lab_drive_closed_loop(tmp_path):
    (tmp_path / "cells.json").write_text(json.dumps(CELLS))
    opt = XAppOptions(cells=str(tmp_path / "cells.json"), bridge="127.0.0.1:0", actuator="bridge",
                      audit=str(tmp_path / "audit.jsonl"), controller=ControllerConfig(dry_run=False))
    runtime, sources = build_xapp(opt)
    threading.Thread(target=runtime.run, daemon=True).start()
    cfg = LabDriveConfig(xapp=f"127.0.0.1:{sources[0].port}", listen="127.0.0.1:0", isd_m=300,
                         speed_kmh=60, time_scale=4.0, report_period_s=0.2, seed=1)
    drive = LabDrive(CellMap.from_dict(CELLS), cfg).start()
    bridge = FakeBridge(drive.listen_port)
    try:
        time.sleep(9.0)                                    # ~36 s virtual: one pass 0 -> 1 and back towards 0
    finally:
        s = drive.summary()
        bridge.stop.set()
        drive.stop()
        runtime.stop()
        runtime.close()
    print(json.dumps({k: v for k, v in s.items() if k != 'events'}), *s['events'], sep='\n')
    assert s["reports_synthetic"] > 100 and s["ue_contexts"] > 0
    assert s["ho_commands"] >= 1 and s["handovers_executed"] >= 1
    first = next(e for e in s["events"] if e["event"] == "ho_command")
    assert first["to"] == 1 and first["x_m"] > 100          # hands over towards cell 1 after leaving cell 0
    assert bridge.commands[0]["ue_ids"] == {"amf_ue_ngap_id": 1, "ran_ue_id": 1}
    assert bridge.commands[0]["target_cell"]["plmn"] == "00101"


def test_ploss_mapping():
    d = LabDrive(CellMap.from_dict(CELLS), LabDriveConfig(ploss_at_ref_db=20, rsrp_ref_dbm=-80, ploss_max_db=60))
    assert d.ploss_for(-80) == 20 and d.ploss_for(-100) == 40 and d.ploss_for(-200) == 60 and d.ploss_for(0) == 0


def test_unreachable_ue_telnet_does_not_slow_reports(tmp_path):
    """RF-simulator coupling runs in its own thread: a dead UE telnet must not delay reports."""
    got = []
    srv = socket.create_server(("127.0.0.1", 0))               # stands in for ran-xapp

    def xapp():
        conn, _ = srv.accept()
        for line in conn.makefile("rb"):
            if json.loads(line).get("type") == "meas_report":
                got.append(1)

    threading.Thread(target=xapp, daemon=True).start()
    cfg = LabDriveConfig(xapp=f"127.0.0.1:{srv.getsockname()[1]}", listen="127.0.0.1:0", report_period_s=0.1,
                         ue_telnet="10.255.255.1:8091", channels={0: 0, 1: 1}, ploss_period_s=0.1)
    drive = LabDrive(CellMap.from_dict(CELLS), cfg).start()
    b = socket.create_connection(("127.0.0.1", drive.listen_port))
    b.sendall((json.dumps({"type": "ue_context", "ue_id": "u", "serving": {"nci": "0x12345678"}}) + "\n").encode())
    time.sleep(3.0)
    drive.stop()
    assert len(got) >= 20                                       # ~30 at 10 Hz; a blocking telnet would give ~3
