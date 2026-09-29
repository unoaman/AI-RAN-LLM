"""Tests for the real-RAN integration layer (ai_ran_llm.ran)."""

import json
import queue
import socket
import sys
import threading

import numpy as np
import pytest
import torch

from ai_ran_llm.config import ModelConfig, ObsConfig
from ai_ran_llm.inference import HandoverLLM, LLMPolicy
from ai_ran_llm.model import HandoverGPT
from ai_ran_llm.ran.actuators import (ActuationError, CommandActuator, ConsoleActuator, OAITelnetActuator,
                                      OranScRicActuator)
from ai_ran_llm.ran.bridge import BridgeClient, BridgeServer
from ai_ran_llm.ran.cells import CellMap
from ai_ran_llm.ran.controller import ControllerConfig, HandoverController
from ai_ran_llm.ran.fake_gnb import FakeGnb
from ai_ran_llm.ran.messages import (CellMeas, CellRef, HandoverCommand, HandoverOutcome, MeasReport, ProtocolError,
                                     decode, encode)
from ai_ran_llm.ran.rrc import (iter_asn1_blocks, parse_measurement_report, rsrp_dbm, rsrp_index, rsrq_db, sinr_db,
                                sinr_index, xer_to_dict)
from ai_ran_llm.ran.runtime import XAppRuntime
from ai_ran_llm.ran.sources import RrcLogSource
from ai_ran_llm.ran.tracker import UEMeasurementTracker
from ai_ran_llm.simulator import generate_episode, run_policy
from ai_ran_llm.tokenizer import HandoverTokenizer

# An RRC MeasurementReport as srsRAN / OCUDU print it (ASN.1 field names, JSON)
SRS_JSON = {"UL-DCCH-Message": {"message": {"c1": {"measurementReport": {"criticalExtensions": {"measurementReport": {
    "measResults": {
        "measId": 1,
        "measResultServingMOList": [{"servCellId": 0, "measResultServingCell": {
            "physCellId": 1, "measResult": {"cellResults": {"resultsSSB-Cell": {"rsrp": 60, "rsrq": 70, "sinr": 80}}}}}],
        "measResultNeighCells": {"measResultListNR": [
            {"physCellId": 2, "measResult": {"cellResults": {"resultsSSB-Cell": {"rsrp": 65}}}},
            {"physCellId": 3, "measResult": {"cellResults": {"resultsSSB-Cell": {"rsrp": 50, "rsrq": 60}}}}]}}}}}}}}}

# The same report as asn1c XER (as OAI's asn1c-based RRC prints it)
XER = """<MeasurementReport>
  <criticalExtensions><measurementReport><measResults>
    <measId>1</measId>
    <measResultServingMOList><MeasResultServMO>
      <servCellId>0</servCellId>
      <measResultServingCell><physCellId>1</physCellId>
        <measResult><cellResults><resultsSSB-Cell><rsrp>60</rsrp><rsrq>70</rsrq><sinr>80</sinr></resultsSSB-Cell></cellResults></measResult>
      </measResultServingCell>
    </MeasResultServMO></measResultServingMOList>
    <measResultNeighCells><measResultListNR>
      <MeasResultNR><physCellId>2</physCellId><measResult><cellResults><resultsSSB-Cell><rsrp>65</rsrp></resultsSSB-Cell></cellResults></measResult></MeasResultNR>
      <MeasResultNR><physCellId>3</physCellId><measResult><cellResults><resultsSSB-Cell><rsrp>50</rsrp><rsrq>60</rsrq></resultsSSB-Cell></cellResults></measResult></MeasResultNR>
    </measResultListNR></measResultNeighCells>
  </measResults></measurementReport></criticalExtensions>
</MeasurementReport>"""


def _tiny_llm():
    tok = HandoverTokenizer()
    torch.manual_seed(0)
    return HandoverLLM(HandoverGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=63, n_layer=1, n_head=2,
                                               n_embd=32)), tok)


def _cells():
    return CellMap.from_dict({"plmn": "00101", "cells": [
        {"pci": 1, "nci": "0x66C000", "gnb": "g1", "e2_node_id": "gnbd_001_001_00019b_0", "neighbours": [1, 2]},
        {"pci": 2, "nci": "0x66C001", "gnb": "g1", "e2_node_id": "gnbd_001_001_00019b_0", "neighbours": [0]},
        {"pci": 3, "nci": "0x77C000", "gnb": "g2", "neighbours": [0]}]})


def _report(t, serving=1, rsrp=None, ue="ue-1", seq=None):
    rsrp = rsrp or {1: -90.0, 2: -95.0, 3: -100.0}
    return MeasReport(ue_id=ue, serving=CellMeas(pci=serving, rsrp_dbm=rsrp[serving], sinr_db=3.0),
                      neighbours=[CellMeas(pci=p, rsrp_dbm=v) for p, v in rsrp.items() if p != serving],
                      timestamp_s=t, ue_ids={"rnti": 0x4601, "cu_ue_id": 7, "amf_ue_ngap_id": 5,
                                             "gnb_cu_ue_f1ap_id": 9}, seq=seq)


# --------------------------------------------------------------------- RRC

def test_38133_mappings():
    assert rsrp_dbm(1) == -156 and rsrp_dbm(126) == -31 and rsrp_dbm(0) == -157 and rsrp_dbm(200) == -30
    assert rsrq_db(127) == 20 and rsrq_db(1) == -43
    assert sinr_db(127) == 40 and sinr_db(1) == -23
    for v in (-140.3, -95.0, -44.9):
        assert rsrp_dbm(rsrp_index(v)) <= v < rsrp_dbm(rsrp_index(v)) + 1
    assert sinr_db(sinr_index(3.3)) == 3.0


@pytest.mark.parametrize("msg", [SRS_JSON, xer_to_dict(XER)], ids=["json", "xer"])
def test_parse_measurement_report(msg):
    p = parse_measurement_report(msg)
    assert p.meas_id == 1
    (sid, serv), = p.serving
    assert sid == 0 and serv.pci == 1 and serv.rsrp_dbm == -97 and serv.rsrq_db == -8.5 and serv.sinr_db == 16.5
    assert [(n.pci, n.rsrp_dbm, n.rsrq_db) for n in p.neighbours] == [(2, -92, None), (3, -107, -13.5)]
    assert parse_measurement_report({"foo": 1}) is None


def test_log_blocks_and_source(tmp_path):
    body = json.dumps(SRS_JSON, indent=2)
    log = ("2026-09-29T10:00:00 [RRC] [I] ue=3 c-rnti=0x4601 pci=1: Rx measurementReport\n" + body + "\n"
           "2026-09-29T10:00:01 [RRC] [I] ue=4 unrelated\n{\"other\": 1}\n"
           "2026-09-29T10:00:02 [RRC] [I] ue=5 rnti=0x4602 Rx measurementReport\n" + XER + "\n")
    blocks = list(iter_asn1_blocks(log.splitlines(True)))
    assert len(blocks) == 3 and "ue=3" in blocks[0].header and "MeasurementReport" in blocks[2].body
    assert len(list(iter_asn1_blocks(log.splitlines(True), header_filter="measurementReport"))) == 2
    path = tmp_path / "gnb.log"
    path.write_text("2026-09-29T09:59:59 [NGAP] [I] ue=3 amf_ue_id=17 ran_ue_id=3: InitialUEMessage\n" + log)
    src = RrcLogSource(str(path), follow=False, clock=lambda: 12.5,
                       learn_patterns=[r"ue=(?P<ue>\d+).*?amf_ue_id=(?P<amf_ue_ngap_id>\d+)"])
    reps = list(src.reports())
    assert [r.ue_id for r in reps] == ["ue=3", "ue=5"]
    assert reps[0].ue_ids == {"ue_index": 3, "rnti": 0x4601, "amf_ue_ngap_id": 17} and reps[0].serving.pci == 1
    assert reps[0].timestamp_s == 12.5 and [n.pci for n in reps[1].neighbours] == [2, 3]


# ---------------------------------------------------------------- protocol

def test_protocol_roundtrip_and_errors():
    rep = _report(1.0, seq=4)
    back = decode(encode(rep))
    assert back == rep
    cmd = HandoverCommand("ue-1", {"rnti": 1}, CellRef(0, pci=1), CellRef(1, pci=2, nci=0x66C001), 0.8)
    assert decode(encode(cmd)).to_dict() == cmd.to_dict()
    assert decode(encode(HandoverOutcome("c1", "ue-1", "success"))).status == "success"
    for bad in (b"not json", b"[1]", b'{"type": "nope"}', b'{"type": "meas_report", "ue_id": "u"}',
                b'{"type": "meas_report", "ue_id": "u", "serving": {"rsrp_dbm": -90}}',
                b'{"type": "ho_outcome", "command_id": "c", "ue_id": "u", "status": "maybe"}'):
        with pytest.raises(ProtocolError):
            decode(bad)


# ---------------------------------------------------------------- cells / tracker

def test_cell_map():
    cm = _cells()
    assert cm.resolve(CellMeas(nci=0x66C001)).index == 1 and cm.resolve(CellMeas(pci=3)).index == 2
    assert cm.resolve(CellMeas(pci=99)) is None
    assert cm.is_neighbour(0, 2) and not cm.is_neighbour(1, 2)
    with pytest.raises(ValueError):
        CellMap.from_dict({"cells": [{"pci": 1}, {"pci": 1}]})
    auto = CellMap.from_dict({"cells": []}, auto_add=True)
    assert auto.resolve(CellMeas(pci=42)).index == 0 and auto.resolve(CellMeas(pci=43)).index == 1


def test_tracker_resamples_sample_and_hold():
    tr = UEMeasurementTracker(_cells(), ObsConfig())
    for t, a, b in [(0.0, -90, -100), (0.25, -91, -98), (0.5, -92, -96), (0.9, -94, -93)]:
        assert tr.update(_report(t, rsrp={1: a, 2: b, 3: -120}))
    assert tr.ready("ue-1")
    # samples at 0.9-0.8=0.1, 0.3, 0.5, 0.7, 0.9 -> hold the latest value at or before each instant
    assert tr.history("ue-1", 0) == [-90, -91, -92, -92, -94]
    rep = tr.build_report("ue-1")
    assert rep["serving_cell"] == 0 and {n["cell_id"] for n in rep["neighbors"]} == {1, 2}
    assert not tr.update(_report(1.0, serving=9, rsrp={9: -80, 1: -90}))          # unknown serving cell
    tr.update(MeasReport("ue-1", CellMeas(pci=1, rsrp_dbm=-95), [CellMeas(pci=2, rsrp_dbm=-90)], timestamp_s=4.0))
    assert [n["cell_id"] for n in tr.build_report("ue-1")["neighbors"]] == [1]     # cell 3 went stale


# ---------------------------------------------------------------- controller

class StubLLM:
    """Always recommends a handover to model cell `target` with probability `p`."""

    def __init__(self, target=1, p=0.9):
        self.tok = HandoverTokenizer()
        self.target, self.p = target, p

    def decide_reports(self, reports, ho_threshold, min_confidence, a3_hyst_db, a3_ttt, explain):
        ho = self.p >= ho_threshold
        return [{"action": "HANDOVER" if ho else "STAY", "target_cell": self.target if ho else None,
                 "rationale": "stub", "confidence": self.p if ho else 1 - self.p, "p_stay": 1 - self.p,
                 "p_handover": {self.target: self.p}, "source": "llm"} for _ in reports]


def _ctl(**cfg):
    return HandoverController(StubLLM(), _cells(), ControllerConfig(**{"dry_run": False, **cfg}))


def test_controller_confirm_and_pending_and_hold_off():
    ctl = _ctl(confirm_count=2, hold_off_s=1.0)
    (d, c), = ctl.on_reports([_report(0.0)])
    assert d.action == "STAY" and d.reason == "confirm" and c is None
    (d, c), = ctl.on_reports([_report(0.1)])
    assert d.action == "HANDOVER" and c.target_cell.pci == 2 and c.source_cell.pci == 1
    assert c.ue_ids["amf_ue_ngap_id"] == 5 and c.target_cell.nci == 0x66C001
    assert ctl.on_reports([_report(0.2)])[0][0].reason == "pending"
    ctl.on_outcome(HandoverOutcome(c.command_id, "ue-1", "success"))
    ctl.cells.neighbours.clear()
    ctl.llm.target = 0
    # hold-off is checked before confirm; the confirmation streak keeps counting meanwhile
    assert ctl.on_reports([_report(0.3, serving=2)])[0][0].reason == "hold_off"     # 0.1 s after the handover
    assert ctl.on_reports([_report(0.4, serving=2)])[0][0].reason == "hold_off"
    assert ctl.on_reports([_report(1.4, serving=2)])[0][0].action == "HANDOVER"


def test_controller_neighbour_backoff_dry_run_and_inferred_success():
    ctl = _ctl()
    ctl.llm.target = 2                                   # PCI 3 is not a neighbour of PCI 2
    assert ctl.on_reports([_report(0.0, serving=2)])[0][0].reason == "not_neighbour"
    ctl.llm.target = 1
    d, c = ctl.on_reports([_report(0.1)])[0]
    ctl.on_outcome(HandoverOutcome(c.command_id, "ue-1", "failure"))
    assert ctl.on_reports([_report(0.2)])[0][0].reason == "failure_backoff"
    ctl2 = _ctl()
    d, c = ctl2.on_reports([_report(0.0)])[0]
    ctl2.on_reports([_report(0.1, serving=2)])       # report from the target: success inferred
    assert ctl2.state["ue-1"].pending is None
    shadow = _ctl(dry_run=True)
    d, c = shadow.on_reports([_report(0.0)])[0]
    assert c.dry_run and shadow.state["ue-1"].pending is None


def test_a3_override_catches_confident_stay():
    ctl = _ctl(a3_override_db=6.0)
    ctl.llm.p = 0.01                                      # model: stay, very confident
    for k in range(4):                                    # neighbour PCI 2 leads by 10 dB throughout
        d, c = ctl.on_reports([_report(0.1 * k, rsrp={1: -100.0, 2: -90.0, 3: -120.0})])[0]
        if c is not None:
            break
    assert d.action == "HANDOVER" and d.reason == "a3_override" and c.target_cell.pci == 2
    assert "cell 1 (PCI 2)" in c.rationale
    assert ctl.stats.overrides == 1
    quiet = _ctl(a3_override_db=6.0)
    quiet.llm.p = 0.01
    assert quiet.on_reports([_report(0.0)])[0][0].action == "STAY"   # neighbour weaker: no override


def test_short_reports_do_not_split_probability():
    llm = _tiny_llm()
    one = {"serving_cell": 0, "serving_rsrp": [-95] * 5, "sinr_db": -3, "neighbors": [{"cell_id": 1, "rsrp": [-90] * 5}]}
    ans = llm.decide_reports([one], ho_threshold=0.0, explain=False)[0]
    p_stay, p_ho = ans["p_stay"], ans["p_handover"]
    assert list(p_ho) == [1] and abs(p_stay + p_ho[1] - 1) < 1e-3   # all HO mass on the one real neighbour


# ---------------------------------------------------------------- actuators

def _cmd(source_gnb="g1", target_gnb="g1"):
    return HandoverCommand("ue-1", {"rnti": "0x4601", "cu_ue_id": 7, "rrc_ue_id": 3, "amf_ue_ngap_id": 5,
                                    "gnb_cu_ue_f1ap_id": 9},
                           CellRef(0, pci=1, nci=0x66C000, plmn="00101", gnb=source_gnb, e2_node_id="gnbd_x"),
                           CellRef(1, pci=2, nci=0x66C001, plmn="00101", gnb=target_gnb), 0.9)


def _fake_telnet(reply=b"ok\n"):
    srv = socket.create_server(("127.0.0.1", 0))
    got = []

    def serve():
        conn, _ = srv.accept()
        with conn:
            conn.sendall(b"softmodem_gnb> ")
            got.append(conn.recv(1024).decode())
            conn.sendall(reply)
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], got


def test_oai_telnet_actuator():
    act = OAITelnetActuator(port=0)
    assert act.command_line(_cmd()) == "ci trigger_f1_ho 7"
    assert act.command_line(_cmd(target_gnb="g2")) == "ci trigger_n2_ho 2,3"
    port, got = _fake_telnet()
    OAITelnetActuator("127.0.0.1", port).send_handover(_cmd())
    assert got == ["ci trigger_f1_ho 7\n"]
    port, _ = _fake_telnet(b"error: UE not found\n")
    with pytest.raises(ActuationError):
        OAITelnetActuator("127.0.0.1", port).send_handover(_cmd())


def test_console_command_and_e2_actuators(tmp_path):
    fifo = tmp_path / "gnb_cmd"
    ConsoleActuator(str(fifo)).send_handover(_cmd())
    assert fifo.read_text() == "ho 1 0x4601 2\n"
    out = tmp_path / "argv.txt"
    act = CommandActuator(f'{sys.executable} -c "import sys; open(sys.argv[1], \'w\').write(repr(sys.argv[2:]))" '
                          f'{out} --ue {{ue_id}} --nci {{target_nci_hex}}')
    cmd = _cmd()
    cmd.ue_id = "ue-1; rm -rf /"                         # must stay one argument
    act.send_handover(cmd)
    assert out.read_text() == repr(["--ue", "ue-1; rm -rf /", "--nci", "0x66c001"])
    with pytest.raises(ActuationError):
        CommandActuator(f"{sys.executable} -c 'import sys; sys.exit(3)'").send_handover(_cmd())

    class FakeRC:
        calls = []

        def control_handover(self, *a):
            self.calls.append(a)

    xapp = type("X", (), {"e2sm_rc": FakeRC()})()
    OranScRicActuator(xapp).send_handover(_cmd())
    assert FakeRC.calls == [("gnbd_x", 5, 9, "00101", 0x66C001)]


# ---------------------------------------------------------------- bridge / runtime / end to end

def test_bridge_errors_and_routing():
    srv = BridgeServer("127.0.0.1", 0).start()
    cli = BridgeClient("127.0.0.1", srv.port, node="t")
    cli.sock.sendall(b"garbage\n")
    assert cli.recv()["type"] == "error"
    cli.send(_report(0.0))
    src, msg = srv.events.get(timeout=5)
    assert src is srv and msg.ue_id == "ue-1"
    srv.send_handover(_cmd())
    assert isinstance(cli.recv(), HandoverCommand)
    with pytest.raises(ConnectionError):
        srv.send_handover(HandoverCommand("nobody", {}, CellRef(0), CellRef(1), 0.5))
    cli.close()
    srv.close()


def test_runtime_actuation_failure_backs_off(tmp_path):
    class Broken:
        def send_handover(self, cmd):
            raise ActuationError("RAN unreachable")
    ctl = _ctl()
    audit = tmp_path / "audit.jsonl"
    rt = XAppRuntime(ctl, Broken(), queue.Queue(), audit_path=str(audit))
    rt.process([(None, _report(0.0))])
    rt.process([(None, _report(0.1))])
    rt.close()
    recs = [json.loads(line) for line in audit.read_text().splitlines()]
    assert "actuation failed" in recs[0]["reason"] and recs[1]["reason"] == "failure_backoff"


def test_fake_gnb_end_to_end_equals_offline_policy():
    llm = _tiny_llm()
    ep = generate_episode(4, 60, np.random.default_rng(3))
    _, ref = run_policy(ep, LLMPolicy(llm, 0.35), ObsConfig())
    srv = BridgeServer("127.0.0.1", 0).start()
    cells = CellMap.for_simulator(ep.n_cells)
    ctl = HandoverController(llm, cells, ControllerConfig(confirm_count=1, hold_off_s=0, min_confidence=0,
                                                          dry_run=False))
    rt = XAppRuntime(ctl, actuator=srv, events=srv.events)
    th = threading.Thread(target=rt.run, daemon=True)
    th.start()
    _, traj = FakeGnb(ep, "127.0.0.1", srv.port, cells).run()
    rt.stop()
    th.join(timeout=5)
    srv.close()
    assert np.array_equal(traj, ref)
    assert ctl.stats.reports == 4 * 60
