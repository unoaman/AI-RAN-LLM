#!/usr/bin/env python3
"""HandoverLLM as an xApp on the O-RAN SC near-RT RIC shipped by srsRAN
(https://github.com/srsran/oran-sc-ric), for OCUDU / srsRAN Project gNBs.

Handovers are sent as E2SM-RC RIC Control Requests (Control Style 3
"Connected Mode Mobility", Action 1 "Handover Control") through the
framework's ``e2sm_rc.control_handover(e2_node_id, amf_ue_ngap_id,
gnb_cu_ue_f1ap_id, plmn, target_nr_cell_id)`` — the same call as the repo's
``simple_rc_ho_xapp.py``.

Measurements come from the ran-bridge (a RAN-side agent) and/or by tailing the
CU-CP log for RRC MeasurementReports; E2SM-KPM does not carry neighbour-cell
RSRP, so it is not used as the measurement source.

Run inside the ``python_xapp_runner`` container, with this repository mounted
and its dependencies installed (``pip install numpy torch`` + ``pip install -e``):

    ./handover_llm_xapp.py --cells /work/integrations/ocudu/cells.example.json \\
        --ckpt /work/checkpoints/handover_llm.pt --rrc-log /logs/gnb.log \\
        --learn-id 'ue=(?P<ue>\\d+).*?amf_ue_id=(?P<amf_ue_ngap_id>\\d+)' \\
        --learn-id 'ue=(?P<ue>\\d+).*?cu_ue_id=(?P<gnb_cu_ue_f1ap_id>\\d+)' \\
        --live

Status: written against the xAppBase API of srsran/oran-sc-ric (verified from
its example xApps); not run against a live RIC in this repository's CI.
Start without ``--live`` (shadow mode) and check the audit log first.
"""

import argparse
import logging
import signal
import sys

from lib.xAppBase import xAppBase          # provided by the oran-sc-ric xApp runner image

from ai_ran_llm.ran.actuators import OranScRicActuator
from ai_ran_llm.ran.app import XAppOptions, build_xapp
from ai_ran_llm.ran.controller import ControllerConfig
from ai_ran_llm.ran.sources import DEFAULT_ID_PATTERNS


class HandoverLLMXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port, options: XAppOptions):
        super().__init__(config, http_server_port, rmr_port)
        self.options = options
        self.runtime = None

    @xAppBase.start_function
    def start(self):
        self.runtime, self.sources = build_xapp(self.options, actuator=OranScRicActuator(self))
        try:
            self.runtime.run()                  # blocks until stop()
        finally:
            self.runtime.close()
            for s in self.sources:
                s.close()
        self.running = False

    def signal_handler(self, sig, frame):
        if self.runtime is not None:
            self.runtime.stop()
        super().signal_handler(sig, frame)


def main():
    p = argparse.ArgumentParser(description="HandoverLLM xApp (E2SM-RC handover control)")
    p.add_argument("--config", default="")
    p.add_argument("--http_server_port", type=int, default=8092)
    p.add_argument("--rmr_port", type=int, default=4562)
    p.add_argument("--ran_func_id", type=int, default=3, help="E2SM-RC RAN function id (srsRAN default 3)")
    p.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    p.add_argument("--cells", required=True)
    p.add_argument("--bridge", default="0.0.0.0:7000", help="ran-bridge listen address ('off' to disable)")
    p.add_argument("--rrc-log")
    p.add_argument("--learn-id", action="append", default=[])
    p.add_argument("--id-regex", action="append", default=[], metavar="KEY=REGEX")
    p.add_argument("--live", action="store_true")
    p.add_argument("--ho-threshold", type=float, default=0.35)
    p.add_argument("--confirm", type=int, default=1)
    p.add_argument("--hold-off-s", type=float, default=0.0)
    p.add_argument("--a3-override-db", type=float, default=6.0)
    p.add_argument("--audit", default="ran_audit.jsonl")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ids = dict(DEFAULT_ID_PATTERNS)
    for kv in a.id_regex:
        k, _, v = kv.partition("=")
        ids[k] = v
    opt = XAppOptions(ckpt=a.ckpt, cells=a.cells, bridge=None if a.bridge == "off" else a.bridge,
                      rrc_log=a.rrc_log, id_patterns=ids, learn_patterns=a.learn_id, audit=a.audit,
                      controller=ControllerConfig(ho_threshold=a.ho_threshold, confirm_count=a.confirm,
                                                  hold_off_s=a.hold_off_s, a3_override_db=a.a3_override_db,
                                                  dry_run=not a.live))
    xapp = HandoverLLMXapp(a.config, a.http_server_port, a.rmr_port, opt)
    xapp.e2sm_rc.set_ran_func_id(a.ran_func_id)
    for s in (signal.SIGQUIT, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, xapp.signal_handler)
    xapp.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
