"""Actuators: turn a HandoverCommand into a RAN action.

====================  ==========================  ============================================
actuator              RAN                         mechanism
====================  ==========================  ============================================
``BridgeServer``      any agent on ran-bridge     ``ho_command`` NDJSON message (bridge.py)
``OranScRicActuator`` OCUDU / srsRAN (+ others)   E2SM-RC Control Style 3 "Connected Mode
                                                  Mobility", Action 1 "Handover Control",
                                                  via the O-RAN SC near-RT RIC xApp framework
                                                  used by srsRAN's ``oran-sc-ric``
``OAITelnetActuator`` OpenAirInterface gNB/CU     telnet ``ci trigger_f1_ho <cu-ue-id>`` (F1,
                                                  intra-gNB) / ``ci trigger_n2_ho <pci>,<rrc-
                                                  ue-id>`` (N2, inter-gNB)
``ConsoleActuator``   srsRAN / OCUDU gNB console  writes ``ho <serving_pci> <rnti> <target_pci>``
                                                  to a FIFO feeding the gnb's stdin (lab use)
``CommandActuator``   anything with a CLI         runs a command template, e.g. FlexRIC's
                                                  ``xapp_rc_handover`` or your own script
``LogActuator``       none                        shadow mode: only logs
====================  ==========================  ============================================

All actuators raise on failure; the runtime turns the exception into a
``failure`` outcome so the controller backs off.
"""

from __future__ import annotations

import logging
import shlex
import socket
import subprocess

from .messages import HandoverCommand

log = logging.getLogger("ai_ran_llm.ran")


class ActuationError(RuntimeError):
    pass


def command_fields(cmd: HandoverCommand) -> dict:
    """Template fields available to text-based actuators."""
    s, t = cmd.source_cell, cmd.target_cell
    fields = {k: v for k, v in cmd.ue_ids.items()}
    rnti = cmd.ue_ids.get("rnti")
    if rnti is not None:
        r = int(rnti, 0) if isinstance(rnti, str) else int(rnti)
        fields.update(rnti=rnti, rnti_dec=r, rnti_hex=hex(r))
    fields.update(ue_id=cmd.ue_id, command_id=cmd.command_id,
                  serving_pci=s.pci, serving_nci=s.nci, serving_index=s.index,
                  target_pci=t.pci, target_nci=t.nci, target_nci_hex=None if t.nci is None else hex(t.nci),
                  target_index=t.index, plmn=t.plmn or s.plmn, e2_node_id=s.e2_node_id or t.e2_node_id,
                  source_gnb=s.gnb, target_gnb=t.gnb)
    return fields


class LogActuator:
    name = "log"

    def send_handover(self, cmd: HandoverCommand) -> None:
        log.info("HO (not actuated) ue=%s %s -> %s conf=%.2f", cmd.ue_id, cmd.source_cell.pci,
                 cmd.target_cell.pci, cmd.confidence)


class OAITelnetActuator:
    """OpenAirInterface telnet server (build with ``--build-lib telnetsrv``; run the
    gNB/CU with ``--telnetsrv --telnetsrv.shrmod ci``; default port 9090).

    * F1 handover (both cells on the same CU): ``ci trigger_f1_ho <cu-ue-id>``.
      OAI picks the target DU itself, so this only matches the model's target in
      a two-DU setup; use E2 (``CommandActuator`` with FlexRIC) for more cells.
    * N2 handover (different gNBs): ``ci trigger_n2_ho <target-pci>,<rrc-ue-id>``.

    ``mode="auto"`` uses N2 when the cell map puts source and target on different
    ``gnb``s, F1 otherwise. UE ids come from ``cmd.ue_ids``: ``cu_ue_id`` (F1) and
    ``rrc_ue_id`` (N2; falls back to ``cu_ue_id``).
    """

    name = "oai-telnet"

    def __init__(self, host: str = "127.0.0.1", port: int = 9090, mode: str = "auto", timeout: float = 2.0):
        if mode not in ("f1", "n2", "auto"):
            raise ValueError("mode must be f1, n2 or auto")
        self.host, self.port, self.mode, self.timeout = host, port, mode, timeout

    def command_line(self, cmd: HandoverCommand) -> str:
        mode = self.mode
        if mode == "auto":
            s, t = cmd.source_cell.gnb, cmd.target_cell.gnb
            mode = "n2" if (s is not None and t is not None and s != t) else "f1"
        if mode == "f1":
            ue = cmd.ue_ids.get("cu_ue_id")
            if ue is None:
                raise ActuationError("OAI F1 handover needs ue_ids['cu_ue_id']")
            return f"ci trigger_f1_ho {int(ue)}"
        ue = cmd.ue_ids.get("rrc_ue_id", cmd.ue_ids.get("cu_ue_id"))
        if ue is None or cmd.target_cell.pci is None:
            raise ActuationError("OAI N2 handover needs ue_ids['rrc_ue_id'] and the target PCI")
        return f"ci trigger_n2_ho {int(cmd.target_cell.pci)},{int(ue)}"

    def send_handover(self, cmd: HandoverCommand) -> None:
        line = self.command_line(cmd)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.settimeout(0.5)
            try:
                s.recv(4096)                       # banner / prompt, if any
            except socket.timeout:
                pass
            s.sendall((line + "\n").encode())
            reply = b""
            try:
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    reply += chunk
            except socket.timeout:
                pass
        text = reply.decode(errors="replace")
        log.info("OAI telnet %r -> %r", line, text.strip()[:200])
        if any(w in text.lower() for w in ("error", "unknown command", "not found", "failed")):
            raise ActuationError(f"OAI telnet rejected {line!r}: {text.strip()[:200]}")


class ConsoleActuator:
    """srsRAN Project / OCUDU gNB console command, written to a FIFO or file.

    The gnb application reads commands from stdin; for automation start it as
    ``mkfifo /tmp/gnb_cmd; tail -f /tmp/gnb_cmd | sudo gnb -c gnb.yml`` and point
    this actuator at ``/tmp/gnb_cmd``. Default template (srsRAN handover
    tutorial): ``ho {serving_pci} {rnti} {target_pci}``. Check your version's
    console help for the RNTI format (``{rnti_hex}`` / ``{rnti_dec}`` available).
    Intended for lab testing; use E2 in production.
    """

    name = "console"

    def __init__(self, path: str, template: str = "ho {serving_pci} {rnti_hex} {target_pci}"):
        self.path, self.template = path, template

    def send_handover(self, cmd: HandoverCommand) -> None:
        try:
            line = self.template.format(**command_fields(cmd))
        except KeyError as e:
            raise ActuationError(f"console template needs {e} (missing in ue_ids?)") from None
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
        log.info("console %s <- %r", self.path, line)


class CommandActuator:
    """Run an external command per handover. The template is split like a shell
    command line *first* and each argument is then formatted, so values can never
    inject extra arguments; no shell is involved. Example (FlexRIC / OAI)::

        --command "./xapp_rc_handover --ran-ue-id {ran_ue_id} --target-nci {target_nci}"

    (adapt the arguments to your xApp; fields are listed in ``command_fields``).
    """

    name = "command"

    def __init__(self, template: str, timeout: float = 5.0):
        self.argv_template = shlex.split(template)
        self.timeout = timeout

    def argv(self, cmd: HandoverCommand) -> list[str]:
        fields = command_fields(cmd)
        try:
            return [a.format(**fields) for a in self.argv_template]
        except KeyError as e:
            raise ActuationError(f"command template needs {e}") from None

    def send_handover(self, cmd: HandoverCommand) -> None:
        argv = self.argv(cmd)
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ActuationError(f"{argv[0]}: {e}") from None
        if r.returncode != 0:
            raise ActuationError(f"{argv[0]} exited {r.returncode}: {r.stderr.strip()[:200]}")


class OranScRicActuator:
    """E2SM-RC handover through the O-RAN SC near-RT RIC Python xApp framework
    (``xAppBase`` from srsRAN's ``oran-sc-ric`` repository).

    Calls ``xapp.e2sm_rc.control_handover(e2_node_id, amf_ue_ngap_id,
    gnb_cu_ue_f1ap_id, plmn, target_nr_cell_id)`` = RIC Control Request, E2SM-RC
    Control Style 3, Action 1, with the target NR-CGI (PLMN + NCI).

    Needs, per command: the serving cell's ``e2_node_id`` and ``plmn`` and the
    target ``nci`` (cell map), and ``ue_ids['amf_ue_ngap_id']`` +
    ``ue_ids['gnb_cu_ue_f1ap_id']`` (from the measurement source).
    """

    name = "e2-rc"

    def __init__(self, xapp):
        self.xapp = xapp

    def send_handover(self, cmd: HandoverCommand) -> None:
        f = command_fields(cmd)
        missing = [k for k in ("e2_node_id", "plmn", "target_nci", "amf_ue_ngap_id", "gnb_cu_ue_f1ap_id")
                   if f.get(k) is None]
        if missing:
            raise ActuationError(f"E2SM-RC handover needs {missing}")
        self.xapp.e2sm_rc.control_handover(f["e2_node_id"], int(f["amf_ue_ngap_id"]), int(f["gnb_cu_ue_f1ap_id"]),
                                           str(f["plmn"]), int(f["target_nci"]))
