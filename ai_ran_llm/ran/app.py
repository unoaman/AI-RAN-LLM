"""Assemble a runnable xApp from options (shared by the CLI and RIC wrappers)."""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field

from ..inference import HandoverLLM
from .actuators import CommandActuator, ConsoleActuator, LogActuator, OAITelnetActuator
from .bridge import BridgeServer
from .cells import CellMap
from .controller import ControllerConfig, HandoverController
from .runtime import XAppRuntime
from .sources import DEFAULT_ID_PATTERNS, RrcLogSource

log = logging.getLogger("ai_ran_llm.ran")


def load_cells(spec: str, auto_add: bool = False) -> CellMap:
    """Path to a cell-map JSON, or ``sim`` for the simulator/fake-gNB identity map."""
    if spec == "sim":
        return CellMap.for_simulator(19)
    return CellMap.load(spec, auto_add=auto_add)


@dataclass
class XAppOptions:
    ckpt: str = "checkpoints/handover_llm.pt"
    cells: str = "sim"
    auto_add_cells: bool = False
    bridge: str | None = "127.0.0.1:7000"       # host:port to listen on, None = off
    rrc_log: str | None = None
    rrc_log_from_start: bool = False
    rrc_log_follow: bool = True
    id_patterns: dict = field(default_factory=lambda: dict(DEFAULT_ID_PATTERNS))
    pci_pattern: str | None = r"\bpci[= ]?(\d+)"
    serv_cell_pci: dict = field(default_factory=dict)
    rrc_header_filter: str | None = None
    learn_patterns: list = field(default_factory=list)
    actuator: str = "log"                        # log | bridge | oai-telnet | console | command
    oai_telnet: str = "127.0.0.1:9090"
    oai_mode: str = "auto"
    console_fifo: str | None = None
    console_template: str = "ho {serving_pci} {rnti_hex} {target_pci}"
    command: str | None = None
    audit: str | None = "ran_audit.jsonl"
    controller: ControllerConfig = field(default_factory=ControllerConfig)


def _hostport(s: str, default_host: str = "127.0.0.1") -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return (host or default_host), int(port)


def build_xapp(opt: XAppOptions, actuator=None):
    """Returns (runtime, sources). Pass `actuator` to override ``opt.actuator``
    (e.g. an ``OranScRicActuator`` inside the O-RAN SC RIC)."""
    llm = HandoverLLM.load(opt.ckpt)
    cells = load_cells(opt.cells, opt.auto_add_cells)
    controller = HandoverController(llm, cells, opt.controller)
    events: queue.Queue = queue.Queue()
    sources = []
    bridge = None
    if opt.bridge:
        bridge = BridgeServer(*_hostport(opt.bridge), events=events).start()
        sources.append(bridge)
        log.info("ran-bridge listening on %s:%s", bridge.host, bridge.port)
    if opt.rrc_log:
        sources.append(RrcLogSource(opt.rrc_log, follow=opt.rrc_log_follow, from_start=opt.rrc_log_from_start,
                                    id_patterns=opt.id_patterns, pci_pattern=opt.pci_pattern,
                                    serv_cell_pci=opt.serv_cell_pci,
                                    header_filter=opt.rrc_header_filter,
                                    learn_patterns=opt.learn_patterns).start(events))
        log.info("tailing RRC log %s", opt.rrc_log)
    if not sources:
        raise ValueError("no measurement source: enable the bridge and/or an RRC log")

    if actuator is None:
        if opt.actuator == "log":
            actuator = LogActuator()
        elif opt.actuator == "bridge":
            if bridge is None:
                raise ValueError("actuator 'bridge' needs the bridge enabled")
            actuator = bridge
        elif opt.actuator == "oai-telnet":
            actuator = OAITelnetActuator(*_hostport(opt.oai_telnet), mode=opt.oai_mode)
        elif opt.actuator == "console":
            if not opt.console_fifo:
                raise ValueError("actuator 'console' needs a FIFO/file path")
            actuator = ConsoleActuator(opt.console_fifo, opt.console_template)
        elif opt.actuator == "command":
            if not opt.command:
                raise ValueError("actuator 'command' needs a command template")
            actuator = CommandActuator(opt.command)
        else:
            raise ValueError(f"unknown actuator {opt.actuator!r}")
    runtime = XAppRuntime(controller, actuator, events, audit_path=opt.audit)
    mode = "SHADOW (dry-run, nothing is actuated)" if opt.controller.dry_run else "LIVE"
    log.info("HandoverLLM xApp: %s mode, actuator=%s, %d cells", mode, getattr(actuator, "name", actuator),
             len(cells.cells))
    return runtime, sources
