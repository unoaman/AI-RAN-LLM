"""Canonical RAN messages and the ran-bridge wire protocol.

Everything that crosses the boundary between a RAN (OCUDU / srsRAN, OAI, a
simulator, ...) and the HandoverLLM xApp is one of these messages. Adapters
convert RAN-specific formats (RRC ASN.1, E2SM payloads, log lines, telnet
commands) to and from them, so the decision logic never sees a vendor format.

Wire format ("ran-bridge", protocol ``ai-ran-llm/ran-bridge/1``): one JSON object
per line (NDJSON, UTF-8) over TCP. Every object has a ``"type"``:

=================  =============  ===========================================
type               direction      meaning
=================  =============  ===========================================
``hello``          RAN -> xApp    optional handshake: node name, protocol
``meas_report``    RAN -> xApp    one UE measurement report (serving + neighbours)
``ho_outcome``     RAN -> xApp    result of a handover command
``ue_release``     RAN -> xApp    UE left (RRC release): drop its state
``ho_command``     xApp -> RAN    execute a handover
``decision``       xApp -> RAN    optional: decision for every evaluated report
``error``          both           protocol error, human-readable ``detail``
=================  =============  ===========================================

Field details are in the dataclasses below and in docs/DESIGN.md (§16.6).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

PROTOCOL = "ai-ran-llm/ran-bridge/1"


class ProtocolError(ValueError):
    """A message does not follow the ran-bridge protocol."""


def _opt_float(v):
    return None if v is None else float(v)


def _opt_int(v):
    if v is None:
        return None
    if isinstance(v, str):
        return int(v, 0)          # accepts "0x66C000"
    return int(v)


@dataclass
class CellMeas:
    """One cell as seen in a measurement report. Identify a cell by PCI and/or
    NR Cell Identity (NCI, 36 bit); the cell map resolves either."""

    pci: int | None = None
    nci: int | None = None
    rsrp_dbm: float | None = None
    rsrq_db: float | None = None
    sinr_db: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "CellMeas":
        if not isinstance(d, dict):
            raise ProtocolError(f"cell must be an object, got {type(d).__name__}")
        c = cls(pci=_opt_int(d.get("pci")), nci=_opt_int(d.get("nci")), rsrp_dbm=_opt_float(d.get("rsrp_dbm")),
                rsrq_db=_opt_float(d.get("rsrq_db")), sinr_db=_opt_float(d.get("sinr_db")))
        if c.pci is None and c.nci is None:
            raise ProtocolError("cell needs 'pci' or 'nci'")
        return c

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MeasReport:
    """A UE measurement report.

    ``ue_id`` is the xApp's stable key for the UE (chosen by the adapter, e.g.
    ``"du1/rnti=0x4601"``). ``ue_ids`` carries the RAN identifiers an actuator
    needs to address the UE, e.g. ``rnti``, ``cu_ue_id`` / ``rrc_ue_id`` (OAI),
    ``amf_ue_ngap_id`` + ``gnb_cu_ue_f1ap_id`` (E2SM-RC on srsRAN/OCUDU),
    ``ran_ue_id`` (E2SM-RC on OAI/FlexRIC).
    """

    ue_id: str
    serving: CellMeas
    neighbours: list[CellMeas] = field(default_factory=list)
    timestamp_s: float = field(default_factory=time.time)
    ue_ids: dict = field(default_factory=dict)
    speed_kmh: float | None = None
    seq: int | None = None            # optional sequence number, echoed in the decision
    source: str = ""

    TYPE = "meas_report"

    @classmethod
    def from_dict(cls, d: dict) -> "MeasReport":
        try:
            return cls(ue_id=str(d["ue_id"]), serving=CellMeas.from_dict(d["serving"]),
                       neighbours=[CellMeas.from_dict(n) for n in d.get("neighbours", [])],
                       timestamp_s=float(d.get("timestamp_s", time.time())), ue_ids=dict(d.get("ue_ids", {})),
                       speed_kmh=_opt_float(d.get("speed_kmh")), seq=_opt_int(d.get("seq")),
                       source=str(d.get("source", "")))
        except KeyError as e:
            raise ProtocolError(f"meas_report missing field {e}") from None
        except (TypeError, ValueError) as e:
            raise ProtocolError(f"bad meas_report: {e}") from None

    def to_dict(self) -> dict:
        d = {"type": self.TYPE, "ue_id": self.ue_id, "serving": self.serving.to_dict(),
             "neighbours": [n.to_dict() for n in self.neighbours], "timestamp_s": self.timestamp_s,
             "ue_ids": self.ue_ids}
        for k in ("speed_kmh", "seq"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.source:
            d["source"] = self.source
        return d


@dataclass
class CellRef:
    """A fully resolved cell: model index plus every RAN identity known for it."""

    index: int
    pci: int | None = None
    nci: int | None = None
    plmn: str | None = None
    gnb: str | None = None            # gNB / CU identity (decides F1 vs N2/Xn handover)
    e2_node_id: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "CellRef":
        return cls(index=int(d["index"]), pci=_opt_int(d.get("pci")), nci=_opt_int(d.get("nci")),
                   plmn=d.get("plmn"), gnb=d.get("gnb"), e2_node_id=d.get("e2_node_id"))


@dataclass
class HandoverCommand:
    ue_id: str
    ue_ids: dict
    source_cell: CellRef
    target_cell: CellRef
    confidence: float
    rationale: str = ""
    decided_by: str = "llm"           # "llm" or "a3_fallback"
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    issued_at: float = field(default_factory=time.time)
    dry_run: bool = False

    TYPE = "ho_command"

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "command_id": self.command_id, "ue_id": self.ue_id, "ue_ids": self.ue_ids,
                "source_cell": self.source_cell.to_dict(), "target_cell": self.target_cell.to_dict(),
                "confidence": self.confidence, "rationale": self.rationale, "decided_by": self.decided_by,
                "issued_at": self.issued_at, "dry_run": self.dry_run}

    @classmethod
    def from_dict(cls, d: dict) -> "HandoverCommand":
        try:
            return cls(ue_id=str(d["ue_id"]), ue_ids=dict(d.get("ue_ids", {})),
                       source_cell=CellRef.from_dict(d["source_cell"]), target_cell=CellRef.from_dict(d["target_cell"]),
                       confidence=float(d.get("confidence", 0.0)), rationale=str(d.get("rationale", "")),
                       decided_by=str(d.get("decided_by", "llm")), command_id=str(d["command_id"]),
                       issued_at=float(d.get("issued_at", time.time())), dry_run=bool(d.get("dry_run", False)))
        except KeyError as e:
            raise ProtocolError(f"ho_command missing field {e}") from None


OUTCOMES = ("success", "failure", "rejected", "timeout")


@dataclass
class HandoverOutcome:
    command_id: str
    ue_id: str
    status: str                       # one of OUTCOMES
    detail: str = ""
    timestamp_s: float = field(default_factory=time.time)

    TYPE = "ho_outcome"

    def __post_init__(self):
        if self.status not in OUTCOMES:
            raise ProtocolError(f"ho_outcome status must be one of {OUTCOMES}, got {self.status!r}")

    def to_dict(self) -> dict:
        return {"type": self.TYPE, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "HandoverOutcome":
        try:
            return cls(command_id=str(d["command_id"]), ue_id=str(d["ue_id"]), status=str(d["status"]),
                       detail=str(d.get("detail", "")), timestamp_s=float(d.get("timestamp_s", time.time())))
        except KeyError as e:
            raise ProtocolError(f"ho_outcome missing field {e}") from None


@dataclass
class UERelease:
    ue_id: str
    TYPE = "ue_release"

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "ue_id": self.ue_id}


@dataclass
class Hello:
    node: str = ""
    protocol: str = PROTOCOL
    TYPE = "hello"

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "node": self.node, "protocol": self.protocol}


@dataclass
class Decision:
    """Outcome of evaluating one report (sent back on the bridge for observability)."""

    ue_id: str
    action: str                       # "HANDOVER" | "STAY" | "SKIP"
    reason: str                       # why: "llm", "a3_fallback", or the guard that blocked
    seq: int | None = None
    target_cell: CellRef | None = None
    confidence: float | None = None
    p_stay: float | None = None
    p_handover: dict | None = None    # model index -> probability
    command_id: str | None = None
    rationale: str = ""
    TYPE = "decision"

    def to_dict(self) -> dict:
        d = {"type": self.TYPE, "ue_id": self.ue_id, "action": self.action, "reason": self.reason}
        for k in ("seq", "confidence", "p_stay", "command_id"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.p_handover is not None:
            d["p_handover"] = {str(k): v for k, v in self.p_handover.items()}
        if self.target_cell is not None:
            d["target_cell"] = self.target_cell.to_dict()
        if self.rationale:
            d["rationale"] = self.rationale
        return d


# ---------------------------------------------------------------------------
# NDJSON framing
# ---------------------------------------------------------------------------

_DECODERS = {
    MeasReport.TYPE: MeasReport.from_dict,
    HandoverOutcome.TYPE: HandoverOutcome.from_dict,
    HandoverCommand.TYPE: HandoverCommand.from_dict,
    UERelease.TYPE: lambda d: UERelease(ue_id=str(d["ue_id"])),
    Hello.TYPE: lambda d: Hello(node=str(d.get("node", "")), protocol=str(d.get("protocol", PROTOCOL))),
}


def encode(msg) -> bytes:
    """Message dataclass (or plain dict with a "type") -> one NDJSON line."""
    d = msg if isinstance(msg, dict) else msg.to_dict()
    return (json.dumps(d, separators=(",", ":")) + "\n").encode()


def decode(line: bytes | str):
    """One NDJSON line -> message dataclass (``decision``/``error`` stay dicts)."""
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON: {e}") from None
    if not isinstance(d, dict) or "type" not in d:
        raise ProtocolError("message must be a JSON object with a 'type'")
    t = d["type"]
    if t in ("decision", "error"):
        return d
    if t not in _DECODERS:
        raise ProtocolError(f"unknown message type {t!r}")
    return _DECODERS[t](d)
