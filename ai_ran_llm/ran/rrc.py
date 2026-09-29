"""Parse 3GPP NR RRC MeasurementReport messages (TS 38.331) into MeasReport.

Real gNBs receive the UE's measurements as an RRC ``MeasurementReport`` on
UL-DCCH. Its content is the ASN.1 structure::

    MeasurementReport
      criticalExtensions.measurementReport.measResults
        measId
        measResultServingMOList: [ {servCellId, measResultServingCell: MeasResultNR, ...} ]
        measResultNeighCells.measResultListNR: [ MeasResultNR, ... ]

    MeasResultNR: { physCellId, measResult.cellResults.resultsSSB-Cell {rsrp, rsrq, sinr} }

Values are *report indices*, not dB (TS 38.133 §10.1): see ``rsrp_dbm`` etc.

The same structure appears, with the same field names, in:

* srsRAN / OCUDU CU-CP logs, which print RRC messages as JSON when the RRC
  log level is high enough;
* ``tshark -T json`` / Wireshark exports of F1AP/NGAP captures;
* E2SM-RC REPORT "message copy" payloads once decoded;
* asn1c XER (XML) dumps, as produced by OAI's asn1c-based RRC (use
  :func:`xer_to_dict` first).

:func:`parse_measurement_report` accepts any of these after conversion to a
Python dict. It searches for ``measResults`` at any depth, so wrapper objects
(``UL-DCCH-Message`` → ``message`` → ``c1`` → ...) do not matter.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .messages import CellMeas

# ---------------------------------------------------------------------------
# TS 38.133 report mappings (L3, SS-based)
# ---------------------------------------------------------------------------


def rsrp_dbm(index: int) -> float:
    """SS-RSRP report index 0..127 -> dBm (lower bin edge).

    Table 10.1.6.1-1: RSRP_n for n = 1..126 means ``n-157 <= RSRP < n-156`` dBm;
    RSRP_0 is ``< -156``. Clamped to [-157, -30].
    """
    return float(min(max(int(index), 0), 127) - 157)


def rsrq_db(index: int) -> float:
    """SS-RSRQ report index 0..127 -> dB (lower bin edge, 0.5 dB steps, -43 .. 20)."""
    return (min(max(int(index), 0), 127) - 87) / 2.0


def sinr_db(index: int) -> float:
    """SS-SINR report index 0..127 -> dB (lower bin edge, 0.5 dB steps, -23 .. 40)."""
    return (min(max(int(index), 0), 127) - 47) / 2.0


def rsrp_index(dbm: float) -> int:
    """Inverse of :func:`rsrp_dbm` (used by test generators and the fake gNB)."""
    return int(min(max(int(dbm // 1) + 157, 0), 127))


def sinr_index(db: float) -> int:
    return int(min(max(int((db * 2) // 1) + 47, 0), 127))


# ---------------------------------------------------------------------------
# ASN.1 (as dict) -> cells
# ---------------------------------------------------------------------------


@dataclass
class ParsedMeasReport:
    meas_id: int | None
    serving: list[tuple[int | None, CellMeas]]   # (servCellId, cell); cell.pci may be None
    neighbours: list[CellMeas]


def _find_key(obj, key):
    """Depth-first search for `key` in nested dicts/lists; returns the value or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _as_list(x) -> list:
    """Normalise ASN.1 SEQUENCE OF representations to a list.

    JSON encoders give a list; XER gives ``{"MeasResultNR": [...]}`` or a single
    ``{"MeasResultNR": {...}}`` wrapper (element named after the item type).
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict) and len(x) == 1:
        (k, v), = x.items()
        if k[:1].isupper():                       # type-name wrapper from XER
            return v if isinstance(v, list) else [v]
    return [x]


def _cell_from_meas_result_nr(m: dict) -> CellMeas:
    pci = m.get("physCellId")
    cell_results = _find_key(m, "cellResults") or {}
    res = cell_results.get("resultsSSB-Cell") or cell_results.get("resultsCSI-RS-Cell") or {}
    get = lambda k: None if res.get(k) is None else int(res[k])
    r, q, s = get("rsrp"), get("rsrq"), get("sinr")
    return CellMeas(pci=None if pci is None else int(pci),
                    rsrp_dbm=None if r is None else rsrp_dbm(r),
                    rsrq_db=None if q is None else rsrq_db(q),
                    sinr_db=None if s is None else sinr_db(s))


def parse_measurement_report(msg: dict) -> ParsedMeasReport | None:
    """ASN.1 MeasurementReport (any wrapper, JSON or XER-derived dict) -> cells.

    Returns None if the object contains no ``measResults``.
    """
    mr = _find_key(msg, "measResults")
    if not isinstance(mr, dict):
        return None
    meas_id = mr.get("measId")
    serving = []
    for item in _as_list(mr.get("measResultServingMOList")):
        cell = item.get("measResultServingCell")
        if isinstance(cell, dict):
            sid = item.get("servCellId")
            serving.append((None if sid is None else int(sid), _cell_from_meas_result_nr(cell)))
    neigh = mr.get("measResultNeighCells") or {}
    nr_list = neigh.get("measResultListNR") if isinstance(neigh, dict) else None
    neighbours = [_cell_from_meas_result_nr(m) for m in _as_list(nr_list) if isinstance(m, dict)]
    neighbours = [c for c in neighbours if c.pci is not None]
    return ParsedMeasReport(None if meas_id is None else int(meas_id), serving, neighbours)


# ---------------------------------------------------------------------------
# XER (asn1c XML) -> dict
# ---------------------------------------------------------------------------


def _xer_node(el: ET.Element):
    children = list(el)
    if not children:
        text = (el.text or "").strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return text if text else {}           # <true/>, <spare/> etc. become {}
    out: dict = {}
    for c in children:
        v = _xer_node(c)
        if c.tag in out:
            if not isinstance(out[c.tag], list):
                out[c.tag] = [out[c.tag]]
            out[c.tag].append(v)
        else:
            out[c.tag] = v
    return out


def xer_to_dict(xml_text: str) -> dict:
    """asn1c XER XML (e.g. ``xer_fprint`` output) -> nested dict with the ASN.1 field names."""
    root = ET.fromstring(xml_text)
    return {root.tag: _xer_node(root)}


# ---------------------------------------------------------------------------
# Log scanning: multi-line JSON / XER blocks with a header line
# ---------------------------------------------------------------------------


@dataclass
class LogBlock:
    header: str          # nearest preceding non-block line (carries UE ids, PCI, time)
    body: dict           # decoded ASN.1 object


def iter_asn1_blocks(lines, header_filter: str | None = None, on_line=None):
    """Yield LogBlock for every JSON object or XER document found in a log.

    A JSON block starts at a line whose first non-blank character is ``{`` and
    ends when braces balance; an XER block starts at ``<MeasurementReport>`` or
    ``<UL-DCCH-Message>`` and ends at its closing tag. The most recent other
    line is attached as ``header``. `header_filter` (regex) skips blocks whose
    header does not match (e.g. only ``measurementReport`` messages).
    `on_line`, if given, is called with every line outside blocks.
    """
    header, buf, depth, xml_tag = "", [], 0, None
    hf = re.compile(header_filter) if header_filter else None
    for raw in lines:
        line = raw.rstrip("\n")
        if xml_tag is None and not buf:
            s = line.strip()
            m = re.match(r"<(MeasurementReport|UL-DCCH-Message)>", s)
            if s.startswith("{"):
                buf, depth = [line], line.count("{") - line.count("}")
                if depth > 0:
                    continue
            elif m:
                xml_tag, buf = m.group(1), [line]
                if f"</{xml_tag}>" not in s:
                    continue
            else:
                header = line
                if on_line is not None:
                    on_line(line)
                continue
        else:
            buf.append(line)
            if xml_tag is None:
                depth += line.count("{") - line.count("}")
                if depth > 0:
                    continue
            elif f"</{xml_tag}>" not in line:
                continue
        text = "\n".join(buf)
        is_xml, buf, depth, xml_tag = xml_tag is not None, [], 0, None
        if hf and not hf.search(header):
            continue
        try:
            body = xer_to_dict(text) if is_xml else json.loads(text)
        except (json.JSONDecodeError, ET.ParseError):
            continue
        yield LogBlock(header, body)
