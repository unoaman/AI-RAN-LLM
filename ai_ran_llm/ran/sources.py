"""Measurement sources other than the bridge.

:class:`RrcLogSource` tails a gNB / CU-CP log and turns every RRC
MeasurementReport it finds (JSON or asn1c XER, see ``rrc.py``) into a
:class:`MeasReport`. This is the quickest way to feed a real OCUDU / srsRAN or
OAI gNB without writing any RAN-side code: enable RRC message logging, point
the source at the log file.

Log header lines differ between projects and versions, so the UE identifiers
are extracted with configurable regular expressions applied to the line
preceding each message (defaults match ``ue=<n>`` and ``rnti=0x....``/
``c-rnti=0x....``). Check a few lines of your own log and adjust
``id_patterns`` / ``pci_pattern`` (``ai_ran_llm ran-parse`` shows what is
extracted).

Identifiers needed for E2 control (``amf_ue_ngap_id``, ``gnb_cu_ue_f1ap_id``,
...) usually appear on *other* lines (NGAP / F1AP procedures). ``learn_patterns``
are regexes with named groups, applied to every log line; a group named ``ue``
must match the same UE key as the report header (``ue_index``, or ``rnti`` when
no index is logged), and every other named group is remembered for that UE and
attached to its later reports, e.g.
``ue=(?P<ue>\d+).*?amf_ue_id=(?P<amf_ue_ngap_id>\d+)``.
"""

from __future__ import annotations

import os
import re
import threading
import time

from .messages import MeasReport
from .rrc import iter_asn1_blocks, parse_measurement_report

DEFAULT_ID_PATTERNS = {
    "ue_index": r"\bue[=_ ]?(?:id)?[= ]?(\d+)",
    "rnti": r"\b(?:c-)?rnti[= ]?(0x[0-9a-fA-F]+|\d+)",
}


def follow_lines(path: str, from_start: bool = False, stop: threading.Event | None = None, poll_s: float = 0.1):
    """``tail -F``: yield lines as they are appended; survives truncation/rotation."""
    f, inode = None, None
    while stop is None or not stop.is_set():
        if f is None:
            try:
                f = open(path, errors="replace")
                inode = os.fstat(f.fileno()).st_ino
                if not from_start:
                    f.seek(0, os.SEEK_END)
                from_start = True             # after a rotation, read the new file from the start
            except FileNotFoundError:
                time.sleep(poll_s)
                continue
        line = f.readline()
        if line:
            yield line
            continue
        try:
            st = os.stat(path)
            if st.st_ino != inode or st.st_size < f.tell():
                f.close()
                f = None
                continue
        except FileNotFoundError:
            pass
        time.sleep(poll_s)


class RrcLogSource:
    name = "rrc-log"

    def __init__(self, path: str, follow: bool = True, from_start: bool = False,
                 id_patterns: dict[str, str] | None = None, pci_pattern: str | None = r"\bpci[= ]?(\d+)",
                 serv_cell_pci: dict[int, int] | None = None, header_filter: str | None = None,
                 ue_prefix: str = "", clock=time.time, learn_patterns: list[str] | None = None):
        self.path, self.follow, self.from_start = path, follow, from_start
        self.id_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in (id_patterns or DEFAULT_ID_PATTERNS).items()}
        self.pci_pattern = re.compile(pci_pattern, re.IGNORECASE) if pci_pattern else None
        self.serv_cell_pci = serv_cell_pci or {}
        self.header_filter = header_filter
        self.ue_prefix = ue_prefix
        self.clock = clock
        self.learn_patterns = [re.compile(p, re.IGNORECASE) for p in (learn_patterns or [])]
        self.learned: dict[str, dict] = {}
        self.last_serving_pci: dict[str, int] = {}
        self.skipped = 0
        self._stop = threading.Event()

    def _lines(self):
        if self.follow:
            return follow_lines(self.path, self.from_start, self._stop)
        return open(self.path, errors="replace")

    def reports(self):
        """Generator of MeasReport."""
        for block in iter_asn1_blocks(self._lines(), self.header_filter, on_line=self.learn):
            rep = self.to_report(block.header, block.body)
            if rep is None:
                continue
            yield rep

    @staticmethod
    def _num(v: str):
        return int(v, 0) if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", v) else v

    def learn(self, line: str) -> None:
        """Remember UE identifiers seen on non-report lines (see module doc)."""
        for rx in self.learn_patterns:
            m = rx.search(line)
            if m and m.groupdict().get("ue") is not None:
                ids = {k: self._num(v) for k, v in m.groupdict().items() if k != "ue" and v is not None}
                self.learned.setdefault(f"{self.ue_prefix}ue={self._num(m.group('ue'))}", {}).update(ids)

    def to_report(self, header: str, body: dict) -> MeasReport | None:
        parsed = parse_measurement_report(body)
        if parsed is None:
            return None
        ids = {}
        for k, rx in self.id_patterns.items():
            m = rx.search(header)
            if m:
                ids[k] = self._num(m.group(1))
        key = ids.get("ue_index", ids.get("rnti"))
        if key is None:
            self.skipped += 1
            return None
        ue_id = f"{self.ue_prefix}ue={key}"
        ids = {**self.learned.get(ue_id, {}), **ids}
        if not parsed.serving:
            self.skipped += 1
            return None
        serv_id, serving = parsed.serving[0]
        if serving.pci is None:                      # physCellId is optional for the serving cell
            m = self.pci_pattern.search(header) if self.pci_pattern else None
            if m:
                serving.pci = int(m.group(1))
            elif serv_id in self.serv_cell_pci:
                serving.pci = self.serv_cell_pci[serv_id]
            else:
                serving.pci = self.last_serving_pci.get(ue_id)
        if serving.pci is None:
            self.skipped += 1
            return None
        self.last_serving_pci[ue_id] = serving.pci
        return MeasReport(ue_id=ue_id, serving=serving, neighbours=parsed.neighbours, timestamp_s=self.clock(),
                          ue_ids=ids, source=self.name)

    def run(self, events):
        """Push reports into the runtime's event queue (thread target)."""
        for rep in self.reports():
            events.put((self, rep))
            if self._stop.is_set():
                break

    def start(self, events) -> "RrcLogSource":
        threading.Thread(target=self.run, args=(events,), daemon=True, name="rrc-log").start()
        return self

    def close(self):
        self._stop.set()
