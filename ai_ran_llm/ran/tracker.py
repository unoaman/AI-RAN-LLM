"""Per-UE measurement history, resampled to what the model expects.

The model was trained on reports holding, per cell, ``hist_len`` (5) L3-filtered
RSRP samples ``hist_stride * dt`` (200 ms) apart. A real RAN delivers reports
at its own period (e.g. 120/240/480/1024 ms periodic, or on A3 events) and each
report lists only the serving cell plus the strongest neighbours (up to
``maxReportCells``). The tracker keeps a short time series per UE and cell and
builds the model's report by **sample-and-hold** resampling:

* value of cell c at time τ = latest sample of c with timestamp ≤ τ;
* before the first sample of c, its earliest sample is used (left padding, as
  in training where history indices are clipped at the episode start);
* cells not reported within ``max_age_s`` are dropped from the neighbour list.

Timestamps are taken from the reports, not the wall clock, so replaying logs
works the same as live operation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from ..config import ObsConfig
from .cells import CellMap
from .messages import MeasReport

_EPS = 1e-6


@dataclass
class _UEState:
    samples: dict[int, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=64)))
    serving: int | None = None
    sinr_db: float | None = None
    speed_kmh: float | None = None
    ue_ids: dict = field(default_factory=dict)
    last_t: float = -np.inf
    n_reports: int = 0
    unknown_cells: int = 0


class UEMeasurementTracker:
    def __init__(self, cell_map: CellMap, obs_cfg: ObsConfig | None = None, dt_s: float = 0.1,
                 max_age_s: float = 2.0, min_reports: int = 1):
        self.cells = cell_map
        self.obs_cfg = obs_cfg or ObsConfig()
        self.period_s = self.obs_cfg.hist_stride * dt_s
        self.max_age_s = max_age_s
        self.min_reports = min_reports
        self.ues: dict[str, _UEState] = {}

    def update(self, rep: MeasReport) -> bool:
        """Add a report. Returns False (and ignores it) if the serving cell is unknown
        or has no RSRP; unknown neighbours are skipped and counted."""
        serving = self.cells.resolve(rep.serving)
        if serving is None or rep.serving.rsrp_dbm is None:
            return False
        st = self.ues.setdefault(rep.ue_id, _UEState())
        t = rep.timestamp_s
        st.serving = serving.index
        st.samples[serving.index].append((t, rep.serving.rsrp_dbm))
        for n in rep.neighbours:
            ref = self.cells.resolve(n)
            if ref is None or n.rsrp_dbm is None:
                st.unknown_cells += 1
                continue
            if ref.index != serving.index:
                st.samples[ref.index].append((t, n.rsrp_dbm))
        if rep.serving.sinr_db is not None:
            st.sinr_db = rep.serving.sinr_db
        if rep.speed_kmh is not None:
            st.speed_kmh = rep.speed_kmh
        st.ue_ids.update(rep.ue_ids)
        st.last_t = max(st.last_t, t)
        st.n_reports += 1
        return True

    def ready(self, ue_id: str) -> bool:
        st = self.ues.get(ue_id)
        return (st is not None and st.n_reports >= self.min_reports and st.serving is not None
                and any(c != st.serving and self._fresh(st, c) for c in st.samples))

    def _fresh(self, st: _UEState, cell: int) -> bool:
        q = st.samples.get(cell)
        return bool(q) and st.last_t - q[-1][0] <= self.max_age_s + _EPS

    @staticmethod
    def _value_at(q: deque, tau: float) -> float:
        v = q[0][1]
        for t, x in q:
            if t <= tau + _EPS:
                v = x
            else:
                break
        return v

    def history(self, ue_id: str, cell: int) -> list[float]:
        st = self.ues[ue_id]
        q = st.samples[cell]
        h = self.obs_cfg.hist_len
        return [self._value_at(q, st.last_t - self.period_s * k) for k in range(h - 1, -1, -1)]

    def build_report(self, ue_id: str) -> dict:
        """The xApp request format consumed by ``HandoverLLM.decide_reports``."""
        st = self.ues[ue_id]
        nbrs = [c for c in st.samples if c != st.serving and self._fresh(st, c)]
        report = {
            "ue_id": ue_id,
            "serving_cell": st.serving,
            "serving_rsrp": self.history(ue_id, st.serving),
            "neighbors": [{"cell_id": c, "rsrp": self.history(ue_id, c)} for c in nbrs],
        }
        if st.sinr_db is not None:
            report["sinr_db"] = st.sinr_db
        if st.speed_kmh is not None:
            report["speed_kmh"] = st.speed_kmh
        return report

    def serving(self, ue_id: str) -> int | None:
        st = self.ues.get(ue_id)
        return None if st is None else st.serving

    def ue_ids(self, ue_id: str) -> dict:
        return dict(self.ues[ue_id].ue_ids)

    def forget(self, ue_id: str) -> None:
        self.ues.pop(ue_id, None)

    def gc(self, now_s: float, idle_s: float = 30.0) -> list[str]:
        """Drop UEs without reports for `idle_s`; returns their ids."""
        gone = [u for u, st in self.ues.items() if now_s - st.last_t > idle_s]
        for u in gone:
            del self.ues[u]
        return gone
