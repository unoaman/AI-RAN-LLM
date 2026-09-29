"""Handover controller: model decisions + guard rails -> handover commands.

For every batch of measurement reports the controller

1. updates the per-UE tracker and builds model reports for the UEs that are ready;
2. scores them in **one** forward pass (``HandoverLLM.decide_reports``) with the
   same threshold rule and A3 fallback as the benchmark;
3. optionally applies the **A3 override** (``a3_override_db``): if the model says
   STAY although a neighbour has beaten the serving cell by more than
   ``a3_override_db`` in each of the last ``a3_ttt`` samples, the recommendation
   becomes a handover to the strongest such neighbour. This is a safety net for
   inputs outside the training distribution, where the model can be confidently
   wrong (the low-confidence fallback cannot catch that);
4. applies guard rails, in this order, to every HANDOVER recommendation:

   ===================  =====================================================
   guard                blocks a handover when …
   ===================  =====================================================
   ``pending``          a command for this UE is still in flight
   ``failure_backoff``  the last command for this UE failed less than N s ago
   ``hold_off``         the UE handed over less than ``hold_off_s`` ago
   ``not_neighbour``    the target is not in the serving cell's neighbour list
   ``confirm``          the same target was not recommended ``confirm_count``
                        times in a row (model-side time-to-trigger)
   ``rate_limit``       more than ``max_commands_per_s`` commands network-wide
   ===================  =====================================================

5. emits a :class:`HandoverCommand` for every recommendation that passes.
   In **shadow mode** (``dry_run=True``, the default) commands are marked
   ``dry_run`` and the runtime only logs them.

Every evaluated report yields a :class:`Decision` (HANDOVER / STAY / SKIP with
the reason), which the runtime writes to the audit log.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

from .cells import CellMap
from .messages import Decision, HandoverCommand, HandoverOutcome, MeasReport
from .tracker import UEMeasurementTracker


@dataclass
class ControllerConfig:
    ho_threshold: float = 0.35        # P(handover to best neighbour) needed to hand over
    min_confidence: float = 0.3       # below this, the A3 fallback decides
    a3_hyst_db: float = 3.0           # A3 fallback parameters
    a3_ttt: int = 3
    a3_override_db: float | None = None  # hand over despite the model if a neighbour leads by this much
    confirm_count: int = 1            # consecutive identical recommendations required
    hold_off_s: float = 0.0           # minimum time between handovers of one UE
    pending_timeout_s: float = 2.0    # give up waiting for an outcome after this
    failure_backoff_s: float = 5.0    # no new command for a UE after a failed one
    max_commands_per_s: float = 50.0  # network-wide rate limit
    require_neighbour: bool = True    # enforce the cell map's neighbour lists
    dry_run: bool = True              # shadow mode: decide and log, never actuate
    explain: bool = False             # generate model rationales (slower) for every report


@dataclass
class _UECtl:
    candidate: int | None = None
    streak: int = 0
    pending: HandoverCommand | None = None
    last_ho_s: float = -1e18
    backoff_until_s: float = -1e18


@dataclass
class ControllerStats:
    reports: int = 0
    evaluated: int = 0
    recommended: int = 0
    commands: int = 0
    overrides: int = 0
    blocked: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)


class HandoverController:
    def __init__(self, llm, cell_map: CellMap, cfg: ControllerConfig | None = None,
                 tracker: UEMeasurementTracker | None = None):
        self.llm = llm
        self.cells = cell_map
        self.cfg = cfg or ControllerConfig()
        self.tracker = tracker or UEMeasurementTracker(cell_map, llm.tok.obs_cfg)
        self.state: dict[str, _UECtl] = {}
        self.stats = ControllerStats()
        self._sent = deque()           # command timestamps for the rate limit

    # ------------------------------------------------------------------ input
    def on_reports(self, reports: list[MeasReport]) -> list[tuple[Decision, HandoverCommand | None]]:
        """Process a batch of reports. Returns (decision, command-or-None) per report,
        in input order. Reports of the same UE in one batch are all tracked; the UE is
        evaluated once, at its last report."""
        results: list = [None] * len(reports)
        last_idx: dict[str, int] = {}
        for i, rep in enumerate(reports):
            self.stats.reports += 1
            if not self.tracker.update(rep):
                results[i] = (Decision(rep.ue_id, "SKIP", "unknown_serving_cell", seq=rep.seq), None)
                continue
            self._check_pending(rep)
            last_idx[rep.ue_id] = i
        for ue, i in list(last_idx.items()):
            if not self.tracker.ready(ue):
                results[i] = (Decision(ue, "SKIP", "not_enough_measurements", seq=reports[i].seq), None)
                del last_idx[ue]
        for i, rep in enumerate(reports):
            if results[i] is None and last_idx.get(rep.ue_id) != i:
                results[i] = (Decision(rep.ue_id, "SKIP", "superseded_in_batch", seq=rep.seq), None)
        if not last_idx:
            return results

        ues = list(last_idx)
        model_reports = [self.tracker.build_report(u) for u in ues]
        answers = self.llm.decide_reports(model_reports, self.cfg.ho_threshold, self.cfg.min_confidence,
                                          self.cfg.a3_hyst_db, self.cfg.a3_ttt, explain=self.cfg.explain)
        for ue, ans, mrep in zip(ues, answers, model_reports):
            i = last_idx[ue]
            ans = self._maybe_override(ans, mrep)
            ans["rationale"] = self._with_pci(ans["rationale"])
            results[i] = self._apply_guards(ue, reports[i], ans)
        return results

    def _with_pci(self, text: str) -> str:
        """Model rationales name model cell indices; add the RAN's PCI for operators."""
        def sub(m):
            ref = self.cells.cells.get(int(m.group(1)))
            return m.group(0) if ref is None or ref.pci is None else f"{m.group(0)} (PCI {ref.pci})"
        return re.sub(r"\bcell (\d+)", sub, text)

    def _maybe_override(self, ans: dict, mrep: dict) -> dict:
        thr, n = self.cfg.a3_override_db, self.cfg.a3_ttt
        if thr is None or ans["action"] == "HANDOVER":
            return ans
        serving = mrep["serving_rsrp"][-n:]
        best, best_rsrp = None, None
        for nb in mrep["neighbors"]:
            r = nb["rsrp"][-n:]
            if all(x - s > thr for x, s in zip(r, serving)) and (best_rsrp is None or r[-1] > best_rsrp):
                best, best_rsrp = nb["cell_id"], r[-1]
        if best is None:
            return ans
        self.stats.overrides += 1
        return {**ans, "action": "HANDOVER", "target_cell": best, "source": "a3_override",
                "confidence": ans["p_handover"].get(best, 0.0),
                "rationale": f"A3 override: cell {best} exceeds serving by >{thr} dB in the last {n} samples "
                             f"although the model said stay (p_stay={ans['p_stay']:.2f})."}

    def on_outcome(self, out: HandoverOutcome) -> None:
        st = self.state.get(out.ue_id)
        self.stats.outcomes[out.status] = self.stats.outcomes.get(out.status, 0) + 1
        if st is None or st.pending is None or st.pending.command_id != out.command_id:
            return
        # all guard timers run on the report clock (RAN time), never the outcome's clock
        ue = self.tracker.ues.get(out.ue_id)
        now = ue.last_t if ue is not None else out.timestamp_s
        if out.status == "success":
            st.last_ho_s = now
        elif out.status in ("failure", "timeout"):
            st.backoff_until_s = now + self.cfg.failure_backoff_s
        st.pending, st.candidate, st.streak = None, None, 0

    def on_release(self, ue_id: str) -> None:
        self.tracker.forget(ue_id)
        self.state.pop(ue_id, None)

    # --------------------------------------------------------------- internals
    def _check_pending(self, rep: MeasReport) -> None:
        """Infer completion from reports: serving cell == target means success."""
        st = self.state.get(rep.ue_id)
        if st is None or st.pending is None:
            return
        if self.tracker.serving(rep.ue_id) == st.pending.target_cell.index:
            st.last_ho_s = rep.timestamp_s
            st.pending, st.candidate, st.streak = None, None, 0
        elif rep.timestamp_s - st.pending.issued_at > self.cfg.pending_timeout_s:
            st.backoff_until_s = rep.timestamp_s + self.cfg.failure_backoff_s
            st.pending, st.candidate, st.streak = None, None, 0
            self.stats.outcomes["timeout"] = self.stats.outcomes.get("timeout", 0) + 1

    def _block(self, ue: str, rep: MeasReport, reason: str, ans: dict, target) -> tuple:
        self.stats.blocked[reason] = self.stats.blocked.get(reason, 0) + 1
        return (Decision(ue, "STAY", reason, seq=rep.seq, target_cell=target, confidence=ans["confidence"],
                         p_stay=ans["p_stay"], p_handover=ans["p_handover"], rationale=ans["rationale"]), None)

    def _apply_guards(self, ue: str, rep: MeasReport, ans: dict):
        self.stats.evaluated += 1
        cfg, now = self.cfg, rep.timestamp_s
        st = self.state.setdefault(ue, _UECtl())
        if ans["action"] != "HANDOVER":
            st.candidate, st.streak = None, 0
            return (Decision(ue, "STAY", ans["source"], seq=rep.seq, confidence=ans["confidence"],
                             p_stay=ans["p_stay"], p_handover=ans["p_handover"], rationale=ans["rationale"]), None)

        self.stats.recommended += 1
        serving = self.tracker.serving(ue)
        target = self.cells[ans["target_cell"]]
        st.streak = st.streak + 1 if st.candidate == target.index else 1
        st.candidate = target.index
        if st.pending is not None:
            return self._block(ue, rep, "pending", ans, target)
        if now < st.backoff_until_s:
            return self._block(ue, rep, "failure_backoff", ans, target)
        if now - st.last_ho_s < cfg.hold_off_s:
            return self._block(ue, rep, "hold_off", ans, target)
        if cfg.require_neighbour and not self.cells.is_neighbour(serving, target.index):
            return self._block(ue, rep, "not_neighbour", ans, target)
        if st.streak < cfg.confirm_count:
            return self._block(ue, rep, "confirm", ans, target)
        wall = time.monotonic()
        while self._sent and wall - self._sent[0] > 1.0:
            self._sent.popleft()
        if len(self._sent) >= cfg.max_commands_per_s:
            return self._block(ue, rep, "rate_limit", ans, target)

        self._sent.append(wall)
        cmd = HandoverCommand(ue_id=ue, ue_ids=self.tracker.ue_ids(ue), source_cell=self.cells[serving],
                              target_cell=target, confidence=ans["confidence"], rationale=ans["rationale"],
                              decided_by=ans["source"], issued_at=now, dry_run=cfg.dry_run)
        if not cfg.dry_run:
            st.pending = cmd
        st.candidate, st.streak = None, 0
        self.stats.commands += 1
        return (Decision(ue, "HANDOVER", ans["source"], seq=rep.seq, target_cell=target,
                         confidence=ans["confidence"], p_stay=ans["p_stay"], p_handover=ans["p_handover"],
                         command_id=cmd.command_id, rationale=ans["rationale"]), cmd)
