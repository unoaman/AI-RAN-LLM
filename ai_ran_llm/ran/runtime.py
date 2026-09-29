"""xApp runtime: sources -> controller -> actuator, with an audit log.

::

    sources (bridge, rrc-log, ...) ──events──▶ queue ──batch──▶ HandoverController
                                                                     │ decisions, commands
                     ┌───────────────────────────────────────────────┤
                     ▼                                               ▼
           audit log (JSONL)                          actuator.send_handover(cmd)
           decision back to bridge agents             (unless shadow mode)

Events are processed in arrival order; consecutive measurement reports are
batched (up to ``max_batch``) so the model scores many UEs in one pass.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

from .actuators import LogActuator
from .controller import HandoverController
from .messages import HandoverOutcome, MeasReport, UERelease

log = logging.getLogger("ai_ran_llm.ran")


class XAppRuntime:
    def __init__(self, controller: HandoverController, actuator=None, events: queue.Queue | None = None,
                 audit_path: str | None = None, max_batch: int = 256, linger_s: float = 0.002):
        self.controller = controller
        self.actuator = actuator or LogActuator()
        self.events = events if events is not None else queue.Queue()
        self.audit = open(audit_path, "a") if audit_path else None
        self.max_batch, self.linger_s = max_batch, linger_s
        self._stop = threading.Event()
        self.errors = 0

    # ------------------------------------------------------------------ loop
    def run(self, duration_s: float | None = None) -> None:
        end = None if duration_s is None else time.monotonic() + duration_s
        while not self._stop.is_set() and (end is None or time.monotonic() < end):
            try:
                first = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.linger_s
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.events.get(timeout=max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            self.process(batch)

    def process(self, batch) -> None:
        """Handle a list of (source, message) events in order."""
        reports = []
        for src, msg in batch:
            if isinstance(msg, MeasReport):
                reports.append((src, msg))
                continue
            self._flush(reports)
            reports = []
            if isinstance(msg, HandoverOutcome):
                self.controller.on_outcome(msg)
                self._audit({"event": "ho_outcome", **msg.to_dict()})
            elif isinstance(msg, UERelease):
                self.controller.on_release(msg.ue_id)
        self._flush(reports)

    def _flush(self, reports) -> None:
        if not reports:
            return
        try:
            results = self.controller.on_reports([m for _, m in reports])
        except Exception:                                  # never let one bad batch kill the xApp
            log.exception("controller failed on a batch of %d reports", len(reports))
            self.errors += 1
            return
        for (src, rep), (decision, cmd) in zip(reports, results):
            if cmd is not None and not cmd.dry_run:
                try:
                    self.actuator.send_handover(cmd)
                except Exception as e:
                    log.warning("handover for %s failed: %s", cmd.ue_id, e)
                    self.controller.on_outcome(HandoverOutcome(cmd.command_id, cmd.ue_id, "failure", str(e),
                                                               timestamp_s=rep.timestamp_s))
                    decision.reason += f" (actuation failed: {e})"
            if hasattr(src, "send"):                        # bridge agents get the decision back
                src.send(rep.ue_id, decision)
            self._audit({"event": "decision", "t": rep.timestamp_s, **decision.to_dict(),
                         **({"command": cmd.to_dict()} if cmd is not None else {})})

    def _audit(self, rec: dict) -> None:
        if self.audit:
            self.audit.write(json.dumps(rec) + "\n")

    def stop(self) -> None:
        self._stop.set()
        if self.audit:
            self.audit.flush()

    def close(self) -> None:
        self.stop()
        if self.audit:
            self.audit.close()
            self.audit = None
