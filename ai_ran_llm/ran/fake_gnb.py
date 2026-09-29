"""Simulator-backed fake gNB speaking the ran-bridge protocol.

It lets the complete real-RAN path (network, protocol, tracker, controller,
guard rails, actuator) run end to end without radio hardware, and scores it with
exactly the same KPIs as the offline benchmark: the fake gNB is a *policy* for
:func:`ai_ran_llm.simulator.run_policy` whose decisions come from the xApp over
TCP.

Per simulation step (100 ms) it sends one ``meas_report`` per UE (every
``report_every`` steps), waits for the xApp's ``decision`` for each (lockstep,
so results do not depend on timing), applies the ``ho_command``s received, and
answers each command with a ``ho_outcome``. With ``report_all_cells`` and the
controller's guard rails off (``confirm_count=1, hold_off_s=0,
min_confidence=0``) the KPIs equal ``evaluate`` exactly — a tested invariant.

It is also a reference implementation of a RAN-side bridge agent.
"""

from __future__ import annotations

import numpy as np

from ..config import ObsConfig
from ..simulator import Episode, run_policy
from .bridge import BridgeClient
from .cells import CellMap
from .messages import CellMeas, HandoverCommand, HandoverOutcome, MeasReport
from .rrc import rsrp_dbm, rsrp_index


class FakeGnb:
    def __init__(self, ep: Episode, host: str, port: int, cell_map: CellMap | None = None,
                 report_every: int = 1, report_all_cells: bool = True, max_neighbours: int = 8,
                 quantize_rrc: bool = False, timeout_s: float = 30.0, node: str = "fake-gnb"):
        self.ep = ep
        self.cells = cell_map or CellMap.for_simulator(ep.n_cells)
        self.report_every = report_every
        self.report_all_cells = report_all_cells
        self.max_neighbours = max_neighbours
        self.quantize_rrc = quantize_rrc
        self.client = BridgeClient(host, port, node=node, timeout=timeout_s)
        self._pending: dict[int, HandoverCommand] = {}     # UE -> command decided this step
        self.decisions = 0
        self.commands = 0

    def ue_id(self, u: int) -> str:
        return f"ue-{u}"

    def _rsrp(self, v: float) -> float:
        return rsrp_dbm(rsrp_index(v)) if self.quantize_rrc else float(v)

    # ---- policy interface for run_policy ------------------------------------
    def decide(self, ep: Episode, t: int, obs) -> np.ndarray:
        target = np.full(ep.n_ue, -1)
        if self._pending:        # commands run_policy did not execute (no on_handover call this step)
            self.on_handover(np.zeros(ep.n_ue, dtype=bool))
        if t % self.report_every:
            return target
        msgs = []
        for u in range(ep.n_ue):
            meas = ep.rsrp_meas[u, t]
            s = int(obs.serving[u])
            order = [c for c in np.argsort(-meas) if c != s]
            if not self.report_all_cells:
                order = order[: self.max_neighbours]
            msgs.append(MeasReport(
                ue_id=self.ue_id(u),
                serving=CellMeas(pci=self.cells[s].pci, rsrp_dbm=self._rsrp(meas[s]), sinr_db=float(obs.sinr_db[u])),
                neighbours=[CellMeas(pci=self.cells[int(c)].pci, rsrp_dbm=self._rsrp(meas[c])) for c in order],
                timestamp_s=round(t * ep.sim.dt_s, 6), speed_kmh=float(ep.speed_kmh[u]), seq=t,
                ue_ids={"rnti": 0x4601 + u, "cu_ue_id": u + 1, "amf_ue_ngap_id": u + 1,
                        "gnb_cu_ue_f1ap_id": u + 1}))
        self.client.send_many(msgs)
        waiting = {self.ue_id(u) for u in range(ep.n_ue)}
        while waiting:
            msg = self.client.recv()
            if msg is None:
                raise ConnectionError("xApp closed the connection")
            if isinstance(msg, HandoverCommand):
                u = int(msg.ue_id.split("-")[1])
                idx = self.cells.by_pci(msg.target_cell.pci).index
                target[u] = idx
                self._pending[u] = msg
                self.commands += 1
            elif isinstance(msg, dict) and msg.get("type") == "decision" and msg.get("seq") == t:
                waiting.discard(msg["ue_id"])
                self.decisions += 1
            elif isinstance(msg, dict) and msg.get("type") == "error":
                raise RuntimeError(f"xApp protocol error: {msg.get('detail')}")
        return target

    def on_handover(self, mask: np.ndarray) -> None:
        """Called by run_policy with the UEs whose handover was executed this step."""
        out = []
        for u, cmd in self._pending.items():
            status = "success" if mask[u] else "rejected"
            out.append(HandoverOutcome(cmd.command_id, cmd.ue_id, status,
                                       "" if mask[u] else "UE in outage / re-establishment"))
        if out:
            self.client.send_many(out)
        self._pending.clear()

    def run(self, obs_cfg: ObsConfig | None = None):
        """Play the whole episode; returns (Metrics, serving trajectory)."""
        try:
            return run_policy(self.ep, self, obs_cfg or ObsConfig())
        finally:
            self.client.close()
