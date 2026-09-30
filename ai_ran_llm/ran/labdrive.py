"""Lab drive: synthetic radio for a real OAI + FlexRIC testbed (docs/RAN_INTEGRATION.md §9.7).

OAI's software UE (nr-uesoftmodem) does not send RRC MeasurementReports, so a testbed built
from OAI's RF simulator gives the xApp nothing to decide on. This relay sits between the
FlexRIC bridge xApp (``llm_bridge``) and ``ran-xapp`` and supplies the missing radio:

* ``llm_bridge`` reports each real UE's E2 identity and **serving cell** (``ue_context``,
  from E2SM-RC REPORT Style 5), and forwards real ``meas_report``s when a UE sends them.
* The relay moves a virtual UE along the line between the cell sites, computes RSRP / SINR
  per cell with the simulator's channel model, and sends ``meas_report``s for the real UE
  (real E2 identity, real serving cell) to ``ran-xapp``.
* Optionally it sets the RF simulator's per-DU path loss on the nrUE (``channelmod modify
  <n> ploss <dB>`` over the UE's telnet server), so the real radio link weakens as the
  virtual UE moves away: a missed handover shows up as a real link failure.
* ``ho_command``s from ``ran-xapp`` go to ``llm_bridge``, which sends E2SM-RC Handover
  Control. The handover itself (F1 or N2) is executed by OAI and shows up as a new serving
  cell in the next ``ue_context``.

Bridge side (llm_bridge -> relay, NDJSON on ``listen``), in addition to the ran-bridge types:

  {"type": "ue_context", "ue_id": "...", "ue_ids": {...}, "serving": {"nci": 305419896, "pci": 0},
   "timestamp_s": 12.3}

Everything else from the bridge (meas_report, ho_outcome, ue_release) is forwarded to
``ran-xapp``; ho_command / decision / error from ``ran-xapp`` go back to the bridge that owns
the UE.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import SimConfig
from .cells import CellMap
from .messages import CellMeas, MeasReport

log = logging.getLogger("ai_ran_llm.ran.labdrive")


@dataclass
class LabDriveConfig:
    xapp: str = "127.0.0.1:7000"          # ran-xapp bridge
    listen: str = "0.0.0.0:7001"          # where llm_bridge connects
    synthetic: str = "auto"               # auto: only for UEs without real reports | on | off
    report_period_s: float = 0.2
    isd_m: float = 500.0                  # site spacing along the drive line (cells in map order)
    speed_kmh: float = 30.0
    lateral_m: float = 30.0               # offset of the road from the line through the sites
    overshoot_m: float = 150.0            # the drive turns around this far beyond the end sites
    shadow_sigma_db: float = 4.0
    shadow_decorr_m: float = 50.0
    meas_sigma_db: float = 1.0
    ue_telnet: str | None = None          # nrUE telnet (ciUE) HOST:PORT for channelmod, None = off
    channels: dict = field(default_factory=dict)   # cell index -> rfsim channel model index
    ploss_at_ref_db: float = 20.0         # rfsim ploss when RSRP == rsrp_ref_dbm
    rsrp_ref_dbm: float = -80.0
    ploss_min_db: float = 0.0
    ploss_max_db: float = 60.0
    ploss_period_s: float = 1.0
    time_scale: float = 1.0               # >1 runs the virtual drive faster than real time (tests)
    seed: int = 0


class _Line:
    """NDJSON over one TCP socket."""

    def __init__(self, sock: socket.socket, name: str):
        self.sock, self.name = sock, name
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rfile = sock.makefile("rb")
        self.lock = threading.Lock()

    def send(self, obj: dict) -> None:
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            self.sock.sendall(data)

    def recv(self) -> dict | None:
        while True:
            line = self.rfile.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s: bad JSON line dropped: %r", self.name, line[:120])

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class _UE:
    ue_id: str
    ue_ids: dict
    conn: _Line
    serving: int | None = None            # cell index
    x_m: float = 0.0
    direction: float = 1.0
    shadow: np.ndarray | None = None
    last_real_s: float = -1e9
    seq: int = 0
    handovers: int = 0


class LabDrive:
    def __init__(self, cells: CellMap, cfg: LabDriveConfig | None = None):
        self.cells, self.cfg = cells, cfg or LabDriveConfig()
        self.sim = SimConfig()
        self.idx = sorted(cells.cells)
        self.sites = np.array([[k * self.cfg.isd_m, 0.0] for k in range(len(self.idx))])
        self.rng = np.random.default_rng(self.cfg.seed)
        self.ues: dict[str, _UE] = {}
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.t0 = time.monotonic()
        self.stats = {"reports_synthetic": 0, "reports_real": 0, "ho_commands": 0, "handovers_executed": 0,
                      "ho_outcomes": {}, "decisions": {}, "ue_contexts": 0, "ues_released": 0}
        self.events: list[dict] = []          # handover timeline, for the test report
        self.xapp: _Line | None = None
        self._last_ploss: dict = {}

    # ---- clock -----------------------------------------------------------------
    def now_s(self) -> float:
        return (time.monotonic() - self.t0) * self.cfg.time_scale

    # ---- radio model -----------------------------------------------------------
    def _pos(self, ue: _UE) -> np.ndarray:
        return np.array([ue.x_m, self.cfg.lateral_m])

    def radio(self, ue: _UE) -> tuple[np.ndarray, float]:
        """RSRP per cell (map order) and SINR on the serving cell, simulator channel model."""
        d = np.maximum(np.linalg.norm(self.sites - self._pos(ue), axis=1), 10.0)
        pl = 128.1 + 37.6 * np.log10(d / 1000.0)
        rsrp = self.sim.tx_power_dbm - pl + ue.shadow + self.rng.normal(0, self.cfg.meas_sigma_db, len(d))
        s = self.idx.index(ue.serving) if ue.serving in self.idx else int(np.argmax(rsrp))
        lin = 10 ** (rsrp / 10)
        interf = self.sim.load * (lin.sum() - lin[s])
        sinr = 10 * math.log10(lin[s] / (interf + 10 ** (self.sim.noise_dbm / 10)))
        return rsrp, sinr

    def _advance(self, ue: _UE, dt_s: float) -> None:
        step = self.cfg.speed_kmh / 3.6 * dt_s
        lo = -self.cfg.overshoot_m
        hi = self.sites[-1, 0] + self.cfg.overshoot_m
        ue.x_m += ue.direction * step
        if ue.x_m > hi or ue.x_m < lo:
            ue.direction *= -1
            ue.x_m = min(max(ue.x_m, lo), hi)
        rho = math.exp(-step / self.cfg.shadow_decorr_m)
        ue.shadow = rho * ue.shadow + math.sqrt(1 - rho * rho) * self.rng.normal(0, self.cfg.shadow_sigma_db, len(self.idx))

    # ---- bridge side -----------------------------------------------------------
    def _on_bridge_msg(self, conn: _Line, m: dict) -> None:
        t = m.get("type")
        if t == "ue_context":
            self.stats["ue_contexts"] += 1
            ue_id = str(m["ue_id"])
            ref = self.cells.resolve(CellMeas.from_dict({**m["serving"], "rsrp_dbm": 0.0}))
            with self.lock:
                ue = self.ues.get(ue_id)
                if ue is None:
                    start = self.idx.index(ref.index) if ref is not None else 0
                    ue = _UE(ue_id, dict(m.get("ue_ids", {})), conn, x_m=float(self.sites[start, 0]) + 50.0,
                             direction=1.0 if start < len(self.idx) - 1 else -1.0,
                             shadow=self.rng.normal(0, self.cfg.shadow_sigma_db, len(self.idx)))
                    self.ues[ue_id] = ue
                    log.info("UE %s attached on cell %s", ue_id, None if ref is None else ref.index)
                ue.conn, ue.ue_ids = conn, dict(m.get("ue_ids", ue.ue_ids))
                new = None if ref is None else ref.index
                if ue.serving is not None and new is not None and new != ue.serving:
                    ue.handovers += 1
                    self.stats["handovers_executed"] += 1
                    self.events.append({"t": round(self.now_s(), 2), "event": "serving_changed", "ue_id": ue_id,
                                        "from": ue.serving, "to": new, "x_m": round(ue.x_m, 1)})
                    log.info("UE %s now served by cell %s (x=%.0f m)", ue_id, new, ue.x_m)
                ue.serving = new
        elif t == "meas_report":
            self.stats["reports_real"] += 1
            with self.lock:
                if str(m.get("ue_id")) in self.ues:
                    self.ues[str(m["ue_id"])].last_real_s = self.now_s()
            self._to_xapp(m)
        elif t == "ue_release":
            self.stats["ues_released"] += 1
            with self.lock:
                self.ues.pop(str(m.get("ue_id")), None)
            self._to_xapp(m)
        elif t == "ho_outcome":
            st = str(m.get("status"))
            self.stats["ho_outcomes"][st] = self.stats["ho_outcomes"].get(st, 0) + 1
            self.events.append({"t": round(self.now_s(), 2), "event": "ho_outcome", "ue_id": m.get("ue_id"),
                                "status": st, "detail": m.get("detail", "")})
            self._to_xapp(m)
        elif t == "hello":
            log.info("bridge %s connected (%s)", m.get("node", ""), m.get("protocol", ""))
        else:
            log.warning("bridge sent unknown message type %r", t)

    def _serve_bridge(self, conn: _Line) -> None:
        while not self.stop_ev.is_set():
            m = conn.recv()
            if m is None:
                break
            try:
                self._on_bridge_msg(conn, m)
            except (KeyError, ValueError, TypeError) as e:
                log.warning("bad bridge message %r: %s", m, e)
        log.info("bridge %s disconnected", conn.name)

    def _accept(self, srv: socket.socket) -> None:
        srv.settimeout(0.2)
        while not self.stop_ev.is_set():
            try:
                sock, addr = srv.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._serve_bridge, args=(_Line(sock, f"bridge {addr[0]}:{addr[1]}"),),
                             daemon=True).start()

    # ---- xApp side -------------------------------------------------------------
    def _to_xapp(self, m: dict) -> None:
        if self.xapp is not None:
            try:
                self.xapp.send(m)
            except OSError as e:
                log.error("lost ran-xapp connection: %s", e)

    def _serve_xapp(self) -> None:
        while not self.stop_ev.is_set():
            m = self.xapp.recv()
            if m is None:
                log.error("ran-xapp closed the connection")
                self.stop_ev.set()
                return
            t = m.get("type")
            if t == "decision":
                a = m.get("action", "?")
                self.stats["decisions"][a] = self.stats["decisions"].get(a, 0) + 1
                continue
            if t == "error":
                log.warning("ran-xapp: %s", m.get("detail"))
                continue
            if t == "ho_command":
                self.stats["ho_commands"] += 1
                with self.lock:
                    ue = self.ues.get(str(m.get("ue_id")))
                self.events.append({"t": round(self.now_s(), 2), "event": "ho_command", "ue_id": m.get("ue_id"),
                                    "to": m.get("target_cell", {}).get("index"),
                                    "x_m": None if ue is None else round(ue.x_m, 1),
                                    "confidence": m.get("confidence")})
                if ue is None:
                    log.warning("ho_command for unknown UE %s dropped", m.get("ue_id"))
                    continue
                log.info("ho_command %s: UE %s -> cell %s (conf %.2f)", m.get("command_id"), ue.ue_id,
                         m.get("target_cell", {}).get("index"), m.get("confidence", 0.0))
                try:
                    ue.conn.send(m)
                except OSError as e:
                    log.error("could not reach the bridge for %s: %s", ue.ue_id, e)

    # ---- synthetic reports + rfsim coupling -----------------------------------
    def _synth_loop(self) -> None:
        period = self.cfg.report_period_s
        last = self.now_s()
        while not self.stop_ev.is_set():
            time.sleep(period / self.cfg.time_scale)
            now = self.now_s()
            dt, last = now - last, now
            with self.lock:
                ues = list(self.ues.values())
            for ue in ues:
                self._advance(ue, dt)
                if self.cfg.synthetic == "off" or ue.serving is None:
                    continue
                if self.cfg.synthetic == "auto" and now - ue.last_real_s < 1.0:
                    continue
                rsrp, sinr = self.radio(ue)
                s = self.idx.index(ue.serving)
                cells = [self.cells[i] for i in self.idx]
                rep = MeasReport(ue_id=ue.ue_id,
                                 serving=CellMeas(pci=cells[s].pci, nci=cells[s].nci, rsrp_dbm=round(float(rsrp[s]), 1),
                                                  sinr_db=round(sinr, 1)),
                                 neighbours=[CellMeas(pci=c.pci, nci=c.nci, rsrp_dbm=round(float(rsrp[k]), 1))
                                             for k, c in enumerate(cells) if k != s],
                                 timestamp_s=now, ue_ids=ue.ue_ids, speed_kmh=self.cfg.speed_kmh, seq=ue.seq,
                                 source="labdrive")
                ue.seq += 1
                self.stats["reports_synthetic"] += 1
                self._to_xapp(rep.to_dict())

    def _ploss_loop(self) -> None:
        """RF-simulator coupling in its own thread: a slow or absent UE telnet never delays reports."""
        while not self.stop_ev.is_set():
            time.sleep(self.cfg.ploss_period_s / self.cfg.time_scale)
            with self.lock:
                ues = list(self.ues.values())
            if ues:
                self._set_ploss(ues[0])

    def ploss_for(self, rsrp_dbm: float) -> float:
        c = self.cfg
        return float(min(max(c.ploss_at_ref_db + (c.rsrp_ref_dbm - rsrp_dbm), c.ploss_min_db), c.ploss_max_db))

    def _set_ploss(self, ue: _UE) -> None:
        """Couple the real RF simulator to the virtual drive (one UE: the rfsim has one channel per DU)."""
        d = np.maximum(np.linalg.norm(self.sites - self._pos(ue), axis=1), 10.0)
        rsrp = self.sim.tx_power_dbm - (128.1 + 37.6 * np.log10(d / 1000.0)) + ue.shadow
        host, _, port = self.cfg.ue_telnet.rpartition(":")
        for k, cell in enumerate(self.idx):
            ch = self.cfg.channels.get(cell)
            if ch is None:
                continue
            p = round(self.ploss_for(float(rsrp[k])))
            if self._last_ploss.get(ch) == p:
                continue
            try:
                with socket.create_connection((host, int(port)), timeout=1.0) as s:
                    s.sendall(f"channelmod modify {ch} ploss {p}\n".encode())
                self._last_ploss[ch] = p
            except OSError as e:
                log.warning("channelmod on %s failed: %s", self.cfg.ue_telnet, e)
                return

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> "LabDrive":
        host, _, port = self.cfg.xapp.rpartition(":")
        deadline = time.monotonic() + 60
        while True:
            try:
                sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=None)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        self.xapp = _Line(sock, "ran-xapp")
        self.xapp.send({"type": "hello", "node": "labdrive", "protocol": "ai-ran-llm/ran-bridge/1"})
        lh, _, lp = self.cfg.listen.rpartition(":")
        self.srv = socket.create_server((lh or "0.0.0.0", int(lp)))
        self.listen_port = self.srv.getsockname()[1]
        loops = [(self._accept, (self.srv,)), (self._serve_xapp, ()), (self._synth_loop, ())]
        if self.cfg.ue_telnet and self.cfg.channels:
            loops.append((self._ploss_loop, ()))
        for target, args in loops:
            threading.Thread(target=target, args=args, daemon=True).start()
        log.info("lab drive: ran-xapp %s, bridge listen %s:%s, synthetic=%s, rfsim coupling=%s",
                 self.cfg.xapp, lh or "0.0.0.0", self.listen_port, self.cfg.synthetic, bool(self.cfg.ue_telnet))
        return self

    def summary(self) -> dict:
        with self.lock:
            ues = {u.ue_id: {"serving": u.serving, "x_m": round(u.x_m, 1), "handovers": u.handovers}
                   for u in self.ues.values()}
        return {"elapsed_s": round(self.now_s(), 1), **self.stats, "ues": ues, "events": self.events[-50:]}

    def stop(self) -> None:
        self.stop_ev.set()
        for c in (self.xapp,):
            if c is not None:
                c.close()
        try:
            self.srv.close()
        except (AttributeError, OSError):
            pass
