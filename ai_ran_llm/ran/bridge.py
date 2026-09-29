"""ran-bridge transport: NDJSON over TCP (see ``messages.py`` for the protocol).

The xApp runs :class:`BridgeServer`; RAN-side agents connect to it, stream
``meas_report`` / ``ho_outcome`` / ``ue_release`` messages, and receive
``ho_command`` (and optional ``decision``) messages for the UEs they reported.
:class:`BridgeClient` is the RAN side (used by the fake gNB and handy for
writing an agent).
"""

from __future__ import annotations

import queue
import socket
import threading
import time

from .messages import HandoverCommand, Hello, ProtocolError, decode, encode


class _Conn:
    def __init__(self, sock: socket.socket, addr):
        self.sock, self.addr = sock, addr
        self.lock = threading.Lock()
        self.node = f"{addr[0]}:{addr[1]}"

    def send(self, msg) -> None:
        with self.lock:
            self.sock.sendall(encode(msg))


class BridgeServer:
    """Accepts RAN agents; incoming messages go to ``events`` as ``(self, msg)``."""

    name = "bridge"

    def __init__(self, host: str = "127.0.0.1", port: int = 7000, events: queue.Queue | None = None):
        self.events = events if events is not None else queue.Queue()
        self._srv = socket.create_server((host, port), reuse_port=False)
        self.host, self.port = self._srv.getsockname()[:2]
        self._routes: dict[str, _Conn] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.protocol_errors = 0

    def start(self) -> "BridgeServer":
        threading.Thread(target=self._accept_loop, daemon=True, name="bridge-accept").start()
        return self

    def _accept_loop(self):
        self._srv.settimeout(0.2)
        while not self._stop.is_set():
            try:
                sock, addr = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn = _Conn(sock, addr)
            threading.Thread(target=self._read_loop, args=(conn,), daemon=True, name=f"bridge-{addr}").start()

    def _read_loop(self, conn: _Conn):
        f = conn.sock.makefile("rb")
        try:
            for line in f:
                if not line.strip():
                    continue
                try:
                    msg = decode(line)
                except ProtocolError as e:
                    self.protocol_errors += 1
                    try:
                        conn.send({"type": "error", "detail": str(e)})
                    except OSError:
                        break
                    continue
                if isinstance(msg, Hello):
                    conn.node = msg.node or conn.node
                    continue
                ue = getattr(msg, "ue_id", None)
                if ue is not None:
                    with self._lock:
                        self._routes[ue] = conn
                self.events.put((self, msg))
        except OSError:
            pass
        finally:
            with self._lock:
                for ue in [u for u, c in self._routes.items() if c is conn]:
                    del self._routes[ue]
            conn.sock.close()

    def send(self, ue_id: str, msg) -> bool:
        """Send to the agent that last reported `ue_id`. False if it is not connected."""
        with self._lock:
            conn = self._routes.get(ue_id)
        if conn is None:
            return False
        try:
            conn.send(msg)
            return True
        except OSError:
            return False

    # Actuator interface
    def send_handover(self, cmd: HandoverCommand) -> None:
        if not self.send(cmd.ue_id, cmd):
            raise ConnectionError(f"no bridge connection for UE {cmd.ue_id}")

    def close(self):
        self._stop.set()
        self._srv.close()


class BridgeClient:
    """RAN-side connection to a BridgeServer."""

    def __init__(self, host: str, port: int, node: str = "", timeout: float | None = 10.0,
                 connect_timeout_s: float = 30.0):
        deadline = time.monotonic() + connect_timeout_s
        while True:                               # the xApp may still be starting (loading the model)
            try:
                self.sock = socket.create_connection((host, port), timeout=timeout)
                break
            except (ConnectionRefusedError, socket.timeout):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rfile = self.sock.makefile("rb")
        self.send(Hello(node=node))

    def send(self, msg) -> None:
        self.sock.sendall(encode(msg))

    def send_many(self, msgs) -> None:
        self.sock.sendall(b"".join(encode(m) for m in msgs))

    def recv(self):
        """Next message (blocking, subject to the socket timeout); None on EOF."""
        line = self._rfile.readline()
        return decode(line) if line else None

    def close(self):
        self.sock.close()
