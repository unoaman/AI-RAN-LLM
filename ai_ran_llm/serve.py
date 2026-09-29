"""Minimal HTTP endpoint so a near-RT RIC / xApp can query the model.

POST /v1/handover   body: a measurement report (see HandoverLLM.handle_report)
GET  /healthz
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .inference import HandoverLLM


def make_handler(llm: HandoverLLM, ho_threshold: float = 0.35, min_confidence: float = 0.3):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/handover":
                return self._send(404, {"error": "not found"})
            try:
                report = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                reports = report if isinstance(report, list) else [report]
                out = [dict(llm.handle_report(r, ho_threshold, min_confidence), ue_id=r.get("ue_id")) for r in reports]
                self._send(200, out if isinstance(report, list) else out[0])
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": f"bad report: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(checkpoint: str, host: str = "0.0.0.0", port: int = 8080, ho_threshold: float = 0.35,
          min_confidence: float = 0.3):
    llm = HandoverLLM.load(checkpoint)
    server = ThreadingHTTPServer((host, port), make_handler(llm, ho_threshold, min_confidence))
    print(f"HandoverLLM xApp listening on http://{host}:{port}/v1/handover")
    server.serve_forever()
