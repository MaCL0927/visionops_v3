from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from production.common.runtime_ipc import RuntimeIpcClient


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        if size:
            self.rfile.read(size)
        body = json.dumps(
            {
                "message_type": "inference_result",
                "status": "ok",
                "timing": {"total_ms": 12.3},
                "detections": [],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-VisionOps-Http-Queue-Ms", "0.08")
        self.send_header("X-VisionOps-Http-Route-Ms", "12.7")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_runtime_ipc_uses_raw_socket_and_preserves_timing_headers():
    server, thread = _serve()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        client = RuntimeIpcClient(base, 2.0, {"raw_http_enabled": True})
        response = client.infer_once_raw()
        assert response.transport == "raw_socket"
        assert response.status_code == 200
        assert response.header_float("x-visionops-http-queue-ms") == 0.08
        assert response.header_float("x-visionops-http-route-ms") == 12.7
        result = client.decode_inference(response.body)
        assert result["status"] == "ok"
        status = client.transport_status()
        assert status["raw_request_count"] == 1
        assert status["urllib_request_count"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_ipc_can_force_urllib_compatibility_path():
    server, thread = _serve()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        client = RuntimeIpcClient(
            base,
            2.0,
            {
                "raw_http_enabled": False,
                "raw_http_fallback_urllib": True,
            },
        )
        response = client.infer_once_raw()
        assert response.transport == "urllib"
        assert client.decode_inference(response.body)["status"] == "ok"
        status = client.transport_status()
        assert status["raw_request_count"] == 0
        assert status["urllib_request_count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
