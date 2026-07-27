"""Collector Runtime 本机 raw HTTP 代理回归测试。"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from apps.collector_web.backend.runtime_client import RuntimeClient


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    last_body = b""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_body = self.rfile.read(length)
        payload = json.dumps(
            {
                "message_type": "inference_result",
                "status": "ok",
                "body": type(self).last_body.decode("utf-8"),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-VisionOps-Test", "raw")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/missing":
            payload = b'{"error":"missing"}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = b"\xff\xd8raw-jpeg\xff\xd9"
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _serve():
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port


def test_collector_runtime_client_uses_raw_socket_for_local_runtime() -> None:
    server, thread, port = _serve()
    try:
        client = RuntimeClient(
            f"http://127.0.0.1:{port}",
            raw_local_enabled=True,
        )
        response = client.request("POST", "/api/runtime/infer_once", body=b"{}")

        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.transport == "raw_socket"
        assert response.headers["x-visionops-test"] == "raw"
        assert response.json()["body"] == "{}"
        assert _Handler.last_body == b"{}"
        assert response.elapsed_ms >= 0.0

        status = client.transport_status()
        assert status["raw_request_count"] == 1
        assert status["urllib_request_count"] == 0
        assert status["last_transport"] == "raw_socket"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_collector_raw_runtime_preserves_image_and_http_error() -> None:
    server, thread, port = _serve()
    try:
        client = RuntimeClient(
            f"http://127.0.0.1:{port}",
            raw_local_enabled=True,
        )
        snapshot = client.request("GET", "/api/runtime/snapshot.jpg")
        assert snapshot.status_code == 200
        assert snapshot.content_type == "image/jpeg"
        assert snapshot.body.startswith(b"\xff\xd8")
        assert snapshot.transport == "raw_socket"

        missing = client.request("GET", "/missing")
        assert missing.status_code == 404
        assert missing.content_type == "application/json"
        assert missing.json() == {"error": "missing"}
        assert missing.transport == "raw_socket"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_non_runtime_client_keeps_urllib_when_raw_is_disabled() -> None:
    server, thread, port = _serve()
    try:
        client = RuntimeClient(f"http://127.0.0.1:{port}")
        response = client.request("GET", "/api/runtime/snapshot.jpg")
        assert response.status_code == 200
        assert response.transport == "urllib"
        status = client.transport_status()
        assert status["raw_request_count"] == 0
        assert status["urllib_request_count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
