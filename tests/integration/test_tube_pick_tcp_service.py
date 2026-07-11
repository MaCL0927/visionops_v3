"""End-to-end TCP trigger -> Runtime -> depth -> framed response test."""
from __future__ import annotations

import json
import socket
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.config import DEFAULT_CONFIG
from production.carton_line.tasks.tube_pick_vision.service import TubePickVisionService
from production.carton_line.tasks.tube_pick_vision.tcp_client import StarHashJsonCodec


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/runtime/infer_once":
            self._send(404, b"{}", "application/json")
            return
        result = {
            "schema_version": "1.0",
            "message_type": "inference_result",
            "status": "ok",
            "task_type": "detection",
            "frame_id": "frame-e2e",
            "result_id": "result-e2e",
            "image": {"width": 640, "height": 480},
            "model": {"model_id": "tube-pick-e2e"},
            "detections": [
                {
                    "id": "product-e2e",
                    "class_id": 0,
                    "class_name": "tube_product",
                    "score": 0.97,
                    "bbox_xyxy": [300, 220, 340, 260],
                    "center_xy": [320, 240],
                },
                {
                    "id": "separator-e2e",
                    "class_id": 1,
                    "class_name": "large_separator",
                    "score": 0.90,
                    "bbox_xyxy": [100, 100, 500, 140],
                    "center_xy": [300, 120],
                },
            ],
        }
        self._send(200, json.dumps(result).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/stream/depth.png":
            depth = np.full((240, 320), 876, dtype=np.uint16)
            ok, encoded = cv2.imencode(".png", depth)
            assert ok
            self._send(200, encoded.tobytes(), "image/png")
        elif self.path == "/stream/depth_meta":
            body = json.dumps(
                {"ok": True, "width": 320, "height": 240, "last_depth_ms": int(time.time() * 1000)}
            ).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"{}", "application/json")


def test_tcp_service_end_to_end(tmp_path) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_port = upstream.server_address[1]

    scheduler = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scheduler.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    scheduler.bind(("127.0.0.1", 0))
    scheduler.listen(1)
    scheduler.settimeout(5.0)
    scheduler_port = scheduler.getsockname()[1]

    config = deepcopy(DEFAULT_CONFIG)
    config["runtimes"]["pick"]["url"] = f"http://127.0.0.1:{upstream_port}"
    config["camera_bridge"]["depth_url"] = f"http://127.0.0.1:{upstream_port}/stream/depth.png"
    config["camera_bridge"]["depth_meta_url"] = f"http://127.0.0.1:{upstream_port}/stream/depth_meta"
    config["pick"]["tcp"]["server_host"] = "127.0.0.1"
    config["pick"]["tcp"]["server_port"] = scheduler_port
    config["pick"]["tcp"]["connect_timeout_ms"] = 500
    config["pick"]["tcp"]["read_timeout_ms"] = 200
    config["pick"]["tcp"]["reconnect_initial_ms"] = 100
    config["pick"]["tcp"]["reconnect_max_ms"] = 200
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}

    service = TubePickVisionService(config)
    service_thread = threading.Thread(target=service.run, daemon=True)
    service_thread.start()
    connection, _address = scheduler.accept()
    connection.settimeout(5.0)
    request = {
        "function": "vision0",
        "timestamp": [1752135960, 123456789],
        "triggerpos": 1752135960,
        "triggerindex": 9,
        "camera": "cam_1",
        "task_id": "tube_pick_vision",
    }
    connection.sendall(StarHashJsonCodec.encode(request))

    codec = StarHashJsonCodec()
    response = None
    deadline = time.time() + 5.0
    while time.time() < deadline and response is None:
        for document in codec.feed(connection.recv(65536)):
            response = document
            break

    service.stop()
    service_thread.join(timeout=2.0)
    connection.close()
    scheduler.close()
    upstream.shutdown()
    upstream.server_close()

    assert response is not None
    assert response["triggerindex"] == 9
    assert response["result"] == 0
    assert response["products"][0]["center"] == {"x": 320.0, "y": 240.0, "z": 876}
    assert response["separators"] == [
        {"class_id": 1, "class_name": "large_separator", "score": 0.9}
    ]
    assert response["types"] == []
