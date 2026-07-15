"""WebSocket trigger -> Runtime -> depth -> SDK deprojection integration test."""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.config import DEFAULT_CONFIG
from production.carton_line.tasks.tube_pick_vision.mock_robot_client import Client, _read_server_frame
from production.carton_line.tasks.tube_pick_vision.service import TubePickVisionService


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
        if self.path == "/api/runtime/infer_once":
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
                    {"id": "p", "class_id": 0, "class_name": "tube", "score": 0.97, "center_xy": [320, 240], "bbox_xyxy": [300, 220, 340, 260]},
                    {"id": "s", "class_id": 1, "class_name": "separator", "score": 0.90, "center_xy": [300, 120], "bbox_xyxy": [100, 100, 500, 140]},
                ],
            }
            self._send(200, json.dumps(result).encode(), "application/json")
            return
        if self.path == "/api/coordinate/deproject":
            size = int(self.headers.get("Content-Length", "0"))
            points = json.loads(self.rfile.read(size))["points"]
            result = {
                "ok": True,
                "points": [
                    {"valid": point[2] > 0, "position_camera": [point[0] - 320, point[1] - 240, point[2]] if point[2] > 0 else [0, 0, 0]}
                    for point in points
                ],
            }
            self._send(200, json.dumps(result).encode(), "application/json")
            return
        self._send(404, b"{}", "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/runtime/status":
            self._send(200, json.dumps({"camera_connected": True, "loaded_model": {"model_id": "tube-pick-e2e"}}).encode(), "application/json")
        elif self.path == "/stream/depth.png":
            depth = np.full((480, 640), 876, dtype=np.uint16)
            ok, encoded = cv2.imencode(".png", depth)
            assert ok
            self._send(200, encoded.tobytes(), "image/png")
        elif self.path == "/health":
            self._send(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "camera_started": True,
                        "camera_connected": True,
                        "camera_state": "running",
                        "last_color_age_ms": 10,
                        "last_depth_age_ms": 10,
                        "reconnect_attempt_count": 1,
                        "reconnect_success_count": 1,
                    }
                ).encode(),
                "application/json",
            )
        else:
            self._send(404, b"{}", "application/json")


def test_websocket_trigger_request_id_round_trip(tmp_path) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    port = upstream.server_address[1]

    config = deepcopy(DEFAULT_CONFIG)
    config["runtimes"]["pick"]["url"] = f"http://127.0.0.1:{port}"
    config["camera_bridge"]["base_url"] = f"http://127.0.0.1:{port}"
    config["camera_bridge"]["depth_url"] = f"http://127.0.0.1:{port}/stream/depth.png"
    config["camera_bridge"]["deproject_url"] = f"http://127.0.0.1:{port}/api/coordinate/deproject"
    config["pick"]["websocket"]["listen_host"] = "127.0.0.1"
    config["pick"]["websocket"]["listen_port"] = 0
    config["pick"]["websocket"]["auto_start"] = False
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}

    service = TubePickVisionService(config)
    service.start()
    client = Client(f"ws://127.0.0.1:{service.websocket.port}/vision")
    try:
        # Initial status.
        opcode, payload = _read_server_frame(client.sock)
        assert opcode == 1
        assert json.loads(payload)["type"] == "status"

        client.send_json({"type": "control", "command": "trigger", "request_id": 77})
        detection = None
        deadline = time.time() + 5
        while time.time() < deadline:
            opcode, payload = _read_server_frame(client.sock)
            if opcode != 1:
                continue
            document = json.loads(payload)
            if document.get("type") == "detection" and document.get("request_id") == 77:
                detection = document
                break
        assert detection is not None
        assert detection["items"][0]["class_id"] == 1
        assert detection["items"][0]["position_camera"] == [-20.0, -120.0, 876.0]
        assert detection["items"][1]["class_id"] == 0
        assert detection["items"][1]["position_camera"] == [0.0, 0.0, 876.0]
    finally:
        client.close()
        service.stop()
        upstream.shutdown()
        upstream.server_close()
