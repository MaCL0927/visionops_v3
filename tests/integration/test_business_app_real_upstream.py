"""M11 业务 App 读取真实风格 inference_result 的闭环测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, method: str = "GET") -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


class LatestResultHandler(BaseHTTPRequestHandler):
    result: dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/api/runtime/latest_result":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(self.result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


@contextmanager
def fake_runtime(result: dict[str, Any]):
    port = free_port()
    handler = type("Handler", (LatestResultHandler,), {"result": result})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def business_service(module: str, upstream_url: str, config_text: str | None = None):
    port = free_port()
    config_path = None
    tmp = None
    args = [
        sys.executable, "-m", module,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--upstream-kind", "runtime",
        "--upstream-url", upstream_url,
        "--poll-interval-ms", "5000",
        "--device-id", "test-edge",
    ]
    if config_text is not None:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
        tmp.write(config_text)
        tmp.close()
        config_path = tmp.name
        args.extend(["--config", config_path])
    process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"服务提前退出\n{stdout}\n{stderr}")
            try:
                if request_json(f"http://127.0.0.1:{port}/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("业务 App 服务未就绪")
        yield f"http://127.0.0.1:{port}"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def tube_result() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "device_id": "lb3576-dev",
        "component": "rknn_runtime",
        "timestamp_ms": 1780000000000,
        "trace_id": "trace-test-tube",
        "frame_id": "frame-hp60c-00000015",
        "source": "runtime:rknn",
        "status": "ok",
        "result_id": "result-rknn-00000015",
        "task_type": "detection",
        "model": {"model_name": "tube-rknn", "backend": "rknn"},
        "image": {"width": 640, "height": 480},
        "timing": {"total_ms": 88.0},
        "detections": [
            {"id": "det-1", "class_id": 0, "class_name": "tube", "score": 0.72, "bbox_xyxy": [100, 120, 180, 210], "center_xy": [140, 165]},
            {"id": "det-2", "class_id": 0, "class_name": "tube", "score": 0.65, "bbox_xyxy": [250, 130, 320, 220], "center_xy": [285, 175]},
        ],
    }


def partition_result() -> dict[str, Any]:
    detections = []
    for row in range(5):
        for col in range(8):
            x1 = 40 + col * 70
            y1 = 60 + row * 60
            detections.append({
                "id": f"cell-{row}-{col}", "class_id": 0, "class_name": "cell", "score": 0.82,
                "bbox_xyxy": [x1, y1, x1 + 40, y1 + 38], "center_xy": [x1 + 20, y1 + 19],
            })
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "device_id": "lb3576-dev",
        "component": "rknn_runtime",
        "timestamp_ms": 1780000001000,
        "trace_id": "trace-test-partition",
        "frame_id": "frame-hp60c-00000016",
        "source": "runtime:rknn",
        "status": "ok",
        "result_id": "result-rknn-00000016",
        "task_type": "detection",
        "model": {"model_name": "partition-rknn", "backend": "rknn"},
        "image": {"width": 640, "height": 480},
        "detections": detections,
    }


def test_tube_business_consumes_real_runtime_result() -> None:
    config = """
schema_version: '1.0'
kind: app
app: {name: carton_tube_check, version: '1.1'}
rules:
  target_class_names: [tube]
  target_class_ids: [0]
  accepted_task_types: [detection]
  score_threshold: 0.5
  allow_multi_target: true
  min_target_count: 1
  max_target_count: 8
  roi_xyxy: [0, 0, 640, 480]
  expected_center_xy: null
  center_tolerance_px: null
  register_base: 100
"""
    with fake_runtime(tube_result()) as upstream:
        with business_service("edge.gateway_adapter.apps.carton_tube_check.service", upstream, config) as app:
            status, decision = request_json(f"{app}/api/app/evaluate_once", "POST")
            assert status == 200
            assert decision["final_label"] == "OK"
            assert decision["object_count"] == 2
            assert decision["details"]["target_count"] == 2
            registers = request_json(f"{app}/api/app/registers")[1]["registers"]
            assert registers[2]["name"] == "final_code" and registers[2]["value"] == 0


def test_partition_business_consumes_real_runtime_result() -> None:
    config = """
schema_version: '1.0'
kind: app
app: {name: carton_partition_check, version: '1.1'}
rules:
  target_class_names: [cell]
  target_class_ids: [0]
  defect_class_names: [missing_cell, broken_partition, foreign_body, defect]
  defect_class_ids: [1, 2, 3]
  accepted_task_types: [detection]
  score_threshold: 0.5
  defect_score_threshold: 0.5
  expected_rows: 5
  expected_cols: 8
  expected_cell_count: 40
  min_cell_count: 40
  max_cell_count: 40
  roi_xyxy: [0, 0, 640, 480]
  register_base: 200
"""
    with fake_runtime(partition_result()) as upstream:
        with business_service("edge.gateway_adapter.apps.carton_partition_check.service", upstream, config) as app:
            status, decision = request_json(f"{app}/api/app/evaluate_once", "POST")
            assert status == 200
            assert decision["final_label"] == "OK"
            assert decision["details"]["cell_count"] == 40
            assert decision["details"]["grid_rows"] == 5
            registers = request_json(f"{app}/api/app/registers")[1]["registers"]
            assert registers[6]["name"] == "cell_count" and registers[6]["value"] == 40
