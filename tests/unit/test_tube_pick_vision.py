"""Unit tests for the external-box tube_pick_vision task."""
from __future__ import annotations

from copy import deepcopy
import json
import threading
import urllib.request

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.config import DEFAULT_CONFIG
from production.carton_line.deploy.merge_line_config import merge
from production.carton_line.tasks.tube_pick_vision.algorithm import TubePickAlgorithm, decode_depth_png
from production.carton_line.tasks.tube_pick_vision.service import (
    ReusableThreadingHTTPServer,
    StatusHandler,
    TubePickVisionService,
)


def _settings() -> dict:
    return deepcopy(DEFAULT_CONFIG["pick"]["algorithm"])


def _runtime_result() -> dict:
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "detection",
        "frame_id": "runtime-frame-1",
        "result_id": "runtime-result-1",
        "image": {"width": 640, "height": 480},
        "model": {"model_id": "tube-pick-test"},
        "detections": [
            {
                "id": "product-1",
                "class_id": 0,
                "class_name": "tube_product",
                "score": 0.95,
                "bbox_xyxy": [300, 220, 340, 260],
                "center_xy": [320, 240],
            },
            {
                "id": "separator-1",
                "class_id": 1,
                "class_name": "large_separator",
                "score": 0.91,
                "bbox_xyxy": [100, 150, 500, 190],
                "center_xy": [300, 170],
            },
            {
                "id": "lying-1",
                "class_id": 2,
                "class_name": "lying",
                "score": 0.89,
                "bbox_xyxy": [180, 180, 260, 220],
                "center_xy": [220, 200],
            },
        ],
    }


def _depth_png(value: int = 1234) -> bytes:
    depth = np.full((480, 640), value, dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", depth)
    assert ok
    return encoded.tobytes()


def test_algorithm_samples_product_separator_and_lying_centres() -> None:
    algorithm = TubePickAlgorithm(_settings())
    classified = algorithm.classify(_runtime_result())
    depth = decode_depth_png(_depth_png(1234))
    sampled = algorithm.sample_items(classified, depth)

    assert [item["semantic"] for item in sampled] == ["separator", "lying", "product"]
    assert sampled[0]["center_x"] == 300.0
    assert sampled[0]["center_y"] == 170.0
    assert sampled[0]["z_mm"] == 1234
    assert sampled[1]["z_mm"] == 1234
    assert sampled[2]["z_mm"] == 1234

    external = algorithm.build_external_items(
        sampled,
        [[-20, -30, 1234], [-40, -10, 1234], [0, 0, 1234]],
    )
    assert external == [
        {
            "id": 0,
            "class_id": 1,
            "confidence": 0.91,
            "position_camera": [-20.0, -30.0, 1234.0],
            "center_px": [300.0, 170.0],
        },
        {
            "id": 1,
            "class_id": 2,
            "confidence": 0.89,
            "position_camera": [-40.0, -10.0, 1234.0],
            "center_px": [220.0, 200.0],
        },
        {
            "id": 2,
            "class_id": 0,
            "confidence": 0.95,
            "position_camera": [0.0, 0.0, 1234.0],
            "center_px": [320.0, 240.0],
        },
    ]


def test_invalid_depth_returns_zero_camera_point() -> None:
    algorithm = TubePickAlgorithm(_settings())
    classified = algorithm.classify(_runtime_result())
    depth = np.zeros((480, 640), dtype=np.uint16)
    sampled = algorithm.sample_items(classified, depth)
    output = algorithm.build_external_items(sampled, [[9, 9, 9], [8, 8, 8], [7, 7, 7]])
    assert output[0]["position_camera"] == [0.0, 0.0, 0.0]
    assert output[1]["position_camera"] == [0.0, 0.0, 0.0]
    assert output[2]["position_camera"] == [0.0, 0.0, 0.0]


def test_fixed_640x480_contract_is_enforced() -> None:
    result = _runtime_result()
    result["image"] = {"width": 1280, "height": 720}
    algorithm = TubePickAlgorithm(_settings())
    try:
        algorithm.classify(result)
    except ValueError as error:
        assert "640x480" in str(error)
    else:
        raise AssertionError("size mismatch must fail")


def test_service_builds_camera_coordinate_detection(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    service.runtime.infer_once = _runtime_result  # type: ignore[method-assign]
    service.bridge.health = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "camera_started": True,
        "camera_connected": True,
        "camera_state": "running",
        "last_color_age_ms": 10,
        "last_depth_age_ms": 10,
    }
    service.bridge.get_depth = lambda _health=None: (decode_depth_png(_depth_png(987)), {"last_depth_age_ms": 10}, _depth_png(987))  # type: ignore[method-assign]
    service.bridge.deproject = lambda points: ([[float(i), float(i + 1), float(point[2])] for i, point in enumerate(points)], {"ok": True})  # type: ignore[method-assign]

    response = service.evaluate_once("req-7")
    assert response["type"] == "detection"
    assert response["request_id"] == "req-7"
    assert response["coordinate_frame"] == "color_camera"
    assert len(response["items"]) == 3
    assert response["items"][0]["class_id"] == 1
    assert response["items"][1]["class_id"] == 2
    assert response["items"][2]["class_id"] == 0
    assert response["items"][1]["position_camera"] == [1.0, 2.0, 987.0]
    assert response["items"][2]["position_camera"] == [2.0, 3.0, 987.0]


def test_installed_line_config_merge_preserves_site_values_and_adds_websocket() -> None:
    defaults = {
        "runtimes": {"partition": {"url": "default"}, "pick": {"url": "pick"}},
        "pick": {"websocket": {"listen_port": 9001}},
    }
    current = {"runtimes": {"partition": {"url": "site"}}, "site_only": 1}
    merged = merge(defaults, current)
    assert merged["runtimes"]["partition"]["url"] == "site"
    assert merged["runtimes"]["pick"]["url"] == "pick"
    assert merged["pick"]["websocket"]["listen_port"] == 9001
    assert merged["site_only"] == 1


def test_status_http_exposes_collector_compatibility_endpoints(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    server = ReusableThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
    server.service = service  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/app/status", timeout=2) as response:
            status = json.loads(response.read())
        with urllib.request.urlopen(base + "/api/app/registers", timeout=2) as response:
            registers = json.loads(response.read())
        assert status["message_type"] == "tube_pick_service_status"
        assert status["websocket"]["listen_port"] == 9001
        assert registers["protocol"] == "websocket"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_camera_disconnect_suppresses_old_detection_results(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    called = {"runtime": 0}

    def infer_once() -> dict:
        called["runtime"] += 1
        return _runtime_result()

    service.runtime.infer_once = infer_once  # type: ignore[method-assign]
    service.bridge.health = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "camera_started": True,
        "camera_connected": False,
        "camera_state": "reconnecting",
        "fault_code": "CAMERA_RECONNECTING",
        "fault_numeric_code": 3102,
        "alarm_active": True,
        "state_age_ms": 20000,
        "last_color_age_ms": 8000,
        "last_depth_age_ms": 8100,
        "last_error": "USB camera disconnected",
        "reconnect_attempt_count": 4,
    }

    response = service.evaluate_once("camera-offline-1")
    assert called["runtime"] == 0
    assert response["items"] == []
    assert response["error"]["code"] == "CAMERA_DISCONNECTED"
    assert response["camera_state"] == "reconnecting"
    assert response["alarm"]["numeric_code"] == 3102
    assert response["alarm"]["modbus_tcp_reserved"] is True


def test_status_uses_bridge_freshness_as_camera_source_of_truth(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    service.runtime.status = lambda: {  # type: ignore[method-assign]
        "camera_connected": True,
        "loaded_model": {"model_id": "test-model"},
    }
    service.bridge.health = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "camera_connected": False,
        "camera_state": "offline",
        "fault_code": "CAMERA_OFFLINE",
        "fault_numeric_code": 3103,
        "alarm_active": True,
        "last_color_age_ms": -1,
        "last_depth_age_ms": -1,
        "last_error": "device not found",
    }

    status = service._status_message()
    assert status["camera_connected"] is False
    assert status["runtime_camera_connected"] is True
    assert status["camera_state"] == "offline"
    assert status["error_code"] == "CAMERA_OFFLINE"
    assert status["alarm"]["active"] is True
    assert status["alarm"]["modbus_tcp_implemented"] is False
