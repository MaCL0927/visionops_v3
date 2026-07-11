"""Unit tests for the TCP-triggered tube-pick production task."""
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
from production.carton_line.tasks.tube_pick_vision.tcp_client import StarHashJsonCodec


def _settings() -> dict:
    return deepcopy(DEFAULT_CONFIG["pick"]["algorithm"])


def _runtime_result() -> dict:
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "detection",
        "frame_id": "frame-1",
        "result_id": "result-1",
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
        ],
    }


def _depth_png(value: int = 1234) -> bytes:
    depth = np.full((240, 320), value, dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", depth)
    assert ok
    return encoded.tobytes()


def test_algorithm_returns_product_xyz_and_separator_class_only() -> None:
    algorithm = TubePickAlgorithm(_settings())
    classified = algorithm.classify(_runtime_result())
    depth = decode_depth_png(_depth_png(1234))
    payload, debug = algorithm.build_detection_payload(classified, depth)

    assert payload["coordinate_units"] == {"x": "pixel", "y": "pixel", "z": "mm"}
    assert payload["products"] == [
        {
            "class_id": 0,
            "class_name": "tube_product",
            "score": 0.95,
            "center": {"x": 320.0, "y": 240.0, "z": 1234},
            "depth_valid": True,
        }
    ]
    assert payload["separators"] == [
        {"class_id": 1, "class_name": "large_separator", "score": 0.91}
    ]
    assert "center" not in payload["separators"][0]
    assert "bbox_xyxy" not in payload["separators"][0]
    assert debug["products"][0]["depth_x"] == 160
    assert debug["products"][0]["depth_y"] == 120


def test_algorithm_does_not_require_depth_when_only_separator_is_detected() -> None:
    result = _runtime_result()
    result["detections"] = [result["detections"][1]]
    algorithm = TubePickAlgorithm(_settings())
    classified = algorithm.classify(result)
    payload, _debug = algorithm.build_detection_payload(classified, None)

    assert payload["product_count"] == 0
    assert payload["separator_count"] == 1
    assert payload["depth"]["required"] is False
    assert payload["invalid_depth_count"] == 0


def test_star_hash_codec_handles_split_and_sticky_packets() -> None:
    codec = StarHashJsonCodec()
    assert codec.feed(b'noise*{"triggerindex":1') == []
    messages = codec.feed(b'}#*{"triggerindex":2}#')
    assert [item["triggerindex"] for item in messages] == [1, 2]
    encoded = StarHashJsonCodec.encode({"ok": True})
    assert encoded == b'*{"ok":true}#'


def test_service_echoes_trigger_and_avoids_duplicate_inference(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    calls = {"infer": 0, "depth": 0}

    def infer_once() -> dict:
        calls["infer"] += 1
        return _runtime_result()

    def get_bytes(_url: str) -> bytes:
        calls["depth"] += 1
        return _depth_png(987)

    service.runtime.infer_once = infer_once  # type: ignore[method-assign]
    service.http.get_bytes = get_bytes  # type: ignore[method-assign]
    request = {
        "function": "vision0",
        "timestamp": [1752135960, 123456789],
        "triggerpos": 1752135960,
        "triggerindex": 7,
        "camera": "cam_1",
        "task_id": "task_001",
    }

    first = service.handle_message(request)
    second = service.handle_message(request)
    assert first is not None and second is not None
    assert first["triggerindex"] == 7
    assert first["timestamp"] == [1752135960, 123456789]
    assert first["products"][0]["center"] == {"x": 320.0, "y": 240.0, "z": 987}
    assert first["types"] == [] and first["poses"] == []
    assert first == second
    assert calls == {"infer": 1, "depth": 1}


def test_service_skips_depth_request_for_separator_only(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["pick"]["debug"] = {"save_every_trigger": False, "save_root": str(tmp_path)}
    service = TubePickVisionService(config)
    result = _runtime_result()
    result["detections"] = [result["detections"][1]]
    service.runtime.infer_once = lambda: result  # type: ignore[method-assign]

    def unexpected_depth(_url: str) -> bytes:
        raise AssertionError("separator-only response must not fetch depth")

    service.http.get_bytes = unexpected_depth  # type: ignore[method-assign]
    response = service.handle_message({"triggerindex": 11, "function": "vision0"})
    assert response is not None
    assert response["result"] == 0
    assert response["products"] == []
    assert response["separator_detected"] is True


def test_installed_line_config_merge_preserves_site_values_and_adds_pick() -> None:
    defaults = {"runtimes": {"partition": {"url": "default"}, "pick": {"url": "pick"}}, "pick": {"tcp": {"server_host": "127.0.0.1"}}}
    current = {"runtimes": {"partition": {"url": "site"}}, "site_only": 1}
    merged = merge(defaults, current)
    assert merged["runtimes"]["partition"]["url"] == "site"
    assert merged["runtimes"]["pick"]["url"] == "pick"
    assert merged["pick"]["tcp"]["server_host"] == "127.0.0.1"
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
        assert registers == {
            "schema_version": "1.0",
            "message_type": "register_snapshot",
            "status": "not_applicable",
            "protocol": "tcp_json",
            "registers": [],
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
