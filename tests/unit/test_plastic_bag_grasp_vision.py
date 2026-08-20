"""plastic_bag_grasp detection-centre, protocol and FPS fast-path tests."""
from __future__ import annotations

from copy import deepcopy
import json

from production.plastic_bag_grasp.config import DEFAULT_CONFIG, load_config
from production.plastic_bag_grasp.tasks.plastic_bag_grasp_vision.algorithm import PlasticBagGraspAlgorithm


def _runtime_result(width=1280, height=720):
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "detection",
        "frame_id": "frame-1",
        "result_id": "result-1",
        "capture_timestamp_ms": 1787100000123,
        "image": {"width": width, "height": height},
        "model": {
            "model_id": "rk3576-252-plastic-bag-grasp-det",
            "model_name": "plastic_bag_grasp_det",
        },
        "timing": {"total_ms": 31.07},
        "detections": [
            {
                "id": "bag-1",
                "class_id": 0,
                "class_name": "plastic_bag",
                "score": 0.95,
                "bbox_xyxy": [668.0, 344.0, 970.0, 596.0],
                "center_xy": [819.0, 470.0],
            }
        ],
    }


def _algorithm():
    return PlasticBagGraspAlgorithm(deepcopy(DEFAULT_CONFIG["algorithm"]))


def _raw_runtime_response(payload=None, transport="raw_socket"):
    from production.common.runtime_ipc import TimedHttpResponse

    document = payload if payload is not None else _runtime_result()
    body = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return TimedHttpResponse(
        body=body,
        status_code=200,
        headers={
            "content-length": str(len(body)),
            "x-visionops-http-queue-ms": "0.02",
            "x-visionops-http-route-ms": "31.07",
        },
        connect_ms=0.10,
        send_ms=0.03,
        headers_wait_ms=31.2,
        body_read_ms=0.08,
        total_ms=31.41,
        transport=transport,
    )


def test_bbox_center_is_the_robot_grasp_point_and_source_size_is_dynamic():
    result = _algorithm().evaluate(_runtime_result())
    assert result.image_width == 1280
    assert result.image_height == 720
    assert len(result.items) == 1
    item = result.items[0]
    assert item == {
        "id": 0,
        "class_id": 0,
        "confidence": 0.95,
        "position_camera": [0.0, 0.0, 0.0],
        "center_px": [819.0, 470.0],
    }


def test_bbox_midpoint_is_used_when_runtime_center_xy_is_absent():
    payload = _runtime_result()
    payload["detections"][0].pop("center_xy")
    result = _algorithm().evaluate(payload)
    assert result.items[0]["center_px"] == [819.0, 470.0]


def test_only_best_plastic_bag_is_returned():
    payload = _runtime_result()
    payload["detections"].extend(
        [
            {
                "id": "duplicate-low",
                "class_id": 0,
                "class_name": "plastic_bag",
                "score": 0.80,
                "bbox_xyxy": [600, 300, 1000, 650],
            },
            {
                "id": "other",
                "class_id": 9,
                "class_name": "robot",
                "score": 0.99,
                "bbox_xyxy": [10, 10, 100, 100],
            },
        ]
    )
    result = _algorithm().evaluate(payload)
    assert len(result.items) == 1
    assert result.selected[0]["source_id"] == "bag-1"
    assert {entry["reason"] for entry in result.ignored} == {"class_not_used"}


def test_robot_message_keeps_unified_five_item_fields_and_depth_xyz(monkeypatch, tmp_path):
    from production.plastic_bag_grasp.tasks.plastic_bag_grasp_vision.service import PlasticBagGraspVisionService

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    config["websocket"]["listen_port"] = 0
    service = PlasticBagGraspVisionService(config)
    monkeypatch.setattr(service.runtime, "infer_once_raw", _raw_runtime_response)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)
    monkeypatch.setattr(
        service.depth,
        "sample",
        lambda points, width, height: (
            [
                {
                    "depth_valid": True,
                    "position_camera": [21.4, -18.7, 842.6],
                    "depth_mm": 843,
                    "valid_pixels": 20,
                }
            ],
            {"ok": True, "mode": "test"},
        ),
    )

    decision = service.evaluate_once("request-1001", "plastic_bag_grasp")
    assert decision["status"] == "ok"
    robot = decision["robot_message"]
    assert robot["request_id"] == "request-1001"
    assert robot["trigger_task_id"] == "plastic_bag_grasp"
    assert robot["image"] == {"width": 1280, "height": 720}
    assert robot["coordinate_frame"] == "color_camera"
    assert robot["coordinate_unit"] == "mm"
    assert robot["fault_code"] == 0
    assert len(robot["items"]) == 1
    item = robot["items"][0]
    assert set(item) == {"id", "class_id", "confidence", "position_camera", "center_px"}
    assert item["center_px"] == [819.0, 470.0]
    assert item["position_camera"] == [21.4, -18.7, 842.6]


def test_depth_failure_does_not_drop_valid_rgb_target(monkeypatch, tmp_path):
    from production.plastic_bag_grasp.tasks.plastic_bag_grasp_vision.service import PlasticBagGraspVisionService

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    config["websocket"]["listen_port"] = 0
    service = PlasticBagGraspVisionService(config)
    monkeypatch.setattr(service.runtime, "infer_once_raw", _raw_runtime_response)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)

    def _depth_fail(*args, **kwargs):
        raise RuntimeError("transparent surface has no reliable depth")

    monkeypatch.setattr(service.depth, "sample", _depth_fail)
    robot = service.evaluate_once("request-1002", "plastic_bag_grasp")["robot_message"]
    assert robot["fault_code"] == 0
    assert len(robot["items"]) == 1
    assert robot["items"][0]["center_px"] == [819.0, 470.0]
    assert robot["items"][0]["position_camera"] == [0.0, 0.0, 0.0]


def test_fps_fast_path_contract_has_no_legacy_five_hz_throttle():
    config = load_config()
    assert config["runtime"]["url"].endswith(":28088")
    assert config["app"]["listen_port"] == 19214
    assert config["collector"]["listen_port"] == 18097
    assert config["websocket"]["listen_port"] == 9001
    assert config["app"]["default_production_inference_fps"] == 30.0
    assert "detection_hz" not in config["websocket"]
    assert "push_hz" not in config["websocket"]
    assert config["pipeline"] == {
        "enabled": True,
        "result_queue_size": 1,
        "max_result_age_ms": 500,
    }
    assert config["runtime_ipc"] == {
        "raw_http_enabled": True,
        "raw_http_fallback_urllib": True,
        "max_response_bytes": 32 * 1024 * 1024,
    }
    assert config["algorithm"]["image"]["require_fixed_size"] is False


def test_production_fps_persists_at_app_level_without_ws_push_hz(tmp_path):
    from production.plastic_bag_grasp.tasks.plastic_bag_grasp_vision.service import PlasticBagGraspVisionService

    config = load_config()
    path = tmp_path / "fps.json"
    config["app"]["inference_settings_path"] = str(path)
    config["websocket"]["listen_port"] = 0
    service = PlasticBagGraspVisionService(config)
    settings = service.set_production_fps(30.0)
    assert settings["production_inference_fps"] == 30.0
    assert settings["push_mode"] == "every_completed_result"
    service2 = PlasticBagGraspVisionService(config)
    assert service2.production_fps() == 30.0
