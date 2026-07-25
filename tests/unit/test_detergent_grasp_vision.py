"""Detergent-grasp OBB parsing, association and protocol tests."""
from __future__ import annotations

from copy import deepcopy

from production.detergent_grasp.config import DEFAULT_CONFIG, load_config
from production.detergent_grasp.tasks.detergent_grasp_vision.algorithm import (
    DetergentGraspAlgorithm,
    resolve_handle_direction_angle,
)


def _det(source_id, class_id, name, score, points):
    return {
        "id": source_id,
        "class_id": class_id,
        "class_name": name,
        "score": score,
        "obb": {"points": points},
    }


def _runtime_result():
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "obb",
        "frame_id": "frame-1",
        "result_id": "result-1",
        "capture_timestamp_ms": 1784900000123,
        "image": {"width": 640, "height": 480},
        "detections": [
            _det("big-1", 0, "big", 0.91, [[80, 100], [180, 100], [180, 260], [80, 260]]),
            _det("head-1", 1, "head", 0.88, [[115, 110], [145, 110], [145, 135], [115, 135]]),
            _det("small-1", 3, "small", 0.84, [[220, 110], [280, 150], [240, 215], [180, 175]]),
            _det("head-2", 1, "head", 0.89, [[215, 135], [235, 145], [225, 165], [205, 155]]),
            _det("box-1", 2, "box", 0.96, [[350, 90], [570, 100], [560, 330], [340, 320]]),
        ],
    }


def _algorithm():
    return DetergentGraspAlgorithm(deepcopy(DEFAULT_CONFIG["algorithm"]))


def test_bottle_head_association_and_box_output_keep_legacy_fields():
    result = _algorithm().evaluate(_runtime_result())
    assert len(result.items) == 3
    by_type = {item["target_type"]: item for item in result.items}
    big = by_type["big_bottle"]
    small = by_type["small_bottle"]
    box = by_type["box"]
    assert big["center_px"] == big["grasp_point_px"]
    assert big["center_px"] != big["object_center_px"]
    assert small["center_px"] == small["grasp_point_px"]
    assert box["grasp_point_px"] is None
    for item in result.items:
        assert {"id", "class_id", "confidence", "position_camera", "angle_deg", "center_px", "type"} <= set(item)
        assert item["position_camera"] == [0.0, 0.0, 0.0]
        assert -180.0 <= item["angle_deg"] < 180.0
        assert len(item["obb_points"]) == 4
    assert result.unmatched_bottles == []
    assert result.unmatched_grasp_points == []


def test_handle_direction_uses_the_long_axis_end_farther_from_grasp_point():
    # Vertical bottle: a grasp point in the upper half means the farther end,
    # and therefore the handle direction, points downward (+90 degrees).
    down = resolve_handle_direction_angle(90.0, [100.0, 100.0], [100.0, 70.0], 100.0)
    # Moving the grasp point to the lower half flips the directed angle by 180
    # degrees, so the handle points upward (-90 degrees).
    up = resolve_handle_direction_angle(90.0, [100.0, 100.0], [100.0, 130.0], 100.0)
    assert down == 90.0
    assert up == -90.0


def test_real_scene_near_parallel_bottles_receive_opposite_360_degree_angles():
    left = resolve_handle_direction_angle(
        -86.386, [454.182, 394.978], [440.771, 410.070], 100.0
    )
    right = resolve_handle_direction_angle(
        -88.414, [525.555, 400.508], [516.577, 398.841], 100.0
    )
    assert abs(left - (-86.386)) < 1e-6
    assert abs(right - 91.586) < 1e-6
    assert abs(abs(right - left) - 177.972) < 1e-3


def test_unmatched_bottle_is_not_sent_when_grasp_point_is_required():
    payload = _runtime_result()
    payload["detections"] = [item for item in payload["detections"] if item["id"] != "head-2"]
    result = _algorithm().evaluate(payload)
    assert {item["target_type"] for item in result.items} == {"big_bottle", "box"}
    assert len(result.unmatched_bottles) == 1
    assert result.unmatched_bottles[0]["semantic"] == "small_bottle"


def test_class_name_takes_precedence_over_reordered_class_id():
    payload = _runtime_result()
    payload["detections"][0]["class_id"] = 99
    result = _algorithm().evaluate(payload)
    big = next(item for item in result.items if item["target_type"] == "big_bottle")
    assert big["class_id"] == 99


def test_config_ports_and_fps_contract_are_independent():
    config = load_config()
    assert config["runtime"]["url"].endswith(":28087")
    assert config["app"]["listen_port"] == 19212
    assert config["collector"]["listen_port"] == 18096
    assert config["websocket"]["listen_port"] == 9001
    assert config["collector"]["production_inference_source"] == "app"
    assert "detection_hz" not in config["websocket"]
    assert config["app"]["default_production_inference_fps"] == 15.0


def test_service_builds_app_decision_and_robot_message(monkeypatch, tmp_path):
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import DetergentGraspVisionService

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    service = DetergentGraspVisionService(config)
    monkeypatch.setattr(service, "_require_camera_ready", lambda: {"connected": True})
    monkeypatch.setattr(service.runtime, "infer_once", _runtime_result)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)

    decision = service.evaluate_once("request-1", "detergent_grasp")
    assert decision["message_type"] == "app_decision"
    assert decision["status"] == "ok"
    robot = decision["robot_message"]
    assert robot["request_id"] == "request-1"
    assert robot["trigger_task_id"] == "detergent_grasp"
    assert robot["fault_code"] == 0
    assert robot["coordinate_frame"] == "image"
    assert len(robot["items"]) == 3
    assert decision["visualization_result"]["message_type"] == "inference_result"
    assert decision["visualization_result"]["detergent_grasp"]["robot_items"] == robot["items"]
    assert decision["producer"]["push_mode"] == "every_completed_result"


def test_production_fps_persists_without_websocket_push_hz(tmp_path):
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import DetergentGraspVisionService

    config = load_config()
    path = tmp_path / "fps.json"
    config["app"]["inference_settings_path"] = str(path)
    service = DetergentGraspVisionService(config)
    settings = service.set_production_fps(12.5)
    assert settings["production_inference_fps"] == 12.5
    assert settings["push_mode"] == "every_completed_result"
    service2 = DetergentGraspVisionService(config)
    assert service2.production_fps() == 12.5
