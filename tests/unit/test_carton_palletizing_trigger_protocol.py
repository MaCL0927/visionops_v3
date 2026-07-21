"""Trigger-mode robot communication tests for carton palletizing."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from production.carton_palletizing.config import load_config
from production.carton_palletizing.tasks.first_layer_placement.service import FirstLayerPlacementService
from production.carton_palletizing.tasks.first_layer_placement.trigger_protocol import (
    select_held_box,
    select_top_surface_targets,
)


def obb_detection(det_id, class_id, class_name, score, box, angle_deg=0.0):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    local = [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]
    points = [[x * cosine - y * sine + cx, x * sine + y * cosine + cy] for x, y in local]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return {
        "id": det_id,
        "class_id": class_id,
        "class_name": class_name,
        "score": score,
        "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
        "center_xy": [cx, cy],
        "obb": {"cx": cx, "cy": cy, "w": width, "h": height, "angle_deg": angle_deg, "points": points},
    }


def runtime_result(detections, frame_id=1):
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "frame_id": frame_id,
        "capture_timestamp_ms": 1700000000123,
        "result_id": "result-{}".format(frame_id),
        "task_type": "obb",
        "image": {"width": 640, "height": 480},
        "detections": deepcopy(detections),
    }


TRAY = obb_detection("tray", 1, "tray", 0.98, [200, 80, 440, 400])
INSIDE_BOX = obb_detection("inside", 0, "box", 0.91, [230, 160, 330, 250], 12.0)
HELD_BOX = obb_detection("held", 0, "box", 0.95, [460, 110, 580, 230], -18.0)


class FakeRuntime:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def infer_once(self):
        self.calls += 1
        return deepcopy(self.result)

    def status(self):
        return {"model": "carton_palletizing_obb"}


class FakeBridge:
    def __init__(self, depth):
        self.depth = depth

    def get_depth(self):
        return self.depth.copy(), {"available": True, "camera_connected": True, "last_depth_age_ms": 10}

    def deproject(self, points):
        return [[float(p[0]), float(p[1]), float(p[2])] for p in points]


def _depth_with_boxes():
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    # Inside pallet carton is lower/farther; robot-held carton is highest/closest.
    depth[195:216, 270:291] = 900
    depth[159:182, 509:532] = 620
    return depth


def test_config_switches_held_box_strategy_without_code_change():
    config = load_config()
    selection = config["task"]["communication"]["held_box_selection"]
    assert selection["mode"] == "nearest_depth"
    selection["mode"] = "outside_tray"
    assert selection["mode"] == "outside_tray"


def test_nearest_depth_selects_highest_carton():
    config = load_config()
    algorithm = FirstLayerPlacementService(config).algorithm
    snapshot = algorithm.detection_candidates(runtime_result([TRAY, INSIDE_BOX, HELD_BOX]))
    selected, diagnostics = select_held_box(
        snapshot["boxes"],
        snapshot["tray_polygon"],
        _depth_with_boxes(),
        640,
        480,
        config["task"]["communication"]["held_box_selection"],
    )
    assert selected is not None
    assert selected["id"] == "held"
    assert diagnostics["reason"] == "SELECTED_NEAREST_DEPTH"


def test_outside_tray_selects_box_outside_pallet_region():
    config = load_config()
    config["task"]["communication"]["held_box_selection"]["mode"] = "outside_tray"
    algorithm = FirstLayerPlacementService(config).algorithm
    snapshot = algorithm.detection_candidates(runtime_result([TRAY, INSIDE_BOX, HELD_BOX]))
    selected, diagnostics = select_held_box(
        snapshot["boxes"],
        snapshot["tray_polygon"],
        _depth_with_boxes(),
        640,
        480,
        config["task"]["communication"]["held_box_selection"],
    )
    assert selected is not None
    assert selected["id"] == "held"
    assert diagnostics["reason"] == "SELECTED_OUTSIDE_TRAY"


def test_place_target_trigger_returns_detected_tray_when_no_box():
    config = load_config()
    service = FirstLayerPlacementService(config)
    runtime = FakeRuntime(runtime_result([TRAY], 11))
    service.runtime = runtime
    service.depth_bridge = FakeBridge(np.full((480, 640), 1000, dtype=np.uint16))
    decision = service.trigger_once("pallet_place_target")
    message = decision["robot_message"]
    assert message["trigger_task_id"] == "pallet_place_target"
    assert message["fault_code"] == 0
    assert len(message["items"]) == 1
    item = message["items"][0]
    assert item["class_id"] == 1
    assert item["position_camera"][2] == 1000.0
    assert -90.0 <= item["angle_deg"] <= 90.0
    assert "slot_id" not in item
    assert "layer" not in item
    assert runtime.calls == 1


def test_place_target_ignores_boxes_outside_tray_and_returns_empty_tray():
    config = load_config()
    outside = obb_detection("outside", 0, "box", 0.99, [470, 100, 580, 220], 0.0)
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    depth[153:168, 518:533] = 600
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, outside], 12))
    service.depth_bridge = FakeBridge(depth)
    decision = service.trigger_once("pallet_place_target")
    items = decision["robot_message"]["items"]
    assert len(items) == 1
    assert items[0]["class_id"] == 1


def test_place_target_returns_only_top_depth_cluster_boxes():
    config = load_config()
    top_left = obb_detection("top-left", 0, "box", 0.96, [220, 110, 300, 190], 8.0)
    top_right = obb_detection("top-right", 0, "box", 0.94, [320, 110, 400, 190], -6.0)
    lower = obb_detection("lower", 0, "box", 0.97, [270, 260, 370, 350], 15.0)
    depth = np.full((480, 640), 1100, dtype=np.uint16)
    depth[143:158, 253:268] = 610
    depth[143:158, 353:368] = 640
    depth[298:313, 313:328] = 860

    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, lower, top_right, top_left], 13))
    service.depth_bridge = FakeBridge(depth)
    decision = service.trigger_once("pallet_place_target")
    items = decision["robot_message"]["items"]
    assert len(items) == 2
    assert all(item["class_id"] == 0 for item in items)
    assert [item["position_camera"][2] for item in items] == [610.0, 640.0]
    assert all("slot_id" not in item and "layer" not in item for item in items)
    assert decision["surface_target_selection"]["target_kind"] == "box"
    assert decision["surface_target_selection"]["selected_count"] == 2


def test_top_surface_selector_caps_output_at_four():
    config = load_config()
    settings = config["task"]["communication"]["surface_target_selection"]
    boxes = []
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    for index in range(5):
        x1 = 210 + index * 35
        box = obb_detection("box-{}".format(index), 0, "box", 0.90 + index * 0.01, [x1, 120, x1 + 30, 180])
        boxes.append(box)
        cx = int((x1 + x1 + 30) / 2)
        depth[143:158, cx - 7:cx + 8] = 600 + index * 5
    service = FirstLayerPlacementService(config)
    snapshot = service.algorithm.detection_candidates(runtime_result([TRAY] + boxes))
    selected, kind, diagnostics = select_top_surface_targets(
        snapshot["boxes"],
        snapshot["trays"],
        snapshot["tray_polygon"],
        depth,
        640,
        480,
        settings,
    )
    assert kind == "box"
    assert len(selected) == 4
    assert diagnostics["selected_count"] == 4


def test_held_box_trigger_returns_selected_obb_center_and_angle_without_advancing_stack():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, INSIDE_BOX, HELD_BOX], 12))
    service.depth_bridge = FakeBridge(_depth_with_boxes())
    before = service.algorithm.tray_reference()["layer"]
    decision = service.trigger_once("held_box_pose")
    after = service.algorithm.tray_reference()["layer"]
    message = decision["robot_message"]
    assert message["trigger_task_id"] == "held_box_pose"
    assert message["fault_code"] == 0
    assert len(message["items"]) == 1
    item = message["items"][0]
    assert item["source_detection_id"] == "held"
    assert item["position_camera"][2] == 620.0
    assert -90.0 <= item["angle_deg"] <= 90.0
    assert before == after == 1


def test_place_trigger_does_not_advance_slot_or_layer_state():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, INSIDE_BOX], 20))
    service.depth_bridge = FakeBridge(_depth_with_boxes())
    before = service.algorithm.tray_reference()["layer"]
    decision = service.trigger_once("pallet_place_target")
    after = service.algorithm.tray_reference()["layer"]
    assert before == after == 1
    assert decision["robot_message"]["items"]
    assert "placement" not in decision


def test_trigger_websocket_status_is_disabled_by_default():
    config = load_config()
    websocket = config["task"]["communication"]["websocket"]
    assert websocket["status_enabled"] is False
    assert websocket["status_on_connect"] is False
    service = FirstLayerPlacementService(config)
    assert service.status_enabled is False
    assert service.status_on_connect is False


def test_robot_detect_region_is_ignored_for_place_target():
    """ROI ownership stays in Runtime/Web even when robot sends config.detect_region."""
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY], 31))
    service.depth_bridge = FakeBridge(np.full((480, 640), 1000, dtype=np.uint16))

    # This region excludes the tray center (320, 240). It must be accepted for
    # protocol compatibility but never applied to the palletizing result.
    service._on_websocket_json(None, {
        "type": "config",
        "detect_region": [0, 0, 100, 100],
    })

    decision = service.trigger_once("pallet_place_target")
    items = decision["robot_message"]["items"]
    diagnostics = decision["surface_target_selection"]["diagnostics"]

    assert len(items) == 1
    assert items[0]["class_id"] == 1
    assert diagnostics["roi_control_source"] == "visionops_web_runtime"
    assert diagnostics["robot_detect_region_applied"] is False
    assert diagnostics["last_ignored_robot_detect_region"] == [0.0, 0.0, 100.0, 100.0]
    assert diagnostics["raw_candidate_tray_count"] == 1
    assert service.state.counters["remote_detect_region_ignored"] == 1


def test_robot_detect_region_is_ignored_for_held_box():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, INSIDE_BOX, HELD_BOX], 32))
    service.depth_bridge = FakeBridge(_depth_with_boxes())

    # Excludes the held box center, but held_box_pose must still use the Runtime
    # detections already filtered by the Web-configured ROI.
    service._on_websocket_json(None, {
        "type": "config",
        "detect_region": [0, 0, 100, 100],
    })

    decision = service.trigger_once("held_box_pose")
    items = decision["robot_message"]["items"]
    diagnostics = decision["held_box_selection"]

    assert len(items) == 1
    assert items[0]["source_detection_id"] == "held"
    assert diagnostics["roi_control_source"] == "visionops_web_runtime"
    assert diagnostics["robot_detect_region_applied"] is False


def test_numeric_task_id_1_triggers_place_target_and_echoes_number():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY], 41))
    service.depth_bridge = FakeBridge(np.full((480, 640), 1000, dtype=np.uint16))

    decision = service.trigger_once(1)

    assert decision["trigger_task_id"] == 1
    assert decision["robot_message"]["trigger_task_id"] == 1
    assert len(decision["robot_message"]["items"]) == 1
    assert decision["robot_message"]["items"][0]["class_id"] == 1


def test_string_task_id_1_triggers_place_target_and_echoes_string():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY], 42))
    service.depth_bridge = FakeBridge(np.full((480, 640), 1000, dtype=np.uint16))

    decision = service.trigger_once("1")

    assert decision["trigger_task_id"] == "1"
    assert decision["robot_message"]["trigger_task_id"] == "1"
    assert decision["robot_message"]["items"][0]["class_id"] == 1


def test_numeric_task_id_2_triggers_held_box_and_echoes_number():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, INSIDE_BOX, HELD_BOX], 43))
    service.depth_bridge = FakeBridge(_depth_with_boxes())

    decision = service.trigger_once(2)

    assert decision["trigger_task_id"] == 2
    assert decision["robot_message"]["trigger_task_id"] == 2
    assert len(decision["robot_message"]["items"]) == 1
    assert decision["robot_message"]["items"][0]["source_detection_id"] == "held"


def test_string_task_id_2_triggers_held_box_and_echoes_string():
    config = load_config()
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY, INSIDE_BOX, HELD_BOX], 44))
    service.depth_bridge = FakeBridge(_depth_with_boxes())

    decision = service.trigger_once("2")

    assert decision["trigger_task_id"] == "2"
    assert decision["robot_message"]["trigger_task_id"] == "2"
    assert decision["robot_message"]["items"][0]["source_detection_id"] == "held"


def test_trigger_aliases_are_configurable_without_code_change():
    config = load_config()
    trigger_tasks = config["task"]["communication"]["trigger_tasks"]
    trigger_tasks["place_target_aliases"] = [101]
    trigger_tasks["held_box_aliases"] = [202]
    service = FirstLayerPlacementService(config)
    service.runtime = FakeRuntime(runtime_result([TRAY], 45))
    service.depth_bridge = FakeBridge(np.full((480, 640), 1000, dtype=np.uint16))

    assert service.trigger_once(101)["robot_message"]["items"][0]["class_id"] == 1
    error = service.trigger_once(1)
    assert error["status"] == "error"
    assert error["robot_message"]["fault_code"] == 3201


def test_websocket_trigger_queue_preserves_numeric_task_id():
    config = load_config()
    service = FirstLayerPlacementService(config)

    service._on_websocket_json(None, {"type": "trigger", "task_id": 1})
    request = service.trigger_queue.get_nowait()

    assert request.task_id == 1
    service.trigger_queue.task_done()
