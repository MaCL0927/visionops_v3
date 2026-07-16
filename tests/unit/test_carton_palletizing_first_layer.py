"""First-layer carton palletizing OBB business logic tests."""

from __future__ import annotations

import math
from copy import deepcopy

from production.carton_palletizing.config import load_config
from production.carton_palletizing.tasks.first_layer_placement.algorithm import FirstLayerPlacementAlgorithm


IMAGE = {"width": 1660, "height": 934}


def obb_detection(det_id, class_id, class_name, score, box, angle_deg=0.0):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    local = [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]
    points = [
        [x * cosine - y * sine + cx, x * sine + y * cosine + cy]
        for x, y in local
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "id": det_id,
        "class_id": class_id,
        "class_name": class_name,
        "score": score,
        "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
        "center_xy": [cx, cy],
        "obb": {
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "angle_deg": angle_deg,
            "points": points,
        },
    }


TRAY = obb_detection("tray-1", 1, "tray", 0.98, [655, 51, 1076, 532])
BOXES = [
    obb_detection("box-1", 0, "box", 0.95, [658, 84, 926, 225]),
    obb_detection("box-2", 0, "box", 0.95, [932, 84, 1073, 352]),
    obb_detection("box-3", 0, "box", 0.95, [658, 231, 799, 499]),
    obb_detection("box-4", 0, "box", 0.95, [805, 358, 1073, 499]),
]


def result(detections, frame):
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "frame_id": "frame-{}".format(frame),
        "result_id": "result-{}".format(frame),
        "task_type": "obb",
        "image": dict(IMAGE),
        "detections": deepcopy(detections),
    }


def algorithm():
    return FirstLayerPlacementAlgorithm(load_config()["task"]["algorithm"])


def test_empty_tray_generates_four_visible_slots_from_obb():
    placement = algorithm().evaluate(result([TRAY], 1))
    assert placement["state"] == "LAYER_1_FILLING"
    assert placement["occupied_count"] == 0
    assert placement["next_slot_id"] == "P3"
    assert [slot["slot_id"] for slot in placement["slots"]] == ["P1", "P2", "P3", "P4"]
    assert all(slot["visible_mask"] for slot in placement["slots"])
    assert placement["slots"][0]["orientation_label"] == "横向"
    assert placement["slots"][1]["orientation_label"] == "竖向"
    assert len(placement["tray"]["obb_points"]) == 4


def test_tray_lock_survives_occlusion_and_one_box_hides_only_one_slot():
    tracker = algorithm()
    tracker.evaluate(result([TRAY], 1))
    first = tracker.evaluate(result([BOXES[2]], 2))
    assert first["tray"]["source"] == "locked"
    assert first["occupied_count"] == 0
    states = {slot["slot_id"]: slot for slot in first["slots"]}
    assert states["P3"]["state"] == "VERIFYING"

    confirmed = tracker.evaluate(result([BOXES[2]], 3))
    assert confirmed["occupied_count"] == 1
    assert confirmed["next_slot_id"] == "P1"
    states = {slot["slot_id"]: slot for slot in confirmed["slots"]}
    assert states["P3"]["occupied"] is True
    assert states["P3"]["visible_mask"] is False
    assert all(states[slot_id]["visible_mask"] for slot_id in ("P1", "P2", "P4"))


def test_four_obb_boxes_complete_first_layer_after_temporal_confirmation():
    tracker = algorithm()
    tracker.evaluate(result([TRAY], 1))
    tracker.evaluate(result(BOXES, 2))
    placement = tracker.evaluate(result(BOXES, 3))
    assert placement["state"] == "LAYER_1_COMPLETE"
    assert placement["complete"] is True
    assert placement["occupied_count"] == 4
    assert placement["next_slot_id"] is None
    assert all(slot["visible_mask"] is False for slot in placement["slots"])


def test_non_obb_detections_are_rejected_when_obb_is_required():
    tracker = algorithm()
    axis_only = deepcopy(TRAY)
    axis_only.pop("obb")
    placement = tracker.evaluate(result([axis_only], 1))
    assert placement["state"] == "WAIT_TRAY"
    assert placement["rejected_non_obb_count"] == 1


def test_rotated_tray_rotates_slot_geometry():
    tracker = algorithm()
    rotated_tray = obb_detection("tray-r", 1, "tray", 0.98, [655, 51, 1076, 532], angle_deg=8.0)
    placement = tracker.evaluate(result([rotated_tray], 1))
    assert placement["state"] == "LAYER_1_FILLING"
    assert 5.0 < placement["tray"]["angle_deg"] < 11.0
    assert 5.0 < placement["slots"][0]["orientation_deg"] < 11.0
    assert 95.0 < placement["slots"][1]["orientation_deg"] < 101.0



def test_rectangular_tray_uses_short_edge_for_centered_square_footprint():
    placement = algorithm().evaluate(result([TRAY], 1))
    footprint = placement["footprint"]
    bounds = footprint["normalized_bounds"]
    assert bounds["u_min"] == 0.0
    assert bounds["u_max"] == 1.0
    assert 0.05 < bounds["v_min"] < 0.10
    assert 0.90 < bounds["v_max"] < 0.95
    assert abs(footprint["square_side_px"] - footprint["tray_width_px"]) < 1.0
    points = footprint["obb_points"]
    width = math.dist(points[0], points[1])
    height = math.dist(points[0], points[3])
    assert abs(width - height) < 1.0


def test_slot_order_starts_at_bottom_left_and_moves_clockwise():
    placement = algorithm().evaluate(result([TRAY], 1))
    assert placement["next_slot_id"] == "P3"
    slots = {slot["slot_id"]: slot for slot in placement["slots"]}
    assert slots["P3"]["center_xy"][0] < slots["P4"]["center_xy"][0]
    assert slots["P3"]["center_xy"][1] > slots["P1"]["center_xy"][1]

def test_reset_clears_locked_tray_and_sticky_occupancy():
    tracker = algorithm()
    tracker.evaluate(result([TRAY], 1))
    tracker.evaluate(result([BOXES[2]], 2))
    tracker.evaluate(result([BOXES[2]], 3))
    tracker.reset()
    placement = tracker.evaluate(result([], 4))
    assert placement["state"] == "WAIT_TRAY"
    assert placement["occupied_count"] == 0
    assert placement["slots"] == []
