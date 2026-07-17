"""Segmentation carton contour/corner/grasp-point tests."""
from __future__ import annotations

import numpy as np

from production.carton_palletizing.config import load_config
from production.carton_palletizing.tasks.box_grasp_vision.algorithm import BoxGraspAlgorithm


def runtime_result(polygon, source="proto", score=0.96):
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "segmentation",
        "frame_id": "frame-1",
        "result_id": "result-1",
        "image": {"width": 640, "height": 480},
        "detections": [
            {
                "id": "seg-1",
                "class_id": 0,
                "class_name": "box",
                "score": score,
                "bbox_xyxy": [220, 210, 460, 340],
                "center_xy": [340, 275],
                "mask": {
                    "encoding": "polygon",
                    "source": source,
                    "size": [480, 640],
                    "polygon": [polygon],
                },
            }
        ],
    }


def algorithm():
    return BoxGraspAlgorithm(load_config()["box_grasp"]["algorithm"])


def test_perspective_mask_produces_ordered_corners_center_and_grasp_midpoints():
    polygon = [
        [235, 221], [300, 216], [426, 214], [440, 260], [457, 329],
        [360, 332], [222, 336], [225, 290], [228, 250],
    ]
    classified = algorithm().classify(runtime_result(polygon))
    assert len(classified.items) == 1
    item = classified.items[0]
    tl, tr, br, bl = item["quad"]
    assert tl[0] < tr[0] and tl[1] < bl[1]
    assert br[0] > bl[0] and br[1] > tr[1]
    left_mid = item["points"]["left_mid"]
    right_mid = item["points"]["right_mid"]
    center = item["points"]["center"]
    assert left_mid[0] < center[0] < right_mid[0]
    assert abs(center[0] - (left_mid[0] + right_mid[0]) / 2.0) < 1e-6
    assert item["quality"]["quad_to_contour_area_ratio"] > 0.65


def test_bbox_fallback_is_rejected():
    polygon = [[220, 210], [460, 210], [460, 340], [220, 340]]
    classified = algorithm().classify(runtime_result(polygon, source="bbox_fallback"))
    assert classified.items == []
    assert classified.ignored[0]["reason"] == "bbox_fallback_mask"


def test_depth_sampling_and_external_contract_for_seven_points():
    polygon = [[235, 221], [426, 214], [457, 329], [222, 336]]
    processor = algorithm()
    item = processor.classify(runtime_result(polygon)).items[0]
    depth = np.full((480, 640), 930, dtype=np.uint16)
    depth_info = processor.sample_item_depth(item, depth, 640, 480)
    assert set(depth_info) == set(processor.POINT_ORDER)
    assert all(value["depth_valid"] for value in depth_info.values())
    deproject = processor.build_deproject_input(item, depth_info)
    assert len(deproject) == 7
    positions = [[float(index), float(index + 1), 930.0] for index in range(7)]
    result = processor.build_external_item(0, item, depth_info, positions)
    assert list(result["corners_px"]) == ["top_left", "top_right", "bottom_right", "bottom_left"]
    assert set(result["grasp_points_px"]) == {"left_mid", "right_mid"}
    assert result["center_camera"] == [4.0, 5.0, 930.0]
    assert result["grasp_points_camera"]["left_mid"] == [5.0, 6.0, 930.0]
    assert result["grasp_points_camera"]["right_mid"] == [6.0, 7.0, 930.0]


def test_config_keeps_stack_and_box_grasp_ports_separate():
    config = load_config()
    ports = {
        28084,
        config["app"]["listen_port"],
        config["collector"]["listen_port"],
        28085,
        config["box_grasp"]["app"]["listen_port"],
        config["box_grasp"]["collector"]["listen_port"],
        config["box_grasp"]["websocket"]["listen_port"],
    }
    assert len(ports) == 7
    assert config["box_grasp"]["runtime"]["accepted_task_types"] == ["segmentation", "segment"]
    assert config["box_grasp"]["websocket"]["listen_port"] == 9001


def test_service_builds_collector_visualization_and_robot_message(monkeypatch):
    from production.carton_palletizing.tasks.box_grasp_vision.service import BoxGraspVisionService

    config = load_config()
    service = BoxGraspVisionService(config)
    polygon = [[235, 221], [426, 214], [457, 329], [222, 336]]
    payload = runtime_result(polygon)
    depth = np.full((480, 640), 930, dtype=np.uint16)

    monkeypatch.setattr(service.bridge, "require_ready", lambda need_depth: {"camera_connected": True})
    monkeypatch.setattr(service.runtime, "infer_once", lambda: payload)
    monkeypatch.setattr(service.bridge, "depth", lambda health: (depth, b"depth", health))
    monkeypatch.setattr(
        service.bridge,
        "deproject",
        lambda points: ([[float(index), float(index + 1), float(point[2])] for index, point in enumerate(points)], {"ok": True}),
    )
    monkeypatch.setattr(service, "_save_debug_async", lambda document, depth_bytes: None)

    decision = service.evaluate_once("request-1")
    assert decision["message_type"] == "app_decision"
    assert decision["status"] == "ok"
    assert decision["visualization_result"]["message_type"] == "inference_result"
    assert len(decision["visualization_result"]["box_grasp"]["items"]) == 1
    robot = decision["robot_message"]
    assert robot["request_id"] == "request-1"
    assert robot["fault_code"] == 0
    assert set(robot) == {
        "type", "request_id", "frame_id", "timestamp", "items", "fault_code", "fault_type"
    }
    assert len(robot["items"]) == 2
    first_point, second_point = robot["items"]
    assert set(first_point) == {
        "id",
        "class_id",
        "confidence",
        "position_camera",
        "center_px",
    }
    assert set(second_point) == set(first_point)
    assert first_point["id"] == second_point["id"] == 0
    assert first_point["class_id"] == second_point["class_id"] == 0
    assert first_point["confidence"] == second_point["confidence"] == 0.96
    assert first_point["center_px"][0] < second_point["center_px"][0]
    assert first_point["position_camera"] == [5.0, 6.0, 930.0]
    assert second_point["position_camera"] == [6.0, 7.0, 930.0]
    # Full contour/corner/depth data remains available only for Collector/debug visualization.
    visual_item = decision["visualization_result"]["box_grasp"]["items"][0]
    assert visual_item["grasp_points_px"]["left_mid"]
    assert visual_item["contour_px"]
