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


def test_perspective_mask_produces_ordered_corners_center_and_inward_grasp_points():
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



def test_grasp_points_move_inward_and_depth_samples_move_farther_inside():
    polygon = [[235, 221], [426, 214], [457, 329], [222, 336]]
    processor = algorithm()
    item = processor.classify(runtime_result(polygon)).items[0]

    left_edge = item["edge_midpoints"]["left_mid"]
    right_edge = item["edge_midpoints"]["right_mid"]
    left_grasp = item["points"]["left_mid"]
    right_grasp = item["points"]["right_mid"]
    center = item["points"]["center"]
    left_sample = item["depth_sample_points"]["left_mid"]
    right_sample = item["depth_sample_points"]["right_mid"]

    # Left moves right/inward and right moves left/inward.
    assert left_edge[0] < left_grasp[0] < center[0]
    assert center[0] < right_grasp[0] < right_edge[0]
    # The depth sampling coordinates are slightly farther inside than the
    # robot-facing points, but projection still uses the robot-facing points.
    assert left_grasp[0] < left_sample[0] < center[0]
    assert center[0] < right_sample[0] < right_grasp[0]
    assert item["grasp_inward_ratio"] == 0.18
    assert item["grasp_depth_extra_inward_ratio"] == 0.05


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
    assert config["box_grasp"]["algorithm"]["geometry"]["grasp_inward_ratio"] == 0.18
    assert config["box_grasp"]["algorithm"]["depth"]["grasp_extra_inward_ratio"] == 0.05


def test_service_builds_collector_visualization_and_robot_message(monkeypatch):
    from production.carton_palletizing.tasks.box_grasp_vision.service import BoxGraspVisionService

    config = load_config()
    service = BoxGraspVisionService(config)
    polygon = [[235, 221], [426, 214], [457, 329], [222, 336]]
    payload = runtime_result(polygon)
    from production.carton_palletizing.tasks.box_grasp_vision.service import HttpBytesResult

    raw_payload = __import__("json").dumps(payload, separators=(",", ":")).encode("utf-8")
    monkeypatch.setattr(
        service.runtime,
        "infer_once_raw",
        lambda: HttpBytesResult(
            body=raw_payload,
            status_code=200,
            headers={
                "x-visionops-http-queue-ms": "0.2",
                "x-visionops-http-route-ms": "5.0",
            },
            headers_wait_ms=5.2,
            body_read_ms=0.1,
            total_ms=5.3,
        ),
    )
    captured = {}

    def sample_deproject(points, *args):
        captured["points"] = points
        captured["args"] = args
        return (
            [
                {
                    "depth_valid": True,
                    "depth_mm": 930,
                    "sample_px": [int(round(point[0])), int(round(point[1]))],
                    "valid_pixels": 81,
                    "position_camera": [float(index), float(index + 1), 930.0],
                    "project_valid": True,
                }
                for index, point in enumerate(points)
            ],
            {"ok": True, "depth_age_ms": 12, "depth_sequence": 8, "sample_ms": 0.3},
        )

    monkeypatch.setattr(
        service.bridge,
        "sample_deproject",
        sample_deproject,
    )
    monkeypatch.setattr(service.bridge, "depth", lambda *_args: (_ for _ in ()).throw(AssertionError("legacy depth PNG must not be used")))
    monkeypatch.setattr(service.bridge, "deproject", lambda *_args: (_ for _ in ()).throw(AssertionError("legacy deproject must not be used")))
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
    assert decision["app_timing"]["depth_sample_deproject_ms"] >= 0
    assert decision["app_timing"]["runtime_response_bytes"] == len(raw_payload)
    assert decision["app_timing"]["runtime_server_queue_ms"] == 0.2
    assert decision["app_timing"]["runtime_server_route_ms"] == 5.0
    assert decision["app_timing"]["runtime_json_decode_ms"] >= 0
    assert decision["visualization_result"]["box_grasp"]["app_timing"] == decision["app_timing"]
    assert len(captured["points"]) == 7
    assert all(len(point) == 4 for point in captured["points"])
    # The first pair samples inward from the corner but deprojects the original corner.
    assert captured["points"][0][0:2] != captured["points"][0][2:4]
    # Grasp points are already moved away from the mask boundary, and their
    # sampling points move a little farther toward the carton centre.
    left_grasp = visual_item["grasp_points_px"]["left_mid"]
    right_grasp = visual_item["grasp_points_px"]["right_mid"]
    edge = visual_item["grasp_geometry"]["edge_midpoints_px"]
    assert edge["left_mid"][0] < left_grasp[0]
    assert right_grasp[0] < edge["right_mid"][0]
    assert captured["points"][5][0] > captured["points"][5][2]
    assert captured["points"][6][0] < captured["points"][6][2]


def test_box_grasp_pipeline_defaults_to_latest_result_queue():
    from production.carton_palletizing.tasks.box_grasp_vision.service import BoxGraspVisionService

    config = load_config()
    service = BoxGraspVisionService(config)
    status = service.pipeline_status()

    assert status["enabled"] is True
    assert status["result_queue_capacity"] == 1
    assert status["max_result_age_ms"] == 500
    assert config["camera_bridge"]["sample_deproject_path"] == "/api/coordinate/sample_deproject"
    assert config["box_grasp"]["algorithm"]["depth"]["use_sample_deproject"] is True


def test_background_inference_fps_can_be_updated_and_persisted(tmp_path):
    from production.carton_palletizing.tasks.box_grasp_vision.service import BoxGraspVisionService

    config = load_config()
    settings_path = tmp_path / "box_grasp_inference_settings.json"
    config["box_grasp"]["app"]["inference_settings_path"] = str(settings_path)
    service = BoxGraspVisionService(config)

    result = service.set_detection_hz(12.5)

    assert result["status"] == "ok"
    assert result["detection_fps"] == 12.5
    assert service.inference_settings()["detection_fps"] == 12.5
    persisted = __import__("json").loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["detection_fps"] == 12.5


def test_background_inference_fps_override_is_loaded_on_start(tmp_path):
    from production.carton_palletizing.tasks.box_grasp_vision.service import BoxGraspVisionService

    settings_path = tmp_path / "box_grasp_inference_settings.json"
    settings_path.write_text('{"detection_fps":7.5}', encoding="utf-8")
    config = load_config()
    config["box_grasp"]["app"]["inference_settings_path"] = str(settings_path)

    service = BoxGraspVisionService(config)

    assert service.detection_hz() == 7.5
