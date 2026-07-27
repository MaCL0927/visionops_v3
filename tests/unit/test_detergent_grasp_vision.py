"""Detergent-grasp OBB parsing, association and protocol tests."""
from __future__ import annotations

from copy import deepcopy
import json

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


def _raw_runtime_response(payload=None, transport="raw_socket"):
    from production.common.runtime_ipc import TimedHttpResponse

    document = payload if payload is not None else _runtime_result()
    body = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return TimedHttpResponse(
        body=body,
        status_code=200,
        headers={
            "content-length": str(len(body)),
            "x-visionops-http-queue-ms": "0.08",
            "x-visionops-http-route-ms": "39.1",
        },
        connect_ms=0.36,
        send_ms=0.05,
        headers_wait_ms=39.5,
        body_read_ms=0.1,
        total_ms=40.01,
        transport=transport,
    )


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
        if item["target_type"] == "box":
            assert -90.0 <= item["angle_deg"] <= 90.0
        else:
            assert 0.0 <= item["angle_deg"] < 360.0
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
    assert up == 270.0


def test_real_scene_near_parallel_bottles_receive_opposite_360_degree_angles():
    left = resolve_handle_direction_angle(
        -86.386, [454.182, 394.978], [440.771, 410.070], 100.0
    )
    right = resolve_handle_direction_angle(
        -88.414, [525.555, 400.508], [516.577, 398.841], 100.0
    )
    assert abs(left - 273.614) < 1e-6
    assert abs(right - 91.586) < 1e-6
    separation = abs(right - left)
    circular_separation = min(separation, 360.0 - separation)
    assert abs(circular_separation - 177.972) < 1e-3


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
    monkeypatch.setattr(service.runtime, "infer_once_raw", _raw_runtime_response)
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


def test_pipeline_defaults_are_enabled_and_latest_only():
    config = load_config()
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


def test_service_exposes_full_app_timing_and_percentiles(monkeypatch, tmp_path):
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import DetergentGraspVisionService

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    config["websocket"]["listen_port"] = 0
    service = DetergentGraspVisionService(config)
    monkeypatch.setattr(service.runtime, "infer_once_raw", _raw_runtime_response)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)

    for index in range(3):
        decision = service.evaluate_once(f"timing-{index}", "detergent_grasp")
        timing = decision["app_timing"]
        for key in (
            "runtime_lock_wait_ms",
            "runtime_request_ms",
            "runtime_connect_ms",
            "runtime_send_ms",
            "runtime_headers_wait_ms",
            "runtime_body_read_ms",
            "runtime_json_decode_ms",
            "runtime_server_queue_ms",
            "runtime_server_route_ms",
            "runtime_internal_ms",
            "inference_stage_ms",
            "result_queue_wait_ms",
            "algorithm_ms",
            "robot_message_build_ms",
            "visualization_build_ms",
            "decision_build_ms",
            "postprocess_stage_ms",
            "state_store_ms",
            "pipeline_age_ms",
            "total_ms",
        ):
            assert key in timing
            assert timing[key] >= 0

    snapshot = service.state.snapshot(service.websocket)
    assert snapshot["latency_ms"]["samples"] == 3
    assert snapshot["latency_ms"]["p50"] >= 0
    assert snapshot["latency_ms"]["p95"] >= snapshot["latency_ms"]["p50"]
    assert snapshot["app_timing_stats"]["runtime_request_ms"]["samples"] == 3
    assert snapshot["app_timing_stats"]["algorithm_ms"]["p95"] >= 0


def test_raw_runtime_bytes_are_decoded_only_in_postprocess(monkeypatch, tmp_path):
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import (
        DetergentGraspVisionService,
    )

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    config["websocket"]["listen_port"] = 0
    service = DetergentGraspVisionService(config)
    monkeypatch.setattr(service.runtime, "infer_once_raw", _raw_runtime_response)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)

    packet = service._run_inference_stage(
        request_id="raw-1",
        trigger_task_id="detergent_grasp",
        continuous=False,
    )
    assert packet.runtime_result is None
    assert packet.runtime_raw.startswith(b"{")
    assert packet.runtime_transport == "raw_socket"
    assert packet.runtime_json_decode_ms == 0.0

    decision = service._complete_packet(packet, dispatch=False)
    timing = decision["app_timing"]
    assert packet.runtime_result is not None
    assert timing["runtime_transport"] == "raw_socket"
    assert timing["runtime_response_bytes"] == len(packet.runtime_raw)
    assert timing["runtime_connect_ms"] == 0.36
    assert timing["runtime_server_route_ms"] == 39.1
    assert timing["runtime_json_decode_ms"] >= 0.0
    assert decision["status"] == "ok"


def test_latest_only_queue_never_replaces_explicit_trigger(tmp_path):
    import queue
    import time
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import (
        DetergentGraspVisionService,
        InferencePacket,
        TriggerRequest,
    )

    class DummySession:
        def send_json(self, _document):
            return None

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    service = DetergentGraspVisionService(config)
    service.result_queue = queue.Queue(maxsize=1)

    def packet(frame_id, trigger=None):
        return InferencePacket(
            frame_id=frame_id,
            request_id=str(frame_id),
            trigger_task_id="detergent_grasp" if trigger else None,
            started_timestamp=time.time(),
            started_monotonic=time.monotonic(),
            trigger=trigger,
            continuous=trigger is None,
        )

    continuous_old = packet(1)
    continuous_new = packet(2)
    service.result_queue.put_nowait(continuous_old)
    service._enqueue_packet(continuous_new)
    assert service.result_queue.get_nowait().frame_id == 2
    assert service.state.counters["pipeline_results_dropped"] == 1

    trigger = TriggerRequest(DummySession(), "trigger-1", "detergent_grasp")
    trigger_packet = packet(3, trigger)
    service.result_queue.put_nowait(trigger_packet)
    service._enqueue_packet(packet(4))
    retained = service.result_queue.get_nowait()
    assert retained.trigger is trigger
    assert retained.frame_id == 3
    assert service.state.counters["pipeline_continuous_dropped_for_trigger"] == 1


def test_inference_and_postprocess_threads_overlap(monkeypatch, tmp_path):
    import threading
    import time
    from production.detergent_grasp.tasks.detergent_grasp_vision.service import DetergentGraspVisionService

    config = load_config()
    config["app"]["inference_settings_path"] = str(tmp_path / "fps.json")
    config["app"]["default_production_inference_fps"] = 30.0
    config["websocket"]["listen_port"] = 0
    config["websocket"]["status_interval_s"] = 60.0
    service = DetergentGraspVisionService(config)

    second_inference_started = threading.Event()
    algorithm_entered = threading.Event()
    calls = {"runtime": 0, "algorithm": 0}
    original_evaluate = service.algorithm.evaluate

    def fake_infer_raw():
        calls["runtime"] += 1
        if calls["runtime"] >= 2:
            second_inference_started.set()
        return _raw_runtime_response()

    def fake_algorithm(payload):
        calls["algorithm"] += 1
        if calls["algorithm"] == 1:
            algorithm_entered.set()
            assert second_inference_started.wait(0.8), "second Runtime request did not overlap postprocess"
        return original_evaluate(payload)

    monkeypatch.setattr(service.runtime, "infer_once_raw", fake_infer_raw)
    monkeypatch.setattr(service.algorithm, "evaluate", fake_algorithm)
    monkeypatch.setattr(service, "_save_debug_async", lambda decision: None)

    service.start()
    try:
        assert algorithm_entered.wait(0.5)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and service.state.counters["inference_success"] < 1:
            time.sleep(0.01)
        assert service.state.counters["inference_success"] >= 1
        assert second_inference_started.is_set()
        pipeline = service.pipeline_status()
        assert pipeline["enabled"] is True
        assert pipeline["inference_thread_alive"] is True
        assert pipeline["postprocess_thread_alive"] is True
    finally:
        service.stop()
