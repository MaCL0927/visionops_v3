"""标准推理结果到 Gateway 消息的映射测试。"""

from __future__ import annotations

from copy import deepcopy

from edge.gateway_adapter.result_to_gateway import inference_result_to_gateway_message


def detection_result() -> dict:
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "device_id": "example-edge-001",
        "component": "rknn_runtime",
        "timestamp_ms": 1,
        "trace_id": "trace-001",
        "frame_id": "frame-mock-00000123",
        "source": "runtime:mock",
        "status": "ok",
        "result_id": "result-mock-00000456",
        "task_type": "detection",
        "model": {},
        "image": {"width": 1920, "height": 1080},
        "timing": {
            "preprocess_ms": 2.2,
            "inference_ms": 12.6,
            "postprocess_ms": 1.1,
            "total_ms": 15.9,
        },
        "detections": [
            {
                "id": "det-low",
                "class_id": 0,
                "class_name": "object",
                "score": 0.5,
                "bbox_xyxy": [10, 20, 110, 220],
            },
            {
                "id": "det-best",
                "class_id": 0,
                "class_name": "object",
                "score": 0.9346,
                "bbox_xyxy": [100.4, 200.4, 500.6, 600.6],
                "center_xy": [301.2, 401.8],
            },
        ],
    }


def test_detection_result_maps_geometry_timing_and_scores() -> None:
    message = inference_result_to_gateway_message(
        detection_result(), "generic_mock", sequence=7, heartbeat=1
    )
    payload = message["payload"]
    assert message["frame_id"] == "frame-mock-00000123"
    assert message["result_id"] == "result-mock-00000456"
    assert message["final_code"] == 1
    assert message["final_label"] == "NG_OR_DETECTED"
    assert message["ok"] is True
    assert payload["object_count"] == 2
    assert payload["score_x1000"] == 935
    assert payload["center_x"] == 301
    assert payload["center_y"] == 402
    assert payload["bbox_x1"] == 100
    assert payload["bbox_y1"] == 200
    assert payload["bbox_x2"] == 501
    assert payload["bbox_y2"] == 601
    assert payload["inference_ms"] == 13
    assert payload["total_ms"] == 16


def test_no_detections_maps_to_ok() -> None:
    result = detection_result()
    result["detections"] = []
    message = inference_result_to_gateway_message(result, "generic_mock", 1, 0)
    assert message["final_code"] == 0
    assert message["final_label"] == "OK"
    assert message["ok"] is True
    assert message["payload"]["object_count"] == 0


def test_final_decision_has_priority_over_detections() -> None:
    result = detection_result()
    result["final_decision"] = {
        "code": "NG_CUSTOM",
        "label": "custom_ng",
        "ok": False,
        "reason": "专用应用判定失败",
    }
    message = inference_result_to_gateway_message(result, "generic_mock", 2, 1)
    assert message["final_code"] == "NG_CUSTOM"
    assert message["final_label"] == "custom_ng"
    assert message["ok"] is False
    assert message["reason"] == "专用应用判定失败"


def test_center_falls_back_to_bbox_midpoint() -> None:
    result = detection_result()
    result["detections"] = [deepcopy(result["detections"][0])]
    message = inference_result_to_gateway_message(result, "generic_mock", 3, 0)
    assert message["payload"]["center_x"] == 60
    assert message["payload"]["center_y"] == 120


def test_id_low_values_are_stable_uint16() -> None:
    first = inference_result_to_gateway_message(detection_result(), "generic_mock", 4, 0)
    second = inference_result_to_gateway_message(detection_result(), "generic_mock", 4, 0)
    for key in ("frame_id_low", "result_id_low"):
        assert first["payload"][key] == second["payload"][key]
        assert 0 <= first["payload"][key] <= 65535
