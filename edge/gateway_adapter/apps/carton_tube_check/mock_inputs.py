"""carton_tube_check 的无图像 Mock inference_result。"""

from __future__ import annotations

from copy import deepcopy


def _base(detections: list[dict], suffix: str) -> dict:
    return {
        "schema_version": "1.0", "message_type": "inference_result",
        "device_id": "example-edge-001", "component": "rknn_runtime_mock",
        "timestamp_ms": 1760000001000, "trace_id": f"trace-tube-{suffix}",
        "frame_id": f"frame-tube-{suffix}-0001", "source": "runtime:mock", "status": "ok",
        "result_id": f"result-tube-{suffix}-0001", "task_type": "detection",
        "model": {"model_id": "tube-mock", "model_name": "tube-detector-mock", "model_version": "1.0.0", "backend": "mock", "input_size": {"width": 640, "height": 640}},
        "image": {"width": 1280, "height": 720},
        "timing": {"preprocess_ms": 1.0, "inference_ms": 5.0, "postprocess_ms": 1.0, "total_ms": 7.0},
        "detections": deepcopy(detections),
    }


def _tube(score: float = 0.92, bbox: list[float] | None = None, ident: str = "1") -> dict:
    box = bbox or [540, 260, 740, 460]
    return {"id": f"tube-{ident}", "class_id": 0, "class_name": "carton_tube", "score": score, "bbox_xyxy": box, "center_xy": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}


def make_ok_result() -> dict: return _base([_tube()], "ok")
def make_no_target_result() -> dict: return _base([], "no-target")
def make_low_confidence_result() -> dict: return _base([_tube(0.2)], "low-confidence")
def make_multi_target_result() -> dict: return _base([_tube(0.92, ident="1"), _tube(0.85, [300, 250, 470, 430], "2")], "multi")
def make_out_of_roi_result() -> dict: return _base([_tube(0.92, [1100, 560, 1260, 710])], "out-of-roi")
def make_size_out_of_range_result() -> dict: return _base([_tube(0.92, [400, 180, 900, 600])], "size")


MOCK_CASES = {
    "ok": make_ok_result, "ng": make_out_of_roi_result, "no_target": make_no_target_result,
    "low_confidence": make_low_confidence_result, "multi_target": make_multi_target_result,
    "out_of_roi": make_out_of_roi_result, "size_out_of_range": make_size_out_of_range_result,
}
