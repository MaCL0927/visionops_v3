"""carton_partition_check 的无图像 Mock inference_result。"""

from __future__ import annotations

from copy import deepcopy


def _base(detections: list[dict], suffix: str) -> dict:
    return {
        "schema_version": "1.0", "message_type": "inference_result",
        "device_id": "example-edge-001", "component": "rknn_runtime_mock",
        "timestamp_ms": 1760000002000, "trace_id": f"trace-partition-{suffix}",
        "frame_id": f"frame-partition-{suffix}-0001", "source": "runtime:mock", "status": "ok",
        "result_id": f"result-partition-{suffix}-0001", "task_type": "detection",
        "model": {"model_id": "partition-mock", "model_name": "partition-detector-mock", "model_version": "1.0.0", "backend": "mock", "input_size": {"width": 640, "height": 640}},
        "image": {"width": 1280, "height": 720},
        "timing": {"preprocess_ms": 1.0, "inference_ms": 6.0, "postprocess_ms": 1.0, "total_ms": 8.0},
        "detections": deepcopy(detections),
    }


def _cell(index: int, score: float = 0.9) -> dict:
    col, row = index % 4, index // 4
    x1, y1 = 180 + col * 220, 160 + row * 180
    return {"id": f"cell-{index}", "class_id": 0, "class_name": "cell", "score": score, "bbox_xyxy": [x1, y1, x1 + 120, y1 + 100], "center_xy": [x1 + 60, y1 + 50]}


def _defect(score: float = 0.88) -> dict:
    return {"id": "defect-1", "class_id": 1, "class_name": "broken_partition", "score": score, "bbox_xyxy": [500, 280, 680, 440], "center_xy": [590, 360]}


def make_ok_result() -> dict: return _base([_cell(index) for index in range(12)], "ok")
def make_no_target_result() -> dict: return _base([], "no-target")
def make_missing_cell_result() -> dict: return _base([_cell(index) for index in range(10)], "missing")
def make_defect_result() -> dict: return _base([*[_cell(index) for index in range(12)], _defect()], "defect")
def make_low_confidence_result() -> dict: return _base([_cell(index, 0.2) for index in range(12)], "low-confidence")


MOCK_CASES = {
    "ok": make_ok_result, "ng": make_defect_result, "no_target": make_no_target_result,
    "missing_cell": make_missing_cell_result, "defect": make_defect_result,
    "low_confidence": make_low_confidence_result,
}
