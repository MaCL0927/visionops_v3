"""carton_partition_check 纯业务决策逻辑。"""

from __future__ import annotations

from typing import Any, Mapping

from edge.gateway_adapter.gateway_message import stable_u16
from edge.gateway_adapter.apps.common.app_decision import AppDecision, FinalCode


def _score(item: Mapping[str, Any]) -> float:
    value = item.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _geometry(item: Mapping[str, Any] | None) -> tuple[list[int], list[int]]:
    if item is None:
        return [0, 0, 0, 0], [0, 0]
    bbox = item.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("defect detection 缺少 bbox_xyxy")
    box = [round(float(value)) for value in bbox]
    center = item.get("center_xy")
    if isinstance(center, list) and len(center) == 2:
        return box, [round(float(center[0])), round(float(center[1]))]
    return box, [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]


def evaluate(
    result: dict,
    rules: Mapping[str, Any],
    sequence: int,
    heartbeat: int,
    device_id: str,
) -> AppDecision:
    cell_names = {str(name) for name in rules.get("target_class_names", [])}
    defect_names = {str(name) for name in rules.get("defect_class_names", [])}
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    raw_cells = [item for item in detections if isinstance(item, Mapping) and str(item.get("class_name", "")) in cell_names]
    raw_defects = [item for item in detections if isinstance(item, Mapping) and str(item.get("class_name", "")) in defect_names]
    cells = [item for item in raw_cells if _score(item) >= float(rules.get("score_threshold", 0.5))]
    defects = [item for item in raw_defects if _score(item) >= float(rules.get("defect_score_threshold", 0.5))]
    cell_count = len(cells)
    defect_count = len(defects)
    expected = int(rules.get("expected_cell_count", 0) or 0)
    missing_count = max(0, expected - cell_count) if expected else 0
    first_defect = max(defects, key=_score, default=None)
    box, center = _geometry(first_defect)
    max_defect_score = round(_score(first_defect) * 1000) if first_defect else 0
    best_cell_score = max((_score(item) for item in cells), default=0.0)

    code = FinalCode.OK
    label = "OK"
    reason = "隔板单元数量和缺陷规则通过"
    if defect_count > 0:
        code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "检测到隔板缺陷目标"
    elif not raw_cells:
        code, label, reason = FinalCode.NO_TARGET, "NO_TARGET", "未检测到隔板单元"
    elif not cells:
        code, label, reason = FinalCode.LOW_CONFIDENCE, "LOW_CONFIDENCE", "隔板单元候选全部低于置信度阈值"
    elif expected and cell_count != expected:
        code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "隔板单元数量与 expected_cell_count 不一致"
    elif "min_cell_count" in rules and cell_count < int(rules["min_cell_count"]):
        code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "隔板单元数低于下限"
    elif "max_cell_count" in rules and cell_count > int(rules["max_cell_count"]):
        code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "隔板单元数高于上限"

    details = {
        "expected_cell_count": expected,
        "cell_count": cell_count,
        "defect_count": defect_count,
        "missing_count": missing_count,
        "max_defect_score_x1000": max_defect_score,
        "first_defect_center_x": center[0],
        "first_defect_center_y": center[1],
        "first_defect_bbox_x1": box[0],
        "first_defect_bbox_y1": box[1],
        "first_defect_bbox_x2": box[2],
        "first_defect_bbox_y2": box[3],
        "frame_id_low": stable_u16(result.get("frame_id")),
        "result_id_low": stable_u16(result.get("result_id")),
        "error_code": stable_u16(result.get("error", {}).get("code")) if isinstance(result.get("error"), Mapping) else 0,
    }
    primary = None if first_defect is None else {
        "class_name": str(first_defect.get("class_name", "")),
        "score": _score(first_defect),
        "bbox_xyxy": box,
        "center_xy": center,
    }
    confidence = max(max_defect_score, round(best_cell_score * 1000))
    return AppDecision(
        app_id="carton_partition_check",
        device_id=device_id,
        frame_id=str(result.get("frame_id", "")),
        result_id=str(result.get("result_id", "")),
        sequence=sequence,
        heartbeat=heartbeat,
        final_code=int(code),
        final_label=label,
        ok=code == FinalCode.OK,
        reason_code=int(code),
        reason=reason,
        object_count=cell_count + defect_count,
        confidence_x1000=confidence,
        primary_target=primary,
        measurements={"cell_count": cell_count, "defect_count": defect_count},
        details=details,
    )
