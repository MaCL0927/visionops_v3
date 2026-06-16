"""carton_partition_check 业务决策逻辑。

M11 起本模块支持真实 Runtime/RKNN 的 cell/defect 检测结果，并提供
轻量网格结构检查。v2 中基于模板的精细校准逻辑不原样迁移，后续可在
本模块继续扩展模板文件和槽位匹配。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from edge.gateway_adapter.gateway_message import stable_u16
from edge.gateway_adapter.apps.common.app_decision import AppDecision, FinalCode


def _score(item: Mapping[str, Any]) -> float:
    value = item.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _name_set(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw if isinstance(item, (str, int, float))}
    return set()


def _id_set(raw: Any) -> set[int]:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return {raw}
    if isinstance(raw, list):
        return {_int(item, -1) for item in raw if _int(item, -1) >= 0}
    return set()


def _geometry(item: Mapping[str, Any] | None) -> tuple[list[int], list[int]]:
    if item is None:
        return [0, 0, 0, 0], [0, 0]
    bbox = item.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("partition detection 缺少 bbox_xyxy")
    box = [round(float(value)) for value in bbox]
    center = item.get("center_xy")
    if isinstance(center, list) and len(center) == 2:
        return box, [round(float(center[0])), round(float(center[1]))]
    return box, [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]


def _matches(item: Mapping[str, Any], names: set[str], ids: set[int]) -> bool:
    class_name = str(item.get("class_name", ""))
    class_id = _int(item.get("class_id"), -1)
    return (class_name in names) or (class_id in ids)


def _error_decision(
    result: Mapping[str, Any],
    *,
    device_id: str,
    sequence: int,
    heartbeat: int,
    code: FinalCode,
    label: str,
    reason: str,
) -> AppDecision:
    error = result.get("error") if isinstance(result.get("error"), Mapping) else None
    return AppDecision(
        app_id="carton_partition_check",
        device_id=device_id,
        frame_id=str(result.get("frame_id", "")),
        result_id=str(result.get("result_id", "")),
        sequence=sequence,
        heartbeat=heartbeat,
        final_code=int(code),
        final_label=label,
        ok=False,
        reason_code=int(code),
        reason=reason,
        object_count=0,
        confidence_x1000=0,
        details={
            "expected_cell_count": 0,
            "cell_count": 0,
            "defect_count": 0,
            "missing_count": 0,
            "result_status": result.get("status"),
            "message_type": result.get("message_type"),
            "task_type": result.get("task_type"),
            "frame_id_low": stable_u16(result.get("frame_id")),
            "result_id_low": stable_u16(result.get("result_id")),
            "error_code": stable_u16(error.get("code")) if isinstance(error, Mapping) else int(code),
        },
        error=dict(error) if isinstance(error, Mapping) else {
            "code": label,
            "message": reason,
            "detail": None,
            "recoverable": True,
        },
    )


def _center_of(item: Mapping[str, Any]) -> tuple[float, float]:
    _, center = _geometry(item)
    return float(center[0]), float(center[1])


def _angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_abs_diff(a: float, target: float) -> float:
    diff = (a - target + 180.0) % 360.0 - 180.0
    return abs(diff)


def _grid_metrics(cells: list[Mapping[str, Any]], rows: int, cols: int) -> dict[str, Any]:
    """基于检测框中心的轻量网格指标，不依赖 numpy/opencv。"""
    if rows <= 0 or cols <= 0 or len(cells) < rows * cols:
        return {
            "grid_check_enabled": bool(rows and cols),
            "grid_rows": rows,
            "grid_cols": cols,
            "grid_valid": False,
            "row_angle_max_abs_deg": 0.0,
            "col_angle_max_abs_deg": 0.0,
        }
    centers = [(_center_of(item), item) for item in cells]
    centers.sort(key=lambda pair: pair[0][1])
    row_groups = [centers[index * cols:(index + 1) * cols] for index in range(rows)]
    row_groups = [sorted(group, key=lambda pair: pair[0][0]) for group in row_groups]
    row_angles = []
    for group in row_groups:
        if len(group) >= 2:
            row_angles.append(_angle_deg(group[0][0], group[-1][0]))
    col_angles = []
    for col in range(cols):
        column = [row_groups[row][col] for row in range(rows) if col < len(row_groups[row])]
        if len(column) >= 2:
            col_angles.append(_angle_deg(column[0][0], column[-1][0]))
    flat = [point for group in row_groups for point, _item in group]
    grid_center_x = sum(point[0] for point in flat) / len(flat) if flat else 0.0
    grid_center_y = sum(point[1] for point in flat) / len(flat) if flat else 0.0
    return {
        "grid_check_enabled": True,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_valid": len(flat) == rows * cols,
        "grid_center_x": round(grid_center_x, 2),
        "grid_center_y": round(grid_center_y, 2),
        "row_angle_mean_deg": round(sum(row_angles) / len(row_angles), 3) if row_angles else 0.0,
        "row_angle_max_abs_deg": round(max((_angle_abs_diff(value, 0.0) for value in row_angles), default=0.0), 3),
        "col_angle_mean_deg": round(sum(col_angles) / len(col_angles), 3) if col_angles else 0.0,
        "col_angle_max_abs_deg": round(max((_angle_abs_diff(value, 90.0) for value in col_angles), default=0.0), 3),
    }


def evaluate(
    result: dict,
    rules: Mapping[str, Any],
    sequence: int,
    heartbeat: int,
    device_id: str,
) -> AppDecision:
    if result.get("message_type") != "inference_result":
        return _error_decision(
            result,
            device_id=device_id,
            sequence=sequence,
            heartbeat=heartbeat,
            code=FinalCode.UPSTREAM_NO_RESULT,
            label="UPSTREAM_NO_RESULT",
            reason="上游返回内容不是 inference_result",
        )
    if result.get("status") not in (None, "ok"):
        return _error_decision(
            result,
            device_id=device_id,
            sequence=sequence,
            heartbeat=heartbeat,
            code=FinalCode.UPSTREAM_NO_RESULT,
            label="UPSTREAM_NO_RESULT",
            reason="上游推理结果状态不是 ok",
        )
    accepted_tasks = _name_set(rules.get("accepted_task_types", ["detection", "detect"]))
    task_type = str(result.get("task_type", ""))
    if accepted_tasks and task_type not in accepted_tasks:
        return _error_decision(
            result,
            device_id=device_id,
            sequence=sequence,
            heartbeat=heartbeat,
            code=FinalCode.UPSTREAM_NO_RESULT,
            label="UPSTREAM_TASK_MISMATCH",
            reason=f"上游 task_type={task_type} 不属于当前隔板业务",
        )

    cell_names = _name_set(rules.get("target_class_names", []))
    cell_ids = _id_set(rules.get("target_class_ids", []))
    defect_names = _name_set(rules.get("defect_class_names", []))
    defect_ids = _id_set(rules.get("defect_class_ids", []))
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    raw_cells = [item for item in detections if isinstance(item, Mapping) and _matches(item, cell_names, cell_ids)]
    raw_defects = [item for item in detections if isinstance(item, Mapping) and _matches(item, defect_names, defect_ids)]
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
    rows = _int(rules.get("expected_rows"), 0)
    cols = _int(rules.get("expected_cols"), 0)
    grid = _grid_metrics(cells, rows, cols)

    code = FinalCode.OK
    label = "OK"
    reason = "隔板单元数量、缺陷和轻量网格规则通过"
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
    else:
        max_row_angle = _float(rules.get("max_row_angle_abs_deg"), 0.0)
        max_col_angle = _float(rules.get("max_col_angle_abs_deg"), 0.0)
        if max_row_angle > 0 and grid.get("row_angle_max_abs_deg", 0.0) > max_row_angle:
            code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "隔板行方向角度超限"
        elif max_col_angle > 0 and grid.get("col_angle_max_abs_deg", 0.0) > max_col_angle:
            code, label, reason = FinalCode.STRUCTURE_ABNORMAL, "STRUCTURE_ABNORMAL", "隔板列方向角度超限"

    model = result.get("model") if isinstance(result.get("model"), Mapping) else {}
    image = result.get("image") if isinstance(result.get("image"), Mapping) else {}
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
        "task_type": task_type,
        "model_name": model.get("model_name"),
        "backend": model.get("backend"),
        "image_width": _int(image.get("width")),
        "image_height": _int(image.get("height")),
        **grid,
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
        measurements={"cell_count": cell_count, "defect_count": defect_count, **grid},
        details=details,
    )
