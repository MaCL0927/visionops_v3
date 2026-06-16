"""carton_tube_check 业务决策逻辑。

M11 起该模块既支持 M6 mock result，也支持真实 Runtime/RKNN 输出的
标准 inference_result。业务规则保持在 Gateway app 层，不进入 Runtime。
"""

from __future__ import annotations

from typing import Any, Mapping

from edge.gateway_adapter.gateway_message import stable_u16
from edge.gateway_adapter.apps.common.app_decision import AppDecision, FinalCode


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
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


def _geometry(detection: Mapping[str, Any]) -> tuple[list[int], list[int], int, int]:
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("纸筒 detection 缺少 bbox_xyxy")
    box = [round(_number(value)) for value in bbox]
    center = detection.get("center_xy")
    if isinstance(center, list) and len(center) == 2:
        center_xy = [round(_number(center[0])), round(_number(center[1]))]
    else:
        center_xy = [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]
    return box, center_xy, max(0, box[2] - box[0]), max(0, box[3] - box[1])


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
        app_id="carton_tube_check",
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
            "target_count": 0,
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


def _matches_target(item: Mapping[str, Any], names: set[str], ids: set[int]) -> bool:
    class_name = str(item.get("class_name", ""))
    class_id = _int(item.get("class_id"), -1)
    return (class_name in names) or (class_id in ids)


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
            reason=f"上游 task_type={task_type} 不属于当前纸筒业务",
        )

    targets = _name_set(rules.get("target_class_names", []))
    target_ids = _id_set(rules.get("target_class_ids", []))
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    candidates = [
        item for item in detections
        if isinstance(item, Mapping) and _matches_target(item, targets, target_ids)
    ]
    best = max(candidates, key=lambda item: _number(item.get("score")), default=None)
    target_count = len(candidates)
    score = _number(best.get("score")) if best else 0.0
    score_x1000 = round(max(0.0, min(1.0, score)) * 1000)
    box = [0, 0, 0, 0]
    center = [0, 0]
    bbox_w = bbox_h = offset_x = offset_y = 0
    if best is not None:
        box, center, bbox_w, bbox_h = _geometry(best)
        expected = rules.get("expected_center_xy")
        if isinstance(expected, list) and len(expected) == 2:
            offset_x = center[0] - round(_number(expected[0]))
            offset_y = center[1] - round(_number(expected[1]))

    code = FinalCode.OK
    label = "OK"
    reason = "纸筒目标满足当前业务规则"
    min_target_count = _int(rules.get("min_target_count"), 0)
    max_target_count = rules.get("max_target_count")
    if best is None or (min_target_count and target_count < min_target_count):
        code, label, reason = FinalCode.NO_TARGET, "NO_TARGET", "未检测到足够数量的纸筒目标"
    elif score < float(rules.get("score_threshold", 0.5)):
        code, label, reason = FinalCode.LOW_CONFIDENCE, "LOW_CONFIDENCE", "纸筒目标置信度低于阈值"
    elif not bool(rules.get("allow_multi_target", False)) and target_count > 1:
        code, label, reason = FinalCode.MULTI_TARGET, "MULTI_TARGET", "检测到多个纸筒目标"
    elif max_target_count is not None and target_count > _int(max_target_count, target_count):
        code, label, reason = FinalCode.MULTI_TARGET, "MULTI_TARGET", "纸筒目标数量高于上限"
    else:
        roi = rules.get("roi_xyxy", [0, 0, 65535, 65535])
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError("roi_xyxy 必须包含四个数值")
        inside_roi = _number(roi[0]) <= center[0] <= _number(roi[2]) and _number(roi[1]) <= center[1] <= _number(roi[3])
        expected = rules.get("expected_center_xy")
        tolerance = rules.get("center_tolerance_px")
        center_ok = True
        if isinstance(expected, list) and len(expected) == 2 and tolerance is not None:
            center_ok = abs(offset_x) <= _number(tolerance) and abs(offset_y) <= _number(tolerance)
        if not inside_roi or not center_ok:
            code, label, reason = FinalCode.OUT_OF_ROI, "OUT_OF_ROI", "纸筒中心超出 ROI 或中心偏差限制"
        else:
            limits = (
                ("min_bbox_width", bbox_w, lambda value, limit: value < limit),
                ("max_bbox_width", bbox_w, lambda value, limit: value > limit),
                ("min_bbox_height", bbox_h, lambda value, limit: value < limit),
                ("max_bbox_height", bbox_h, lambda value, limit: value > limit),
            )
            if any(key in rules and compare(value, _number(rules[key])) for key, value, compare in limits):
                code, label, reason = FinalCode.SIZE_OUT_OF_RANGE, "SIZE_OUT_OF_RANGE", "纸筒包围框尺寸超出配置范围"

    model = result.get("model") if isinstance(result.get("model"), Mapping) else {}
    image = result.get("image") if isinstance(result.get("image"), Mapping) else {}
    details = {
        "target_count": target_count,
        "best_score_x1000": score_x1000,
        "center_x": center[0],
        "center_y": center[1],
        "bbox_x1": box[0],
        "bbox_y1": box[1],
        "bbox_x2": box[2],
        "bbox_y2": box[3],
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "frame_id_low": stable_u16(result.get("frame_id")),
        "result_id_low": stable_u16(result.get("result_id")),
        "error_code": stable_u16(result.get("error", {}).get("code")) if isinstance(result.get("error"), Mapping) else 0,
        "task_type": task_type,
        "model_name": model.get("model_name"),
        "backend": model.get("backend"),
        "image_width": _int(image.get("width")),
        "image_height": _int(image.get("height")),
    }
    primary = None if best is None else {
        "class_name": str(best.get("class_name", "")),
        "score": score,
        "bbox_xyxy": box,
        "center_xy": center,
    }
    return AppDecision(
        app_id="carton_tube_check",
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
        object_count=target_count,
        confidence_x1000=score_x1000,
        primary_target=primary,
        measurements={"bbox_width": bbox_w, "bbox_height": bbox_h},
        details=details,
    )
