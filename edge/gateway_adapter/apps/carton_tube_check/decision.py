"""carton_tube_check 纯业务决策逻辑。"""

from __future__ import annotations

from typing import Any, Mapping

from edge.gateway_adapter.gateway_message import stable_u16
from edge.gateway_adapter.apps.common.app_decision import AppDecision, FinalCode


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


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


def evaluate(
    result: dict,
    rules: Mapping[str, Any],
    sequence: int,
    heartbeat: int,
    device_id: str,
) -> AppDecision:
    targets = {
        str(name) for name in rules.get("target_class_names", []) if isinstance(name, str)
    }
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    candidates = [
        item for item in detections
        if isinstance(item, Mapping) and str(item.get("class_name", "")) in targets
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
    reason = "纸筒目标满足当前 Mock 规则"
    if best is None:
        code, label, reason = FinalCode.NO_TARGET, "NO_TARGET", "未检测到纸筒目标"
    elif score < float(rules.get("score_threshold", 0.5)):
        code, label, reason = FinalCode.LOW_CONFIDENCE, "LOW_CONFIDENCE", "纸筒目标置信度低于阈值"
    elif not bool(rules.get("allow_multi_target", False)) and target_count > 1:
        code, label, reason = FinalCode.MULTI_TARGET, "MULTI_TARGET", "检测到多个纸筒目标"
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
