"""将标准 inference_result 转换为 Gateway/Modbus 中间消息。"""

from __future__ import annotations

from typing import Any, Mapping

from .gateway_message import make_message_id, numeric_code, stable_u16, timestamp_ms
from .register_map import DEFAULT_REGISTER_MAP


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _best_detection(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    detections = result.get("detections")
    if not isinstance(detections, list):
        return None
    candidates = [item for item in detections if isinstance(item, Mapping)]
    return max(candidates, key=lambda item: _number(item.get("score")), default=None)


def _geometry(detection: Mapping[str, Any] | None) -> dict[str, int]:
    values = {
        "score_x1000": 0,
        "center_x": 0,
        "center_y": 0,
        "bbox_x1": 0,
        "bbox_y1": 0,
        "bbox_x2": 0,
        "bbox_y2": 0,
    }
    if detection is None:
        return values

    values["score_x1000"] = round(max(0.0, min(1.0, _number(detection.get("score")))) * 1000)
    bbox = detection.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = (_number(item) for item in bbox)
        values.update(
            {
                "bbox_x1": round(x1),
                "bbox_y1": round(y1),
                "bbox_x2": round(x2),
                "bbox_y2": round(y2),
            }
        )
        center = detection.get("center_xy")
        if isinstance(center, list) and len(center) == 2:
            values["center_x"] = round(_number(center[0]))
            values["center_y"] = round(_number(center[1]))
        else:
            values["center_x"] = round((x1 + x2) / 2.0)
            values["center_y"] = round((y1 + y2) / 2.0)
    return values


def _decision(result: Mapping[str, Any], object_count: int) -> tuple[Any, str, bool, str]:
    decision = result.get("final_decision")
    if isinstance(decision, Mapping):
        return (
            decision.get("code", 0),
            str(decision.get("label", "")),
            bool(decision.get("ok", False)),
            str(decision.get("reason", "")),
        )
    if object_count > 0:
        return 1, "NG_OR_DETECTED", True, "检测到一个或多个目标"
    return 0, "OK", True, "未检测到目标"


def inference_result_to_gateway_message(
    result: dict,
    app_id: str,
    sequence: int,
    heartbeat: int,
) -> dict:
    """把标准推理结果映射为 Gateway 消息和默认寄存器值。"""
    if not isinstance(result, dict):
        raise TypeError("result 必须是对象")
    for field in ("frame_id", "result_id"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"inference_result 缺少必需字段: {field}")
    if not app_id:
        raise ValueError("app_id 不能为空")

    detections = result.get("detections")
    object_count = len(detections) if isinstance(detections, list) else 0
    best = _best_detection(result)
    geometry = _geometry(best)
    final_code, final_label, ok, reason = _decision(result, object_count)
    timing = result.get("timing") if isinstance(result.get("timing"), Mapping) else {}
    error = result.get("error") if isinstance(result.get("error"), Mapping) else {}

    payload: dict[str, Any] = {
        "heartbeat": int(heartbeat) & 1,
        "sequence": int(sequence) & 0xFFFF,
        "status_code": 0 if result.get("status", "ok") == "ok" else 1,
        "final_code": numeric_code(final_code),
        "ok": 1 if ok else 0,
        "reason_code": stable_u16(reason),
        "error_code": stable_u16(error.get("code")) if error else 0,
        "object_count": object_count,
        **geometry,
        "inference_ms": round(_number(timing.get("inference_ms"))),
        "total_ms": round(_number(timing.get("total_ms"))),
        "frame_id_low": stable_u16(result["frame_id"]),
        "result_id_low": stable_u16(result["result_id"]),
        "reserved": 0,
    }
    registers = [
        {
            "address": definition.address,
            "name": definition.name,
            "type": definition.data_type,
            "value": payload[definition.name],
            "scale": definition.scale,
            "byte_order": "big",
        }
        for definition in DEFAULT_REGISTER_MAP
    ]
    now = timestamp_ms()
    return {
        "schema_version": "1.0",
        "message_type": "gateway_message",
        "device_id": str(result.get("device_id", "unknown-device")),
        "component": "gateway_mock",
        "timestamp_ms": now,
        "trace_id": str(result.get("trace_id", f"trace-gateway-{now}")),
        "frame_id": result["frame_id"],
        "source": f"inference_result:{result['result_id']}",
        "status": "ok",
        "message_id": make_message_id(sequence, result["result_id"]),
        "result_id": result["result_id"],
        "app_id": app_id,
        "protocol": "mock",
        "sequence": int(sequence),
        "heartbeat": bool(int(heartbeat) & 1),
        "final_code": final_code,
        "final_label": final_label,
        "ok": ok,
        "reason": reason,
        "registers": registers,
        "payload": payload,
        "ack": {
            "required": False,
            "timeout_ms": 1000,
            "correlation_id": make_message_id(sequence, result["result_id"]),
        },
    }
