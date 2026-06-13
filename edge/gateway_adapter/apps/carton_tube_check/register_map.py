"""carton_tube_check 业务寄存器映射。"""

from __future__ import annotations

from edge.modbus_adapter.modbus_registers import RegisterDefinition
from edge.gateway_adapter.apps.common.app_decision import AppDecision
from edge.gateway_adapter.apps.common.app_register_bank import encode_int16


NAMES = (
    "heartbeat", "sequence", "final_code", "ok", "reason_code", "target_count",
    "confidence_x1000", "center_x", "center_y", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "bbox_w", "bbox_h", "offset_x_signed", "offset_y_signed",
    "frame_id_low", "result_id_low", "error_code",
)


def make_register_map(base: int = 100) -> tuple[RegisterDefinition, ...]:
    descriptions = {
        "offset_x_signed": "中心 X 偏差，int16 以 uint16 传输",
        "offset_y_signed": "中心 Y 偏差，int16 以 uint16 传输",
    }
    return tuple(
        RegisterDefinition(base + index, name, "uint16", 1.0, descriptions.get(name, f"纸筒业务 {name}"))
        for index, name in enumerate(NAMES)
    )


def decision_register_values(decision: AppDecision) -> dict[str, int]:
    details = decision.details
    return {
        "heartbeat": decision.heartbeat,
        "sequence": decision.sequence & 0xFFFF,
        "final_code": decision.final_code,
        "ok": int(decision.ok),
        "reason_code": decision.reason_code,
        "target_count": int(details.get("target_count", 0)),
        "confidence_x1000": decision.confidence_x1000,
        "center_x": int(details.get("center_x", 0)),
        "center_y": int(details.get("center_y", 0)),
        "bbox_x1": int(details.get("bbox_x1", 0)),
        "bbox_y1": int(details.get("bbox_y1", 0)),
        "bbox_x2": int(details.get("bbox_x2", 0)),
        "bbox_y2": int(details.get("bbox_y2", 0)),
        "bbox_w": int(details.get("bbox_w", 0)),
        "bbox_h": int(details.get("bbox_h", 0)),
        "offset_x_signed": encode_int16(int(details.get("offset_x", 0))),
        "offset_y_signed": encode_int16(int(details.get("offset_y", 0))),
        "frame_id_low": int(details.get("frame_id_low", 0)),
        "result_id_low": int(details.get("result_id_low", 0)),
        "error_code": int(details.get("error_code", 0)),
    }
