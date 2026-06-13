"""业务 App 决策到 GatewayMessage 和寄存器的通用转换。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from edge.gateway_adapter.gateway_message import make_message_id, stable_u16
from edge.modbus_adapter.modbus_registers import HoldingRegisterBank, RegisterDefinition

from .app_decision import AppDecision


def encode_int16(value: int) -> int:
    """将 -32768..32767 编码为 Modbus uint16。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("int16 值必须是整数")
    if not -32768 <= value <= 32767:
        raise ValueError(f"int16 值超出范围: {value}")
    return value & 0xFFFF


def decode_int16(value: int) -> int:
    """将 Modbus uint16 解码为 int16。"""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF:
        raise ValueError("uint16 值必须位于 0..65535")
    return value - 0x10000 if value >= 0x8000 else value


def decision_to_gateway_message(
    decision: AppDecision,
    definitions: Iterable[RegisterDefinition],
    register_values: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(register_values)
    registers = [
        {
            "address": definition.address,
            "name": definition.name,
            "type": definition.data_type,
            "value": payload.get(definition.name, 0),
            "scale": definition.scale,
            "byte_order": "big",
        }
        for definition in definitions
    ]
    message_id = make_message_id(decision.sequence, decision.result_id)
    return {
        "schema_version": "1.0",
        "message_type": "gateway_message",
        "device_id": decision.device_id,
        "component": f"{decision.app_id}_mock",
        "timestamp_ms": decision.timestamp_ms,
        "trace_id": f"trace-{decision.app_id}-{stable_u16(decision.result_id):05d}",
        "frame_id": decision.frame_id,
        "source": f"app_decision:{decision.app_id}",
        "status": "ok" if decision.error is None else "error",
        "message_id": message_id,
        "result_id": decision.result_id,
        "app_id": decision.app_id,
        "protocol": "modbus_tcp",
        "sequence": decision.sequence,
        "heartbeat": bool(decision.heartbeat),
        "final_code": decision.final_code,
        "final_label": decision.final_label,
        "ok": decision.ok,
        "reason": decision.reason,
        "registers": registers,
        "payload": payload,
        "ack": {"required": False, "timeout_ms": 1000, "correlation_id": message_id},
    }


class AppRegisterBank(HoldingRegisterBank):
    """带业务 register map 的 M5 HoldingRegisterBank。"""

    def __init__(self, definitions: Iterable[RegisterDefinition]) -> None:
        self.definitions = tuple(definitions)
        super().__init__(self.definitions)

    def update_decision(
        self,
        decision: AppDecision,
        register_values: Mapping[str, Any],
    ) -> dict[str, Any]:
        message = decision_to_gateway_message(decision, self.definitions, register_values)
        self.update_from_gateway_message(message)
        return message

    def register_map(self) -> list[dict[str, Any]]:
        return [
            {
                "address": item.address,
                "name": item.name,
                "type": item.data_type,
                "scale": item.scale,
                "description": item.description,
            }
            for item in self.definitions
        ]
