"""Robot protocol holding-register definitions and helpers."""

from __future__ import annotations

from typing import Iterable

from edge.modbus_adapter.modbus_registers import HoldingRegisterBank, RegisterDefinition


REG_VISION_HEARTBEAT = 0
REG_PARTITION_RESULT = 1
REG_PRODUCT_RESULT = 2
REG_COORD_RESULT = 3
REG_COORD_BASE = 20
REG_ROBOT_HEARTBEAT = 100
REG_TRIGGER_PARTITION = 101
REG_TRIGGER_PRODUCT = 102
REG_TRIGGER_COORD = 103

RESULT_NONE = 0
RESULT_OK = 1
RESULT_NG = 2


def _name_description(offset: int) -> tuple[str, str, str]:
    fixed = {
        REG_VISION_HEARTBEAT: ("vision_heartbeat", "uint16", "视觉服务心跳"),
        REG_PARTITION_RESULT: ("partition_result", "uint16", "隔板判断结果: 0空闲/1正常/2异常"),
        REG_PRODUCT_RESULT: ("product_result", "uint16", "纸筒产品判断结果: 0空闲/1正常/2异常"),
        REG_COORD_RESULT: ("coordinate_result", "uint16", "坐标识别结果: 0空闲/1正常/2异常"),
        REG_ROBOT_HEARTBEAT: ("robot_heartbeat", "uint16", "PLC/机器人心跳"),
        REG_TRIGGER_PARTITION: ("trigger_partition", "uint16", "隔板检测触发: 0空闲/1触发"),
        REG_TRIGGER_PRODUCT: ("trigger_product", "uint16", "纸筒触发: 1左/2右/3全部"),
        REG_TRIGGER_COORD: ("trigger_coordinate", "uint16", "坐标识别触发: 0空闲/1触发"),
    }
    if offset in fixed:
        return fixed[offset]
    if REG_COORD_BASE <= offset < REG_COORD_BASE + 80:
        slot = (offset - REG_COORD_BASE) // 2 + 1
        axis = "x" if (offset - REG_COORD_BASE) % 2 == 0 else "y"
        return f"slot_{slot}_{axis}", "int16", f"第 {slot} 个槽位 {axis.upper()} 坐标（补码）"
    return f"reserved_{offset}", "uint16", "预留保持寄存器"


def make_definitions(address_base: int, register_count: int) -> tuple[RegisterDefinition, ...]:
    definitions = []
    for offset in range(register_count):
        name, data_type, description = _name_description(offset)
        definitions.append(RegisterDefinition(address_base + offset, name, data_type, 1.0, description))
    return tuple(definitions)


class ProtocolRegisterBank(HoldingRegisterBank):
    def __init__(self, address_base: int = 0, register_count: int = 200) -> None:
        self.address_base = int(address_base)
        self.register_count = int(register_count)
        self.definitions = make_definitions(self.address_base, self.register_count)
        super().__init__(self.definitions)

    def _address(self, logical_offset: int) -> int:
        if not 0 <= logical_offset < self.register_count:
            raise ValueError(f"逻辑寄存器超出范围: {logical_offset}")
        return self.address_base + logical_offset

    def get(self, logical_offset: int) -> int:
        return self.read(self._address(logical_offset), 1)[0]

    def set(self, logical_offset: int, value: int) -> None:
        self.write(self._address(logical_offset), [int(value) & 0xFFFF])

    def set_many(self, logical_offset: int, values: Iterable[int]) -> None:
        self.write(self._address(logical_offset), [int(value) & 0xFFFF for value in values])

    def logical_snapshot(self) -> list[dict[str, object]]:
        rows = self.snapshot()
        for row in rows:
            row["logical_address"] = int(row["address"]) - self.address_base
        return rows
