"""线程安全的 Holding Register 数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Mapping


class RegisterAddressError(IndexError):
    """寄存器地址或范围无效。"""


@dataclass(frozen=True)
class RegisterDefinition:
    address: int
    name: str
    data_type: str
    scale: float
    description: str


class HoldingRegisterBank:
    """只保存 16 位无符号 Holding Register 的线程安全内存库。"""

    def __init__(self, definitions: Iterable[RegisterDefinition] | None = None) -> None:
        if definitions is None:
            raise ValueError("寄存器定义不能为空；请由具体生产协议显式传入")
        ordered = sorted(definitions, key=lambda item: item.address)
        if not ordered:
            raise ValueError("寄存器定义不能为空")
        addresses = [item.address for item in ordered]
        if len(addresses) != len(set(addresses)) or any(address < 0 for address in addresses):
            raise ValueError("寄存器地址必须唯一且非负")
        self._definitions = {item.address: item for item in ordered}
        self._values = {item.address: 0 for item in ordered}
        self._lock = RLock()

    @staticmethod
    def _uint16(value: object, scale: float = 1.0) -> int:
        if isinstance(value, bool):
            numeric = int(value)
        elif isinstance(value, (int, float)):
            numeric = round(float(value) * scale)
        else:
            raise TypeError("寄存器值必须是整数、浮点数或布尔值")
        if not 0 <= numeric <= 0xFFFF:
            raise ValueError(f"寄存器值超出 16 位无符号范围: {numeric}")
        return int(numeric)

    def update_from_gateway_message(self, message: dict) -> None:
        """按寄存器名称读取 payload，并在一次锁内完成全部更新。"""
        payload = message.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("gateway_message.payload 必须是对象")
        updates: dict[int, int] = {}
        for address, definition in self._definitions.items():
            if definition.name not in payload:
                continue
            updates[address] = self._uint16(payload[definition.name], definition.scale)
        with self._lock:
            self._values.update(updates)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "address": address,
                    "name": definition.name,
                    "value": self._values[address],
                    "type": definition.data_type,
                    "scale": definition.scale,
                    "description": definition.description,
                }
                for address, definition in sorted(self._definitions.items())
            ]

    def read(self, address: int, count: int) -> list[int]:
        if count <= 0:
            raise RegisterAddressError("读取数量必须大于 0")
        requested = range(address, address + count)
        with self._lock:
            if any(item not in self._values for item in requested):
                raise RegisterAddressError(f"读取范围不存在: {address}..{address + count - 1}")
            return [self._values[item] for item in requested]

    def write(self, address: int, values: Iterable[object]) -> None:
        normalized = [self._uint16(value) for value in values]
        if not normalized:
            raise RegisterAddressError("写入值不能为空")
        requested = range(address, address + len(normalized))
        with self._lock:
            if any(item not in self._values for item in requested):
                raise RegisterAddressError(
                    f"写入范围不存在: {address}..{address + len(normalized) - 1}"
                )
            for item, value in zip(requested, normalized):
                self._values[item] = value
