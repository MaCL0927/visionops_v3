"""Gateway 消息的公共构造与稳定编码工具。"""

from __future__ import annotations

import re
import time
import zlib
from typing import Any


def timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


def stable_u16(value: Any) -> int:
    """将标识稳定映射到 0 到 65535，优先使用字符串中的数字。"""
    text = str(value or "")
    digits = re.findall(r"\d+", text)
    if digits:
        return int(digits[-1]) & 0xFFFF
    return zlib.crc32(text.encode("utf-8")) & 0xFFFF


def numeric_code(value: Any) -> int:
    """将业务代码转换为 16 位整数，不依赖进程级随机 hash。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value & 0xFFFF
    if isinstance(value, float) and value.is_integer():
        return int(value) & 0xFFFF
    return stable_u16(value)


def make_message_id(sequence: int, result_id: str) -> str:
    return f"gateway-message-{sequence:08d}-{stable_u16(result_id):05d}"


def make_error_document(
    *,
    device_id: str,
    component: str,
    code: str,
    message: str,
    recoverable: bool,
    detail: Any = None,
) -> dict[str, Any]:
    now = timestamp_ms()
    return {
        "schema_version": "1.0",
        "message_type": "gateway_error",
        "device_id": device_id,
        "component": component,
        "timestamp_ms": now,
        "trace_id": f"trace-gateway-error-{now}",
        "source": "gateway:http_api",
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "detail": detail,
            "recoverable": recoverable,
        },
    }
