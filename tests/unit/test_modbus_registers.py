"""Holding Register 内存库测试。"""

from __future__ import annotations

import pytest

from edge.gateway_adapter.result_to_gateway import inference_result_to_gateway_message
from edge.modbus_adapter.modbus_registers import HoldingRegisterBank, RegisterAddressError


def gateway_message() -> dict:
    result = {
        "frame_id": "frame-12",
        "result_id": "result-34",
        "device_id": "example-edge",
        "trace_id": "trace-1",
        "status": "ok",
        "timing": {"inference_ms": 10.4, "total_ms": 14.6},
        "detections": [
            {
                "score": 0.8,
                "bbox_xyxy": [100, 200, 300, 400],
                "center_xy": [200, 300],
            }
        ],
    }
    return inference_result_to_gateway_message(result, "generic_mock", 5, 1)


def test_register_bank_initializes_default_map() -> None:
    bank = HoldingRegisterBank()
    snapshot = bank.snapshot()
    assert len(snapshot) == 20
    assert snapshot[0]["name"] == "heartbeat"
    assert [item["value"] for item in snapshot] == [0] * 20


def test_update_from_gateway_message_updates_registers() -> None:
    bank = HoldingRegisterBank()
    bank.update_from_gateway_message(gateway_message())
    values = bank.read(0, 20)
    assert values[0] == 1
    assert values[1] == 5
    assert values[7] == 1
    assert values[8] == 800
    assert values[9:15] == [200, 300, 100, 200, 300, 400]
    assert values[15] == 10
    assert values[16] == 15


def test_read_and_write_are_supported() -> None:
    bank = HoldingRegisterBank()
    bank.write(0, [12, 34, 56])
    assert bank.read(0, 3) == [12, 34, 56]


def test_invalid_address_raises() -> None:
    bank = HoldingRegisterBank()
    with pytest.raises(RegisterAddressError):
        bank.read(19, 2)
    with pytest.raises(RegisterAddressError):
        bank.write(20, [1])


def test_values_must_fit_uint16() -> None:
    bank = HoldingRegisterBank()
    with pytest.raises(ValueError):
        bank.write(0, [65536])
    with pytest.raises(ValueError):
        bank.write(0, [-1])
    with pytest.raises(ValueError):
        bank.write(0, [100000.5])
