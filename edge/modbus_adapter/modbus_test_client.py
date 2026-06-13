#!/usr/bin/env python3
"""用于 M5 验证的最小 Modbus TCP 测试客户端。"""

from __future__ import annotations

import argparse
import socket
import struct
from typing import Sequence


class ModbusClientError(RuntimeError):
    """Modbus 响应无效或返回 exception。"""


class ModbusTestClient:
    def __init__(self, host: str, port: int, unit_id: int = 1, timeout_s: float = 3.0) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout_s = timeout_s
        self._transaction_id = 0

    def _exchange(self, pdu: bytes) -> bytes:
        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        request = struct.pack(">HHHB", self._transaction_id, 0, len(pdu) + 1, self.unit_id) + pdu
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(request)
            header = self._recv_exact(sock, 7)
            transaction_id, protocol_id, length, _unit_id = struct.unpack(">HHHB", header)
            if transaction_id != self._transaction_id or protocol_id != 0 or length < 2:
                raise ModbusClientError("Modbus MBAP 响应无效")
            response_pdu = self._recv_exact(sock, length - 1)
        if response_pdu[0] & 0x80:
            code = response_pdu[1] if len(response_pdu) > 1 else -1
            raise ModbusClientError(f"Modbus exception code={code}")
        return response_pdu

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ModbusClientError("Modbus 连接提前关闭")
            data.extend(chunk)
        return bytes(data)

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        response = self._exchange(bytes((0x03,)) + struct.pack(">HH", address, count))
        if response[0] != 0x03 or len(response) < 2 or response[1] != count * 2:
            raise ModbusClientError("FC03 响应长度错误")
        return list(struct.unpack(f">{count}H", response[2:]))

    def write_single_register(self, address: int, value: int) -> None:
        response = self._exchange(bytes((0x06,)) + struct.pack(">HH", address, value))
        if response != bytes((0x06,)) + struct.pack(">HH", address, value):
            raise ModbusClientError("FC06 响应内容错误")

    def write_multiple_registers(self, address: int, values: list[int]) -> None:
        payload = b"".join(struct.pack(">H", value) for value in values)
        request = bytes((0x10,)) + struct.pack(">HHB", address, len(values), len(payload)) + payload
        response = self._exchange(request)
        expected = bytes((0x10,)) + struct.pack(">HH", address, len(values))
        if response != expected:
            raise ModbusClientError("FC16 响应内容错误")


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("端口必须位于 1 到 65535")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps Modbus TCP Mock 测试客户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=1502)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--read-start", type=int, default=0)
    parser.add_argument("--read-count", type=int, default=20)
    parser.add_argument("--write-address", type=int)
    parser.add_argument("--write-value", type=int)
    parser.add_argument("--print-registers", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = ModbusTestClient(args.host, args.port, args.unit_id)
    if (args.write_address is None) != (args.write_value is None):
        build_parser().error("write-address 和 write-value 必须同时提供")
    if args.write_address is not None:
        client.write_single_register(args.write_address, args.write_value)
    values = client.read_holding_registers(args.read_start, args.read_count)
    if args.print_registers:
        for offset, value in enumerate(values):
            print(f"{args.read_start + offset}: {value}")
    else:
        print(" ".join(str(value) for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
