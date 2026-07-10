"""仅支持 Holding Registers 的最小 Modbus TCP Mock Server。"""

from __future__ import annotations

import socketserver
import struct
import threading
from typing import ClassVar

from .modbus_registers import HoldingRegisterBank, RegisterAddressError


ILLEGAL_FUNCTION = 0x01
ILLEGAL_DATA_ADDRESS = 0x02
ILLEGAL_DATA_VALUE = 0x03


def _recv_exact(sock, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Modbus 连接提前关闭")
        chunks.extend(chunk)
    return bytes(chunks)


class _ModbusRequestHandler(socketserver.BaseRequestHandler):
    server: "ModbusTcpServer"

    def handle(self) -> None:
        self.request.settimeout(3.0)
        while True:
            try:
                header = _recv_exact(self.request, 7)
            except (ConnectionError, OSError):
                return
            transaction_id, protocol_id, length, unit_id = struct.unpack(">HHHB", header)
            if protocol_id != 0 or length < 2 or length > 260:
                return
            try:
                pdu = _recv_exact(self.request, length - 1)
            except (ConnectionError, OSError):
                return
            response_pdu = self.server.handle_pdu(pdu)
            response = struct.pack(">HHHB", transaction_id, 0, len(response_pdu) + 1, unit_id)
            self.request.sendall(response + response_pdu)


class ModbusTcpServer(socketserver.ThreadingTCPServer):
    """使用标准库 socketserver 的 Modbus TCP Mock。"""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16
    bank: ClassVar[HoldingRegisterBank]

    def __init__(self, host: str, port: int, bank: HoldingRegisterBank) -> None:
        self.bank = bank
        super().__init__((host, port), _ModbusRequestHandler)

    @staticmethod
    def _exception(function_code: int, exception_code: int) -> bytes:
        return bytes((function_code | 0x80, exception_code))

    def handle_pdu(self, pdu: bytes) -> bytes:
        if not pdu:
            return self._exception(0, ILLEGAL_DATA_VALUE)
        function_code = pdu[0]
        try:
            if function_code == 0x03:
                return self._read_holding_registers(pdu)
            if function_code == 0x06:
                return self._write_single_register(pdu)
            if function_code == 0x10:
                return self._write_multiple_registers(pdu)
            return self._exception(function_code, ILLEGAL_FUNCTION)
        except RegisterAddressError:
            return self._exception(function_code, ILLEGAL_DATA_ADDRESS)
        except (ValueError, TypeError, struct.error):
            return self._exception(function_code, ILLEGAL_DATA_VALUE)

    def _read_holding_registers(self, pdu: bytes) -> bytes:
        if len(pdu) != 5:
            raise ValueError("FC03 PDU 长度错误")
        address, count = struct.unpack(">HH", pdu[1:])
        if not 1 <= count <= 125:
            raise ValueError("FC03 数量超出范围")
        values = self.bank.read(address, count)
        payload = b"".join(struct.pack(">H", value) for value in values)
        return bytes((0x03, len(payload))) + payload

    def _write_single_register(self, pdu: bytes) -> bytes:
        if len(pdu) != 5:
            raise ValueError("FC06 PDU 长度错误")
        address, value = struct.unpack(">HH", pdu[1:])
        self.bank.write(address, [value])
        return pdu

    def _write_multiple_registers(self, pdu: bytes) -> bytes:
        if len(pdu) < 6:
            raise ValueError("FC16 PDU 长度错误")
        address, count, byte_count = struct.unpack(">HHB", pdu[1:6])
        if not 1 <= count <= 123 or byte_count != count * 2 or len(pdu) != 6 + byte_count:
            raise ValueError("FC16 数量或字节数错误")
        values = list(struct.unpack(f">{count}H", pdu[6:]))
        self.bank.write(address, values)
        return bytes((0x10,)) + struct.pack(">HH", address, count)


def start_modbus_server(
    host: str,
    port: int,
    bank: HoldingRegisterBank,
) -> tuple[ModbusTcpServer, threading.Thread]:
    server = ModbusTcpServer(host, port, bank)
    thread = threading.Thread(target=server.serve_forever, name="modbus-tcp-mock", daemon=True)
    thread.start()
    return server, thread
