#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free robot-side WebSocket client for integration testing."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time
from urllib.parse import urlsplit


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("server disconnected")
        data.extend(chunk)
    return bytes(data)


def _client_frame(opcode: int, payload: bytes) -> bytes:
    mask = os.urandom(4)
    size = len(payload)
    first = 0x80 | opcode
    if size < 126:
        header = bytes((first, 0x80 | size))
    elif size <= 0xFFFF:
        header = bytes((first, 0x80 | 126)) + struct.pack("!H", size)
    else:
        header = bytes((first, 0x80 | 127)) + struct.pack("!Q", size)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return header + mask + masked


def _read_server_frame(sock: socket.socket):
    first, second = _recv_exact(sock, 2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    return opcode, _recv_exact(sock, int(length))


class Client:
    def __init__(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "ws" or not parsed.hostname:
            raise ValueError("URL must be ws://host:port/path")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self.sock.settimeout(30.0)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(self.sock.recv(4096))
        if not response.startswith(b"HTTP/1.1 101"):
            raise ConnectionError(response.decode("utf-8", errors="replace"))
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected}".lower().encode() not in bytes(response).lower():
            raise ConnectionError("invalid Sec-WebSocket-Accept")
        self.stop_event = threading.Event()

    def send_json(self, document) -> None:
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.sock.sendall(_client_frame(0x1, payload))

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.sock.sendall(_client_frame(0x8, struct.pack("!H", 1000)))
        except OSError:
            pass
        self.sock.close()

    def run(self) -> None:
        def ping_loop() -> None:
            while not self.stop_event.wait(10.0):
                try:
                    self.sock.sendall(_client_frame(0x9, b"vision-ping"))
                except OSError:
                    return

        threading.Thread(target=ping_loop, daemon=True).start()
        while not self.stop_event.is_set():
            opcode, payload = _read_server_frame(self.sock)
            if opcode == 0x1:
                try:
                    print(json.dumps(json.loads(payload.decode("utf-8")), ensure_ascii=False, indent=2))
                except Exception:
                    print(payload.decode("utf-8", errors="replace"))
            elif opcode == 0x9:
                self.sock.sendall(_client_frame(0xA, payload))
            elif opcode == 0x8:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock robot WebSocket client")
    parser.add_argument("--url", default="ws://127.0.0.1:9001/vision")
    args = parser.parse_args()
    client = Client(args.url)
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()
    request_id = 1
    try:
        while True:
            command = input("输入 start/stop/trigger/q（直接回车=trigger）：").strip().lower() or "trigger"
            if command == "q":
                break
            if command not in {"start", "stop", "trigger"}:
                print("unsupported command")
                continue
            document = {"type": "control", "command": command, "request_id": request_id}
            client.send_json(document)
            request_id += 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
