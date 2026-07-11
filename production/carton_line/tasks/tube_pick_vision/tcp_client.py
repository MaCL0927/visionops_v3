#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-lived TCP client for ``*<utf8-json>#`` framed messages."""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable


class FramingError(ValueError):
    """A framed JSON message is malformed or exceeds limits."""


class StarHashJsonCodec:
    """Incremental parser for the VisionInterfacer star/hash framing."""

    def __init__(self, max_frame_bytes: int = 1024 * 1024) -> None:
        self.max_frame_bytes = max(1024, int(max_frame_bytes))
        self.buffer = bytearray()

    @staticmethod
    def encode(document: dict[str, Any]) -> bytes:
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return b"*" + body + b"#"

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        if data:
            self.buffer.extend(data)
        messages: list[dict[str, Any]] = []
        while True:
            start = self.buffer.find(b"*")
            if start < 0:
                if len(self.buffer) > self.max_frame_bytes:
                    self.buffer.clear()
                    raise FramingError("未找到消息起始符 *，缓冲区已超过限制")
                # Bytes before '*' can never become part of a valid frame.
                self.buffer.clear()
                break
            if start > 0:
                del self.buffer[:start]
            end = self.buffer.find(b"#", 1)
            if end < 0:
                if len(self.buffer) > self.max_frame_bytes:
                    self.buffer.clear()
                    raise FramingError("消息未找到结束符 #，帧超过大小限制")
                break
            raw = bytes(self.buffer[1:end])
            del self.buffer[: end + 1]
            if len(raw) > self.max_frame_bytes:
                raise FramingError("JSON 帧超过大小限制")
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FramingError(f"JSON 解析失败: {error}") from error
            if not isinstance(document, dict):
                raise FramingError("JSON 顶层必须是对象")
            messages.append(document)
        return messages


class ReconnectingJsonTcpClient:
    """Connect to the scheduler server, parse triggers and send responses."""

    def __init__(
        self,
        host: str,
        port: int,
        on_message: Callable[[dict[str, Any]], dict[str, Any] | None],
        on_state: Callable[[str, dict[str, Any]], None] | None = None,
        connect_timeout_s: float = 3.0,
        read_timeout_s: float = 1.0,
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 10.0,
        max_frame_bytes: int = 1024 * 1024,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.on_message = on_message
        self.on_state = on_state
        self.connect_timeout_s = max(0.1, float(connect_timeout_s))
        self.read_timeout_s = max(0.1, float(read_timeout_s))
        self.reconnect_initial_s = max(0.1, float(reconnect_initial_s))
        self.reconnect_max_s = max(self.reconnect_initial_s, float(reconnect_max_s))
        self.max_frame_bytes = max_frame_bytes
        self._socket_lock = threading.RLock()
        self._socket: socket.socket | None = None

    def _emit(self, state: str, **detail: Any) -> None:
        if self.on_state is not None:
            self.on_state(state, detail)

    def close(self) -> None:
        with self._socket_lock:
            sock = self._socket
            self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _connect(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        sock.settimeout(self.read_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Linux keepalive tuning is best effort because constants vary by platform.
        for name, value in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3)):
            option = getattr(socket, name, None)
            if option is not None:
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, option, value)
                except OSError:
                    pass
        with self._socket_lock:
            self._socket = sock
        return sock

    def run(self, stop_event: threading.Event) -> None:
        delay = self.reconnect_initial_s
        while not stop_event.is_set():
            sock: socket.socket | None = None
            try:
                self._emit("connecting", host=self.host, port=self.port)
                sock = self._connect()
                self._emit("connected", host=self.host, port=self.port)
                delay = self.reconnect_initial_s
                codec = StarHashJsonCodec(self.max_frame_bytes)
                while not stop_event.is_set():
                    try:
                        chunk = sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise ConnectionError("调度系统关闭了 TCP 连接")
                    for request in codec.feed(chunk):
                        response = self.on_message(request)
                        if response is not None:
                            sock.sendall(StarHashJsonCodec.encode(response))
            except (OSError, ConnectionError, FramingError) as error:
                self._emit("disconnected", error=str(error), retry_s=delay)
            except Exception as error:  # A task error must not terminate reconnect supervision.
                self._emit("client_error", error=f"{type(error).__name__}: {error}", retry_s=delay)
            finally:
                self.close()
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if stop_event.wait(delay):
                break
            delay = min(self.reconnect_max_s, delay * 2.0)
        self._emit("stopped")
