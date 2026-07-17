#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small dependency-free RFC6455 WebSocket server for box_grasp_vision.

Only the protocol features needed by the robot integration are implemented:
HTTP upgrade, text JSON frames, native ping/pong, close frames, masking checks,
payload limits, and multiple long-lived clients.
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlsplit


_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketProtocolError(ConnectionError):
    pass


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        data = sock.recv(count - len(chunks))
        if not data:
            raise ConnectionError("WebSocket 客户端已断开")
        chunks.extend(data)
    return bytes(chunks)


def _server_frame(opcode: int, payload: bytes = b"") -> bytes:
    first = 0x80 | (opcode & 0x0F)
    size = len(payload)
    if size < 126:
        header = bytes((first, size))
    elif size <= 0xFFFF:
        header = bytes((first, 126)) + struct.pack("!H", size)
    else:
        header = bytes((first, 127)) + struct.pack("!Q", size)
    return header + payload


@dataclass(frozen=True)
class IncomingFrame:
    opcode: int
    fin: bool
    payload: bytes


def _read_client_frame(sock: socket.socket, max_payload_bytes: int) -> IncomingFrame:
    header = _recv_exact(sock, 2)
    first, second = header[0], header[1]
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if not masked:
        raise WebSocketProtocolError("客户端 WebSocket 帧必须设置 mask")
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if length > max_payload_bytes:
        raise WebSocketProtocolError(f"WebSocket 帧超过限制: {length} > {max_payload_bytes}")
    mask = _recv_exact(sock, 4)
    payload = bytearray(_recv_exact(sock, int(length)))
    for index in range(len(payload)):
        payload[index] ^= mask[index % 4]
    return IncomingFrame(opcode=opcode, fin=fin, payload=bytes(payload))


class WebSocketSession:
    def __init__(self, sock: socket.socket, address: tuple[str, int], path: str) -> None:
        self.socket = sock
        self.address = address
        self.path = path
        self.connected_at = time.time()
        self.last_seen_at = self.connected_at
        self.send_lock = threading.Lock()
        self.closed = threading.Event()

    @property
    def client_id(self) -> str:
        return f"{self.address[0]}:{self.address[1]}"

    def send_frame(self, opcode: int, payload: bytes = b"") -> None:
        if self.closed.is_set():
            raise ConnectionError("WebSocket session 已关闭")
        frame = _server_frame(opcode, payload)
        with self.send_lock:
            self.socket.sendall(frame)

    def send_json(self, document: Mapping[str, Any]) -> None:
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_frame(0x1, body)

    def send_pong(self, payload: bytes) -> None:
        self.send_frame(0xA, payload)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            payload = struct.pack("!H", code) + reason.encode("utf-8")[:120]
            with self.send_lock:
                self.socket.sendall(_server_frame(0x8, payload))
        except OSError:
            pass
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.socket.close()
        except OSError:
            pass


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class WebSocketJsonServer:
    def __init__(
        self,
        host: str,
        port: int,
        path: str,
        on_json: Callable[[WebSocketSession, dict[str, Any]], None],
        on_connect: Callable[[WebSocketSession], None] | None = None,
        on_disconnect: Callable[[WebSocketSession], None] | None = None,
        token: str = "",
        max_clients: int = 4,
        max_payload_bytes: int = 1024 * 1024,
        read_timeout_s: float = 30.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.path = path if path.startswith("/") else "/" + path
        self.on_json = on_json
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.token = token
        self.max_clients = max(1, int(max_clients))
        self.max_payload_bytes = max(1024, int(max_payload_bytes))
        self.read_timeout_s = max(1.0, float(read_timeout_s))
        self.sessions: set[WebSocketSession] = set()
        self.sessions_lock = threading.RLock()
        self.server: _ThreadingServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                owner._handle_socket(self.request, self.client_address)

        self.server = _ThreadingServer((self.host, self.port), Handler)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, name="box-grasp-websocket", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.sessions_lock:
            sessions = list(self.sessions)
        for session in sessions:
            session.close(1001, "server stopping")
        server, self.server = self.server, None
        thread, self.thread = self.thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def client_count(self) -> int:
        with self.sessions_lock:
            return len(self.sessions)

    def client_snapshot(self) -> list[dict[str, Any]]:
        with self.sessions_lock:
            sessions = list(self.sessions)
        return [
            {
                "client_id": session.client_id,
                "path": session.path,
                "connected_at": session.connected_at,
                "last_seen_at": session.last_seen_at,
            }
            for session in sessions
        ]

    def broadcast_json(self, document: Mapping[str, Any]) -> int:
        with self.sessions_lock:
            sessions = list(self.sessions)
        sent = 0
        for session in sessions:
            try:
                session.send_json(document)
                sent += 1
            except (OSError, ConnectionError):
                session.close(1006, "send failed")
        return sent

    def _read_handshake(self, sock: socket.socket) -> tuple[str, dict[str, str]]:
        buffer = bytearray()
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket 握手连接已关闭")
            buffer.extend(chunk)
            if len(buffer) > 65536:
                raise WebSocketProtocolError("WebSocket 握手头过大")
        header_text = bytes(buffer).split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
        lines = header_text.split("\r\n")
        parts = lines[0].split()
        if len(parts) < 3 or parts[0] != "GET":
            raise WebSocketProtocolError("仅支持 WebSocket GET Upgrade")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return parts[1], headers

    @staticmethod
    def _http_error(sock: socket.socket, code: int, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
        reason = {400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 503: "Service Unavailable"}.get(code, "Error")
        response = (
            f"HTTP/1.1 {code} {reason}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body
        sock.sendall(response)

    def _upgrade(self, sock: socket.socket, address: tuple[str, int]) -> WebSocketSession | None:
        request_target, headers = self._read_handshake(sock)
        parsed = urlsplit(request_target)
        if parsed.path != self.path:
            self._http_error(sock, 404, f"WebSocket path 必须为 {self.path}")
            return None
        if self.token:
            supplied = (parse_qs(parsed.query).get("token") or [""])[0]
            if supplied != self.token:
                self._http_error(sock, 401, "token invalid")
                return None
        if headers.get("upgrade", "").lower() != "websocket":
            self._http_error(sock, 400, "missing Upgrade: websocket")
            return None
        key = headers.get("sec-websocket-key", "")
        if not key:
            self._http_error(sock, 400, "missing Sec-WebSocket-Key")
            return None
        with self.sessions_lock:
            if len(self.sessions) >= self.max_clients:
                self._http_error(sock, 503, "too many websocket clients")
                return None
        accept = base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        sock.sendall(response.encode("ascii"))
        return WebSocketSession(sock, address, request_target)

    def _handle_socket(self, sock: socket.socket, address: tuple[str, int]) -> None:
        session: WebSocketSession | None = None
        try:
            sock.settimeout(self.read_timeout_s)
            session = self._upgrade(sock, address)
            if session is None:
                return
            with self.sessions_lock:
                self.sessions.add(session)
            if self.on_connect is not None:
                self.on_connect(session)

            fragmented_opcode: int | None = None
            fragmented = bytearray()
            while not session.closed.is_set():
                try:
                    frame = _read_client_frame(sock, self.max_payload_bytes)
                except socket.timeout:
                    # Idle clients are kept alive; robot side is expected to send native ping.
                    continue
                session.last_seen_at = time.time()
                if frame.opcode == 0x8:
                    session.close(1000, "peer close")
                    break
                if frame.opcode == 0x9:
                    session.send_pong(frame.payload)
                    continue
                if frame.opcode == 0xA:
                    continue
                if frame.opcode in {0x1, 0x2}:
                    if frame.fin:
                        payload = frame.payload
                        opcode = frame.opcode
                    else:
                        fragmented_opcode = frame.opcode
                        fragmented = bytearray(frame.payload)
                        continue
                elif frame.opcode == 0x0 and fragmented_opcode is not None:
                    fragmented.extend(frame.payload)
                    if len(fragmented) > self.max_payload_bytes:
                        raise WebSocketProtocolError("WebSocket 分片消息超过限制")
                    if not frame.fin:
                        continue
                    opcode = fragmented_opcode
                    payload = bytes(fragmented)
                    fragmented_opcode = None
                    fragmented.clear()
                else:
                    raise WebSocketProtocolError(f"不支持的 WebSocket opcode={frame.opcode}")

                if opcode != 0x1:
                    raise WebSocketProtocolError("机器人到盒子只允许 JSON 文本帧")
                try:
                    document = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise WebSocketProtocolError("WebSocket 文本帧不是有效 JSON") from error
                if not isinstance(document, dict):
                    raise WebSocketProtocolError("WebSocket JSON 顶层必须是对象")
                self.on_json(session, document)
        except (ConnectionError, OSError, WebSocketProtocolError):
            pass
        finally:
            if session is not None:
                with self.sessions_lock:
                    self.sessions.discard(session)
                if self.on_disconnect is not None:
                    self.on_disconnect(session)
                session.close(1001, "session ended")
            else:
                try:
                    sock.close()
                except OSError:
                    pass
