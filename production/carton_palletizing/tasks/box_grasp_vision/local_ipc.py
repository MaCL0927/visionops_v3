"""Low-overhead local IPC helpers for the box-grasp task."""
from __future__ import annotations

import mmap
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, List, Any
from urllib.parse import urlsplit

import numpy as np  # type: ignore

SHARED_DEPTH_MAGIC = 0x564F505344455031
SHARED_DEPTH_VERSION = 1
SHARED_DEPTH_HEADER_SIZE = 256
SHARED_DEPTH_STATE_RUNNING = 1
SHARED_DEPTH_PIXEL_UINT16_MM = 1
SHARED_DEPTH_HEADER = struct.Struct("<QIIQQQ" + "I" * 12 + "Q" * 5 + "d" * 4 + "Q" * 12)
assert SHARED_DEPTH_HEADER.size == SHARED_DEPTH_HEADER_SIZE


@dataclass(frozen=True)
class RawHttpResponse:
    body: bytes
    status_code: int
    headers: Mapping[str, str]
    connect_ms: float
    send_ms: float
    headers_wait_ms: float
    body_read_ms: float
    total_ms: float
    transport: str = "raw_socket"


class RawLocalHttpClient:
    def __init__(self, timeout_s: float, max_response_bytes: int) -> None:
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)

    @staticmethod
    def supports(url: str) -> bool:
        parsed = urlsplit(url)
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}

    def request(self, method: str, url: str, body: Optional[bytes] = None) -> RawHttpResponse:
        parsed = urlsplit(url)
        if not self.supports(url):
            raise ValueError("raw local HTTP only supports localhost http URLs")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        payload = body or b""
        request_lines = [
            f"{method} {target} HTTP/1.1",
            f"Host: {host}:{port}",
            "Accept: application/json,image/jpeg,image/png,*/*",
            "User-Agent: visionops-box-grasp-raw/1.0",
            "Connection: close",
            f"Content-Length: {len(payload)}",
        ]
        if body is not None:
            request_lines.append("Content-Type: application/json")
        request = ("\r\n".join(request_lines) + "\r\n\r\n").encode("ascii") + payload

        started = time.perf_counter()
        sock: Optional[socket.socket] = None
        try:
            connect_started = time.perf_counter()
            sock = socket.create_connection((host, port), timeout=self.timeout_s)
            sock.settimeout(self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connected = time.perf_counter()
            sock.sendall(request)
            sent = time.perf_counter()

            received = bytearray()
            header_end = -1
            while header_end < 0:
                chunk = sock.recv(8192)
                if not chunk:
                    raise ConnectionError("upstream closed before response headers")
                received.extend(chunk)
                if len(received) > 128 * 1024:
                    raise ConnectionError("upstream response headers too large")
                header_end = received.find(b"\r\n\r\n")
            headers_received = time.perf_counter()
            header_raw = bytes(received[:header_end]).decode("iso-8859-1")
            body_buffer = bytearray(received[header_end + 4 :])
            lines = header_raw.split("\r\n")
            parts = lines[0].split(" ", 2)
            if len(parts) < 2:
                raise ConnectionError("invalid upstream status line")
            status_code = int(parts[1])
            headers: Dict[str, str] = {}
            for line in lines[1:]:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
            if "transfer-encoding" in headers and headers["transfer-encoding"].lower() != "identity":
                raise ConnectionError("chunked upstream responses are not supported")
            content_length = int(headers.get("content-length", len(body_buffer)))
            if content_length < 0 or content_length > self.max_response_bytes:
                raise ConnectionError("upstream response exceeds size limit")
            while len(body_buffer) < content_length:
                chunk = sock.recv(min(65536, content_length - len(body_buffer)))
                if not chunk:
                    raise ConnectionError("upstream closed before complete response body")
                body_buffer.extend(chunk)
            finished = time.perf_counter()
            return RawHttpResponse(
                body=bytes(body_buffer[:content_length]),
                status_code=status_code,
                headers=headers,
                connect_ms=(connected - connect_started) * 1000.0,
                send_ms=(sent - connected) * 1000.0,
                headers_wait_ms=(headers_received - sent) * 1000.0,
                body_read_ms=(finished - headers_received) * 1000.0,
                total_ms=(finished - started) * 1000.0,
            )
        finally:
            if sock is not None:
                sock.close()


class SharedDepthReader:
    """Read the latest D2C uint16 depth frame without copying the whole image."""

    def __init__(self, name: str, max_age_ms: int) -> None:
        self.name = str(name)
        self.max_age_ms = max(1, int(max_age_ms))
        self.path = "/dev/shm/" + self.name.lstrip("/")
        self._fd = -1
        self._mapping: Optional[mmap.mmap] = None
        self._mapping_size = 0
        self.retry_count = 0
        self.last_error = ""

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._mapping_size = 0

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "name": self.name,
            "path": self.path,
            "mapped": self._mapping is not None,
            "mapping_size": self._mapping_size,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
        }

    def _open(self) -> mmap.mmap:
        try:
            size = os.stat(self.path).st_size
            if self._mapping is not None and size == self._mapping_size:
                return self._mapping
            self.close()
            self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            self._mapping = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
            self._mapping_size = size
            return self._mapping
        except OSError as error:
            self.close()
            self.last_error = str(error)
            raise

    @staticmethod
    def _header(mapping: mmap.mmap) -> Tuple[Any, ...]:
        return SHARED_DEPTH_HEADER.unpack_from(mapping, 0)

    def sample_deproject(
        self,
        points: Sequence[Sequence[float]],
        image_width: int,
        image_height: int,
        radius_px: int,
        percentile: float,
        min_valid_pixels: int,
        min_depth_mm: int,
        max_depth_mm: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        started = time.perf_counter()
        mapping = self._open()
        for attempt in range(4):
            h = self._header(mapping)
            (magic, version, header_size, total_size, frame_capacity, frame_bytes,
             width, height, stride_bytes, pixel_format, buffer_count, state,
             active_buffer, calibration_ready, aligned_to_color, flip_horizontal,
             flip_vertical, _reserved0, sequence, timestamp_ms, _writer_pid,
             publish_count, _dropped_count, fx, fy, cx, cy, *_reserved) = h
            if magic != SHARED_DEPTH_MAGIC or version != SHARED_DEPTH_VERSION or header_size != SHARED_DEPTH_HEADER_SIZE:
                raise ValueError("shared depth header is incompatible")
            if total_size > self._mapping_size or buffer_count != 2 or pixel_format != SHARED_DEPTH_PIXEL_UINT16_MM:
                raise ValueError("shared depth mapping is invalid")
            if state != SHARED_DEPTH_STATE_RUNNING or not calibration_ready or not aligned_to_color:
                raise ValueError("shared depth is not ready")
            age_ms = int(time.time() * 1000) - int(timestamp_ms)
            if age_ms < 0 or age_ms > self.max_age_ms:
                raise ValueError(f"shared depth is stale: {age_ms}ms")
            if width <= 0 or height <= 0 or stride_bytes < width * 2 or fx <= 0 or fy <= 0:
                raise ValueError("shared depth dimensions/intrinsics are invalid")
            offset = SHARED_DEPTH_HEADER_SIZE + int(frame_capacity) * (int(active_buffer) % 2)
            depth = np.ndarray(
                shape=(int(height), int(width)),
                dtype="<u2",
                buffer=mapping,
                offset=offset,
                strides=(int(stride_bytes), 2),
            )
            sx = float(width) / max(1.0, float(image_width))
            sy = float(height) / max(1.0, float(image_height))
            output: List[Dict[str, Any]] = []
            radius_x = max(0, int(round(radius_px * sx)))
            radius_y = max(0, int(round(radius_px * sy)))
            for point in points:
                sample_u, sample_v, project_u, project_v = [float(v) for v in point[:4]]
                px = int(round(sample_u * sx))
                py = int(round(sample_v * sy))
                x0, x1 = max(0, px - radius_x), min(int(width), px + radius_x + 1)
                y0, y1 = max(0, py - radius_y), min(int(height), py + radius_y + 1)
                values = depth[y0:y1, x0:x1].reshape(-1)
                valid = values[(values >= int(min_depth_mm)) & (values <= int(max_depth_mm))]
                depth_valid = int(valid.size) >= int(min_valid_pixels)
                z = float(np.percentile(valid, percentile)) if depth_valid else 0.0
                project_x = float(project_u) * sx
                project_y = float(project_v) * sy
                position = [
                    (project_x - float(cx)) * z / float(fx),
                    (project_y - float(cy)) * z / float(fy),
                    z,
                ] if depth_valid else [0.0, 0.0, 0.0]
                output.append({
                    "depth_valid": depth_valid,
                    "depth_mm": int(round(z)) if depth_valid else 0,
                    "sample_px": [px, py],
                    "valid_pixels": int(valid.size),
                    "position_camera": position,
                    "project_valid": depth_valid,
                })
            sequence_after = self._header(mapping)[18]
            if sequence_after == sequence:
                elapsed = (time.perf_counter() - started) * 1000.0
                self.last_error = ""
                return output, {
                    "ok": True,
                    "mode": "shared_depth",
                    "depth_age_ms": age_ms,
                    "depth_sequence": int(sequence),
                    "publish_count": int(publish_count),
                    "sample_ms": elapsed,
                    "shared_memory_name": self.name,
                    "retry_count": self.retry_count,
                    "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                    "flip_horizontal": bool(flip_horizontal),
                    "flip_vertical": bool(flip_vertical),
                }
            self.retry_count += 1
        raise RuntimeError("shared depth changed repeatedly while sampling")
