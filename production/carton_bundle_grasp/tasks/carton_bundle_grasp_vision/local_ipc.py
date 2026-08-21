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
    """Read one stable D2C ROI snapshot, then sample/deproject it locally.

    M41.2 deliberately keeps the mmap consistency window short: only the ROI
    copy must see one stable published sequence.  All percentile sampling and
    XYZ calculations run afterwards on the private NumPy snapshot, so the 30 Hz
    camera producer may continue publishing without forcing retries.
    """

    SNAPSHOT_ATTEMPTS = 4

    def __init__(self, name: str, max_age_ms: int) -> None:
        self.name = str(name)
        self.max_age_ms = max(1, int(max_age_ms))
        self.path = "/dev/shm/" + self.name.lstrip("/")
        self._fd = -1
        self._mapping: Optional[mmap.mmap] = None
        self._mapping_size = 0
        self.retry_count = 0
        self.last_error = ""
        self.last_snapshot_roi = [0, 0, 0, 0]
        self.last_snapshot_copy_ms = 0.0
        self.last_vectorized_sample_ms = 0.0

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
            "last_snapshot_roi": list(self.last_snapshot_roi),
            "last_snapshot_copy_ms": round(float(self.last_snapshot_copy_ms), 3),
            "last_vectorized_sample_ms": round(float(self.last_vectorized_sample_ms), 3),
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

    def _decode_header(self, mapping: mmap.mmap) -> Dict[str, Any]:
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
            raise ValueError("shared depth is stale: {}ms".format(age_ms))
        if width <= 0 or height <= 0 or stride_bytes < width * 2 or fx <= 0 or fy <= 0:
            raise ValueError("shared depth dimensions/intrinsics are invalid")
        return {
            "total_size": int(total_size),
            "frame_capacity": int(frame_capacity),
            "frame_bytes": int(frame_bytes),
            "width": int(width),
            "height": int(height),
            "stride_bytes": int(stride_bytes),
            "active_buffer": int(active_buffer) % 2,
            "sequence": int(sequence),
            "timestamp_ms": int(timestamp_ms),
            "age_ms": int(age_ms),
            "publish_count": int(publish_count),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "flip_horizontal": bool(flip_horizontal),
            "flip_vertical": bool(flip_vertical),
        }

    @staticmethod
    def _map_coordinate(values: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
        if target_size <= 1:
            return np.zeros_like(values, dtype=np.float64)
        if source_size <= 1:
            return np.clip(np.rint(values), 0.0, float(target_size - 1))
        scale = float(target_size - 1) / float(source_size - 1)
        return np.clip(values.astype(np.float64, copy=False) * scale, 0.0, float(target_size - 1))

    @classmethod
    def _map_pixel(cls, values: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
        mapped = cls._map_coordinate(values, source_size, target_size)
        # Inputs are non-negative image coordinates.  np.floor(x+0.5) matches
        # C++ lround for these values more closely than bankers rounding.
        return np.floor(mapped + 0.5).astype(np.int32)

    def read_geometry_context(self, image_width: int, image_height: int) -> Dict[str, Any]:
        """Read a stable header-only camera geometry context.

        This remains available even if the ROI sampling path falls back to HTTP,
        allowing M41.2 corner rays to stay intrinsics-based without another
        depth/deprojection request.
        """
        mapping = self._open()
        for _attempt in range(self.SNAPSHOT_ATTEMPTS):
            first = self._decode_header(mapping)
            second = self._decode_header(mapping)
            if first["sequence"] == second["sequence"] and first["active_buffer"] == second["active_buffer"]:
                self.last_error = ""
                return {
                    "intrinsics": {
                        "fx": first["fx"], "fy": first["fy"],
                        "cx": first["cx"], "cy": first["cy"],
                    },
                    "depth_width": first["width"],
                    "depth_height": first["height"],
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "flip_horizontal": first["flip_horizontal"],
                    "flip_vertical": first["flip_vertical"],
                    "depth_sequence": first["sequence"],
                    "depth_age_ms": first["age_ms"],
                }
            self.retry_count += 1
        raise RuntimeError("shared depth header changed repeatedly")

    @staticmethod
    def _vectorized_depths(
        snapshot: np.ndarray,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        roi_x0: int,
        roi_y0: int,
        radius_x: int,
        radius_y: int,
        percentile: float,
        min_valid_pixels: int,
        min_depth_mm: int,
        max_depth_mm: int,
        depth_width: int,
        depth_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-point robust depth and valid-pixel counts without Python loops."""
        dx_grid, dy_grid = np.meshgrid(
            np.arange(-radius_x, radius_x + 1, dtype=np.int32),
            np.arange(-radius_y, radius_y + 1, dtype=np.int32),
        )
        dx = dx_grid.reshape(-1)
        dy = dy_grid.reshape(-1)
        xs_global = sample_x[:, None] + dx[None, :]
        ys_global = sample_y[:, None] + dy[None, :]
        in_bounds = (
            (xs_global >= 0) & (xs_global < int(depth_width)) &
            (ys_global >= 0) & (ys_global < int(depth_height))
        )
        xs = np.clip(xs_global - int(roi_x0), 0, snapshot.shape[1] - 1)
        ys = np.clip(ys_global - int(roi_y0), 0, snapshot.shape[0] - 1)
        values = snapshot[ys, xs]
        valid = in_bounds & (values >= int(min_depth_mm)) & (values <= int(max_depth_mm))
        valid_counts = np.sum(valid, axis=1, dtype=np.int32)

        # Sort one compact N x K matrix (normally 96 x 25). Invalid entries are
        # moved behind all valid uint16 depth values, then percentile indices are
        # gathered row-wise. This reproduces the Bridge's linear percentile rule
        # without 96 separate np.percentile() calls.
        sentinel = np.uint32(65536)
        sortable = np.where(valid, values.astype(np.uint32, copy=False), sentinel)
        ordered = np.sort(sortable, axis=1)
        z = np.zeros(sample_x.shape[0], dtype=np.float64)
        valid_rows = np.flatnonzero(valid_counts >= int(min_valid_pixels))
        if valid_rows.size:
            counts = valid_counts[valid_rows].astype(np.float64)
            positions = (float(percentile) / 100.0) * (counts - 1.0)
            lower = np.floor(positions).astype(np.int32)
            upper = np.ceil(positions).astype(np.int32)
            fraction = positions - lower.astype(np.float64)
            low_values = ordered[valid_rows, lower].astype(np.float64)
            high_values = ordered[valid_rows, upper].astype(np.float64)
            interpolated = low_values * (1.0 - fraction) + high_values * fraction
            z[valid_rows] = np.floor(interpolated + 0.5)
        return z, valid_counts

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
        if not points:
            raise ValueError("shared depth sampling requires at least one point")
        mapping = self._open()
        point_array = np.asarray([list(point[:4]) for point in points], dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] < 4 or not np.all(np.isfinite(point_array)):
            raise ValueError("shared depth sample points are invalid")

        snapshot = None  # type: Optional[np.ndarray]
        stable = None  # type: Optional[Dict[str, Any]]
        sample_x = sample_y = None
        roi_x0 = roi_y0 = roi_x1 = roi_y1 = 0
        snapshot_copy_ms = 0.0
        snapshot_attempts = 0

        for attempt in range(self.SNAPSHOT_ATTEMPTS):
            snapshot_attempts = attempt + 1
            header = self._decode_header(mapping)
            width = int(header["width"])
            height = int(header["height"])
            sample_x = self._map_pixel(point_array[:, 0], int(image_width), width)
            sample_y = self._map_pixel(point_array[:, 1], int(image_height), height)
            radius_x = max(0, int(round(float(radius_px) * float(max(1, width - 1)) / float(max(1, image_width - 1)))))
            radius_y = max(0, int(round(float(radius_px) * float(max(1, height - 1)) / float(max(1, image_height - 1)))))
            roi_x0 = max(0, int(np.min(sample_x)) - radius_x)
            roi_y0 = max(0, int(np.min(sample_y)) - radius_y)
            roi_x1 = min(width, int(np.max(sample_x)) + radius_x + 1)
            roi_y1 = min(height, int(np.max(sample_y)) + radius_y + 1)
            if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
                raise ValueError("shared depth ROI is empty")

            offset = SHARED_DEPTH_HEADER_SIZE + int(header["frame_capacity"]) * int(header["active_buffer"])
            depth_view = np.ndarray(
                shape=(height, width),
                dtype="<u2",
                buffer=mapping,
                offset=offset,
                strides=(int(header["stride_bytes"]), 2),
            )
            copy_started = time.perf_counter()
            candidate = depth_view[roi_y0:roi_y1, roi_x0:roi_x1].copy()
            copy_ms = (time.perf_counter() - copy_started) * 1000.0
            after = self._decode_header(mapping)
            if after["sequence"] == header["sequence"] and after["active_buffer"] == header["active_buffer"]:
                snapshot = candidate
                stable = header
                snapshot_copy_ms = copy_ms
                break
            self.retry_count += 1

        if snapshot is None or stable is None or sample_x is None or sample_y is None:
            raise RuntimeError("shared depth changed repeatedly while taking ROI snapshot")

        sampling_started = time.perf_counter()
        radius_x = max(0, int(round(float(radius_px) * float(max(1, stable["width"] - 1)) / float(max(1, image_width - 1)))))
        radius_y = max(0, int(round(float(radius_px) * float(max(1, stable["height"] - 1)) / float(max(1, image_height - 1)))))
        z_values, valid_counts = self._vectorized_depths(
            snapshot,
            sample_x,
            sample_y,
            roi_x0,
            roi_y0,
            radius_x,
            radius_y,
            percentile,
            min_valid_pixels,
            min_depth_mm,
            max_depth_mm,
            int(stable["width"]),
            int(stable["height"]),
        )
        vectorized_sample_ms = (time.perf_counter() - sampling_started) * 1000.0

        deproject_started = time.perf_counter()
        project_x = self._map_coordinate(point_array[:, 2], int(image_width), int(stable["width"]))
        project_y = self._map_coordinate(point_array[:, 3], int(image_height), int(stable["height"]))
        if bool(stable["flip_horizontal"]):
            project_x = float(stable["width"] - 1) - project_x
        if bool(stable["flip_vertical"]):
            project_y = float(stable["height"] - 1) - project_y
        depth_valid = valid_counts >= int(min_valid_pixels)
        x_values = (project_x - float(stable["cx"])) * z_values / float(stable["fx"])
        y_values = (project_y - float(stable["cy"])) * z_values / float(stable["fy"])
        positions = np.stack((x_values, y_values, z_values), axis=1)
        positions[~depth_valid, :] = 0.0
        deproject_ms = (time.perf_counter() - deproject_started) * 1000.0

        output = []  # type: List[Dict[str, Any]]
        for index in range(point_array.shape[0]):
            valid = bool(depth_valid[index])
            output.append({
                "depth_valid": valid,
                "depth_mm": int(z_values[index]) if valid else 0,
                "sample_px": [int(sample_x[index]), int(sample_y[index])],
                "valid_pixels": int(valid_counts[index]),
                "position_camera": positions[index].tolist() if valid else [0.0, 0.0, 0.0],
                "project_valid": valid,
            })

        elapsed = (time.perf_counter() - started) * 1000.0
        self.last_error = ""
        self.last_snapshot_roi = [int(roi_x0), int(roi_y0), int(roi_x1), int(roi_y1)]
        self.last_snapshot_copy_ms = snapshot_copy_ms
        self.last_vectorized_sample_ms = vectorized_sample_ms
        return output, {
            "ok": True,
            "mode": "shared_depth_roi_snapshot",
            "depth_age_ms": int(stable["age_ms"]),
            "depth_sequence": int(stable["sequence"]),
            "publish_count": int(stable["publish_count"]),
            "sample_ms": elapsed,
            "snapshot_copy_ms": snapshot_copy_ms,
            "vectorized_sample_ms": vectorized_sample_ms,
            "vectorized_deproject_ms": deproject_ms,
            "snapshot_attempts": int(snapshot_attempts),
            "snapshot_roi_px": list(self.last_snapshot_roi),
            "snapshot_roi_bytes": int(snapshot.nbytes),
            "shared_memory_name": self.name,
            "retry_count": self.retry_count,
            "intrinsics": {
                "fx": stable["fx"], "fy": stable["fy"],
                "cx": stable["cx"], "cy": stable["cy"],
            },
            "depth_width": int(stable["width"]),
            "depth_height": int(stable["height"]),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "flip_horizontal": bool(stable["flip_horizontal"]),
            "flip_vertical": bool(stable["flip_vertical"]),
        }
