#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-overhead D2C depth sampling for plastic_bag_grasp.

Normal production reads the Orbbec shared-depth ring directly and samples only a
small ROI around the bbox centre.  If shared depth is unavailable, the code can
fall back to the Bridge sample+deproject endpoint.  A depth failure is not an RGB
detection failure: the caller keeps center_px and emits zero camera XYZ.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from production.carton_line.gateway.runtime_client import HttpClient

SHARED_DEPTH_MAGIC = 0x564F505344455031
SHARED_DEPTH_VERSION = 1
SHARED_DEPTH_HEADER_SIZE = 256
SHARED_DEPTH_STATE_RUNNING = 1
SHARED_DEPTH_PIXEL_UINT16_MM = 1
SHARED_DEPTH_HEADER = struct.Struct("<QIIQQQ" + "I" * 12 + "Q" * 5 + "d" * 4 + "Q" * 12)


class SharedDepthReader:
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
        size = os.stat(self.path).st_size
        if self._mapping is not None and size == self._mapping_size:
            return self._mapping
        self.close()
        self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        self._mapping = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
        self._mapping_size = size
        return self._mapping

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
        for _attempt in range(4):
            h = self._header(mapping)
            (
                magic, version, header_size, total_size, frame_capacity, _frame_bytes,
                width, height, stride_bytes, pixel_format, buffer_count, state,
                active_buffer, calibration_ready, aligned_to_color, _flip_horizontal,
                _flip_vertical, _reserved0, sequence, timestamp_ms, _writer_pid,
                publish_count, _dropped_count, fx, fy, cx, cy, *_reserved
            ) = h
            if magic != SHARED_DEPTH_MAGIC or version != SHARED_DEPTH_VERSION or header_size != SHARED_DEPTH_HEADER_SIZE:
                raise ValueError("shared depth header incompatible")
            if total_size > self._mapping_size or buffer_count != 2 or pixel_format != SHARED_DEPTH_PIXEL_UINT16_MM:
                raise ValueError("shared depth mapping invalid")
            if state != SHARED_DEPTH_STATE_RUNNING or not calibration_ready or not aligned_to_color:
                raise ValueError("shared depth not ready/aligned")
            age_ms = int(time.time() * 1000) - int(timestamp_ms)
            if age_ms < 0 or age_ms > self.max_age_ms:
                raise ValueError("shared depth stale: {}ms".format(age_ms))
            if width <= 0 or height <= 0 or stride_bytes < width * 2 or fx <= 0 or fy <= 0:
                raise ValueError("shared depth dimensions/intrinsics invalid")

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
            radius_x = max(0, int(round(radius_px * sx)))
            radius_y = max(0, int(round(radius_px * sy)))
            output: List[Dict[str, Any]] = []
            for point in points:
                u = float(point[0])
                v = float(point[1])
                px = max(0, min(int(width) - 1, int(round(u * sx))))
                py = max(0, min(int(height) - 1, int(round(v * sy))))
                x0, x1 = max(0, px - radius_x), min(int(width), px + radius_x + 1)
                y0, y1 = max(0, py - radius_y), min(int(height), py + radius_y + 1)
                values = depth[y0:y1, x0:x1].reshape(-1)
                valid = values[(values >= int(min_depth_mm)) & (values <= int(max_depth_mm))]
                depth_valid = int(valid.size) >= int(min_valid_pixels)
                z = float(np.percentile(valid, percentile)) if depth_valid else 0.0
                project_x = u * sx
                project_y = v * sy
                position = [
                    (project_x - float(cx)) * z / float(fx),
                    (project_y - float(cy)) * z / float(fy),
                    z,
                ] if depth_valid else [0.0, 0.0, 0.0]
                output.append(
                    {
                        "depth_valid": depth_valid,
                        "depth_mm": int(round(z)) if depth_valid else 0,
                        "sample_px": [px, py],
                        "valid_pixels": int(valid.size),
                        "position_camera": position,
                    }
                )
            if self._header(mapping)[18] == sequence:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.last_error = ""
                return output, {
                    "ok": True,
                    "mode": "shared_depth",
                    "depth_age_ms": age_ms,
                    "depth_sequence": int(sequence),
                    "publish_count": int(publish_count),
                    "sample_ms": elapsed_ms,
                }
            self.retry_count += 1
        raise RuntimeError("shared depth changed repeatedly while sampling")


class DepthCoordinateClient:
    def __init__(self, bridge: Mapping[str, Any], depth: Mapping[str, Any], timeout_s: float) -> None:
        self.base_url = str(bridge.get("base_url") or "http://127.0.0.1:18182").rstrip("/")
        self.sample_url = self.base_url + str(bridge.get("sample_deproject_path") or "/api/coordinate/sample_deproject")
        self.http = HttpClient(timeout_s=timeout_s)
        self.enabled = bool(depth.get("enabled", True))
        self.radius_px = max(0, int(depth.get("roi_radius_px", 6)))
        self.percentile = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
        self.min_valid_pixels = max(1, int(depth.get("min_valid_pixels", 5)))
        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))
        self.max_age_ms = max(1, int(depth.get("max_age_ms", 1500)))
        camera_model = str(bridge.get("camera_model") or "").strip().lower()
        self.shared_fallback_http = bool(bridge.get("shared_depth_fallback_http", True))
        self.shared = None  # type: Optional[SharedDepthReader]
        if bool(bridge.get("shared_depth_enabled", True)) and camera_model == "orbbec336l":
            self.shared = SharedDepthReader(str(bridge.get("shared_depth_name") or "/visionops_orbbec336l_depth"), self.max_age_ms)
        self.last_mode = "disabled" if not self.enabled else "none"
        self.last_error = ""

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_mode": self.last_mode,
            "last_error": self.last_error,
            "shared": self.shared.status() if self.shared is not None else None,
            "sample_url": self.sample_url,
        }

    def _http_sample(self, points: Sequence[Sequence[float]], image_width: int, image_height: int):
        document = {
            "points": [[float(point[0]), float(point[1]), float(point[0]), float(point[1])] for point in points],
            "image_width": int(image_width),
            "image_height": int(image_height),
            "radius_px": self.radius_px,
            "percentile": self.percentile,
            "min_valid_pixels": self.min_valid_pixels,
            "min_depth_mm": self.min_depth_mm,
            "max_depth_mm": self.max_depth_mm,
            "max_depth_age_ms": self.max_age_ms,
        }
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self.http.request("POST", self.sample_url, body).json()
        if response.get("ok") is not True:
            raise RuntimeError("Bridge sample_deproject failed: {}".format(response.get("error") or "unknown"))
        raw_points = response.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != len(points):
            raise RuntimeError("Bridge sample_deproject result count mismatch")
        output: List[Dict[str, Any]] = []
        for raw in raw_points:
            item = raw if isinstance(raw, Mapping) else {}
            position = item.get("position_camera") if isinstance(item.get("position_camera"), list) else [0, 0, 0]
            if len(position) < 3 or not bool(item.get("depth_valid")) or not bool(item.get("valid")):
                position = [0.0, 0.0, 0.0]
            output.append(
                {
                    "depth_valid": bool(item.get("depth_valid")) and position != [0.0, 0.0, 0.0],
                    "depth_mm": int(item.get("depth_mm") or 0),
                    "sample_px": list(item.get("sample_px") or [0, 0]),
                    "valid_pixels": int(item.get("valid_pixels") or 0),
                    "position_camera": [float(position[0]), float(position[1]), float(position[2])],
                }
            )
        return output, response

    def sample(self, points: Sequence[Sequence[float]], image_width: int, image_height: int):
        if not self.enabled or not points:
            return [], {"ok": True, "mode": "disabled", "sample_ms": 0.0}
        if self.shared is not None:
            try:
                output, debug = self.shared.sample_deproject(
                    points,
                    image_width,
                    image_height,
                    self.radius_px,
                    self.percentile,
                    self.min_valid_pixels,
                    self.min_depth_mm,
                    self.max_depth_mm,
                )
                self.last_mode = "shared_depth"
                self.last_error = ""
                return output, debug
            except (OSError, ValueError, RuntimeError) as error:
                self.last_error = str(error)
                if not self.shared_fallback_http:
                    raise
        started = time.perf_counter()
        output, debug = self._http_sample(points, image_width, image_height)
        debug = dict(debug)
        debug["client_roundtrip_ms"] = (time.perf_counter() - started) * 1000.0
        debug["mode"] = str(debug.get("mode") or "http_sample_deproject")
        self.last_mode = "http_sample_deproject"
        self.last_error = ""
        return output, debug
