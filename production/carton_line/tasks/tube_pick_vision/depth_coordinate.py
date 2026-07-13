#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera Bridge depth and Orbbec SDK coordinate-conversion client."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from production.carton_line.gateway.runtime_client import HttpClient, UpstreamError
from production.carton_line.tasks.tube_pick_vision.algorithm import decode_depth_png


class BridgeCoordinateClient:
    def __init__(
        self,
        http: HttpClient,
        depth_url: str,
        health_url: str,
        deproject_url: str,
        max_depth_age_ms: int,
    ) -> None:
        self.http = http
        self.depth_url = depth_url
        self.health_url = health_url
        self.deproject_url = deproject_url
        self.max_depth_age_ms = max(0, int(max_depth_age_ms))

    def get_depth(self):
        health: dict[str, Any] = {}
        try:
            health = self.http.request("GET", self.health_url).json()
        except Exception as error:
            raise UpstreamError(f"读取 Orbbec Bridge 健康状态失败: {error}") from error
        age = health.get("last_depth_age_ms")
        try:
            age_ms = int(age)
        except (TypeError, ValueError, OverflowError):
            age_ms = -1
        if self.max_depth_age_ms > 0 and (age_ms < 0 or age_ms > self.max_depth_age_ms):
            raise ValueError(f"深度帧过旧: age={age_ms}ms, max={self.max_depth_age_ms}ms")
        depth_bytes = self.http.get_bytes(self.depth_url)
        return decode_depth_png(depth_bytes), health, depth_bytes

    def deproject(self, points: Sequence[Sequence[float]]) -> tuple[list[list[float]], dict[str, Any]]:
        """Call the Bridge batch endpoint backed by Orbbec SDK conversion.

        Input points are ``[u, v, depth_mm]`` in the fixed 640x480 color image.
        Invalid depth values are sent as zero and must return ``[0,0,0]``.
        """
        body = json.dumps({"points": [list(point[:3]) for point in points]}, separators=(",", ":")).encode("utf-8")
        response = self.http.request("POST", self.deproject_url, body).json()
        if response.get("ok") is not True:
            raise UpstreamError(f"Orbbec SDK 三维反投影失败: {response.get('error') or 'unknown'}")
        raw_points = response.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != len(points):
            raise UpstreamError("Orbbec SDK 三维反投影返回数量不一致")
        output: list[list[float]] = []
        for item in raw_points:
            if not isinstance(item, Mapping):
                output.append([0.0, 0.0, 0.0])
                continue
            position = item.get("position_camera")
            if not isinstance(position, list) or len(position) < 3 or item.get("valid") is not True:
                output.append([0.0, 0.0, 0.0])
                continue
            try:
                output.append([float(position[0]), float(position[1]), float(position[2])])
            except (TypeError, ValueError, OverflowError):
                output.append([0.0, 0.0, 0.0])
        return output, response
