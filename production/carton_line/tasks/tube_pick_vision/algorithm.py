#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection/depth preparation for the external bin-picking contract.

The Runtime still receives the complete RGB image. ROI filtering is performed by
Runtime before this module receives ``inference_result``. This module classifies
product/separator/lying detections, samples the D2C-aligned depth around each
bbox centre, and prepares the points that are deprojected by the Orbbec SDK
bridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore


class DetectionFormatError(ValueError):
    """Raised when Runtime output is missing required fields."""


@dataclass(frozen=True)
class ClassifiedDetections:
    image_width: int
    image_height: int
    items: list[dict[str, Any]]
    ignored: list[dict[str, Any]]


def _float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else default
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _int_set(values: object, default: Sequence[int]) -> set[int]:
    if not isinstance(values, (list, tuple, set)):
        values = default
    output: set[int] = set()
    for item in values:
        if isinstance(item, bool):
            continue
        try:
            output.add(int(item))
        except (TypeError, ValueError):
            continue
    return output or set(default)


def _name_set(values: object) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(item).strip().lower() for item in values if str(item).strip()}


def decode_depth_png(depth_bytes: bytes) -> "np.ndarray":
    """Decode the Bridge 16UC1 PNG whose pixel values are millimetres."""
    if not depth_bytes:
        raise ValueError("深度图为空")
    encoded = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.size == 0:
        raise ValueError("深度 PNG 解码失败")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"深度图维度非法: {depth.shape}")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    return depth


class TubePickAlgorithm:
    """Classify detections and sample D2C-aligned depth around bbox centres."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        depth = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}
        image = settings.get("image") if isinstance(settings.get("image"), Mapping) else {}

        self.product_ids = _int_set(classes.get("product_ids"), [0])
        self.separator_ids = _int_set(classes.get("separator_ids"), [1])
        self.lying_ids = _int_set(classes.get("lying_ids"), [2])
        self.product_names = _name_set(classes.get("product_names"))
        self.separator_names = _name_set(classes.get("separator_names"))
        self.lying_names = _name_set(classes.get("lying_names"))
        self.product_min_confidence = float(classes.get("product_min_confidence", 0.50))
        self.separator_min_confidence = float(classes.get("separator_min_confidence", 0.50))
        self.lying_min_confidence = float(classes.get("lying_min_confidence", 0.50))
        self.output_order = str(classes.get("output_order", "row_major")).strip().lower()
        if self.output_order not in {"row_major", "column_major", "confidence"}:
            raise ValueError("pick.algorithm.classes.output_order 必须是 row_major/column_major/confidence")

        self.expected_width = max(1, int(image.get("width", 640)))
        self.expected_height = max(1, int(image.get("height", 480)))
        self.require_fixed_size = bool(image.get("require_fixed_size", True))

        self.roi_radius_px = max(0, int(depth.get("roi_radius_px", 4)))
        self.percentile = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
        self.min_valid_pixels = max(1, int(depth.get("min_valid_pixels", 3)))
        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))
        self.max_age_ms = max(0, int(depth.get("max_age_ms", 1500)))

    @staticmethod
    def _image_size(runtime_result: Mapping[str, Any]) -> tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        width = int(_float(image.get("width")))
        height = int(_float(image.get("height")))
        if width <= 0 or height <= 0:
            raise DetectionFormatError("Runtime inference_result 缺少有效 image.width/image.height")
        return width, height

    @staticmethod
    def _center(item: Mapping[str, Any]) -> tuple[float, float] | None:
        center = item.get("center_xy")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            return _float(center[0]), _float(center[1])
        bbox = item.get("bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = (_float(value) for value in bbox[:4])
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return None

    def _semantic(self, class_id: int | None, class_name: str) -> str | None:
        lower_name = class_name.lower()
        if class_id in self.product_ids or (self.product_names and lower_name in self.product_names):
            return "product"
        if class_id in self.separator_ids or (self.separator_names and lower_name in self.separator_names):
            return "separator"
        if class_id in self.lying_ids or (self.lying_names and lower_name in self.lying_names):
            return "lying"
        return None

    def classify(self, runtime_result: Mapping[str, Any]) -> ClassifiedDetections:
        width, height = self._image_size(runtime_result)
        if self.require_fixed_size and (width != self.expected_width or height != self.expected_height):
            raise DetectionFormatError(
                f"tube_pick_vision 固定图像尺寸为 {self.expected_width}x{self.expected_height}，"
                f"Runtime 当前为 {width}x{height}"
            )

        detections = runtime_result.get("detections")
        detections = detections if isinstance(detections, list) else []
        accepted: list[dict[str, Any]] = []
        ignored: list[dict[str, Any]] = []

        for index, raw in enumerate(detections):
            if not isinstance(raw, Mapping):
                continue
            raw_class_id = raw.get("class_id")
            try:
                class_id = int(raw_class_id) if raw_class_id is not None and not isinstance(raw_class_id, bool) else None
            except (TypeError, ValueError):
                class_id = None
            class_name = str(raw.get("class_name") or "")
            score = _float(raw.get("score"))
            center = self._center(raw)
            semantic = self._semantic(class_id, class_name)
            detection_id = str(raw.get("id") or f"det-{index}")

            if semantic is None:
                ignored.append({"id": detection_id, "reason": "class_not_used", "class_id": class_id})
                continue
            threshold_by_semantic = {
                "product": self.product_min_confidence,
                "separator": self.separator_min_confidence,
                "lying": self.lying_min_confidence,
            }
            threshold = threshold_by_semantic[semantic]
            if score < threshold or center is None:
                ignored.append({"id": detection_id, "reason": "low_confidence_or_missing_center"})
                continue

            default_by_semantic = {
                "product": (0, "tube_product"),
                "separator": (1, "large_separator"),
                "lying": (2, "lying"),
            }
            default_id, default_name = default_by_semantic[semantic]
            accepted.append(
                {
                    "source_id": detection_id,
                    "semantic": semantic,
                    "class_id": class_id if class_id is not None else default_id,
                    "class_name": class_name or default_name,
                    "confidence": score,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "bbox_xyxy": list(raw.get("bbox_xyxy") or []),
                }
            )

        if self.output_order == "row_major":
            accepted.sort(key=lambda item: (float(item["center_y"]), float(item["center_x"])))
        elif self.output_order == "column_major":
            accepted.sort(key=lambda item: (float(item["center_x"]), float(item["center_y"])))
        else:
            accepted.sort(key=lambda item: -float(item["confidence"]))
        return ClassifiedDetections(width, height, accepted, ignored)

    @staticmethod
    def _map_pixel(value: float, source_size: int, target_size: int) -> int:
        if target_size <= 1:
            return 0
        if source_size <= 1:
            return int(round(value))
        mapped = value * float(target_size - 1) / float(source_size - 1)
        return max(0, min(target_size - 1, int(round(mapped))))

    def sample_depth(
        self,
        depth: "np.ndarray",
        image_width: int,
        image_height: int,
        center_x: float,
        center_y: float,
    ) -> dict[str, Any]:
        depth_height, depth_width = int(depth.shape[0]), int(depth.shape[1])
        if self.require_fixed_size and (depth_width != self.expected_width or depth_height != self.expected_height):
            raise ValueError(
                f"tube_pick_vision 固定深度尺寸为 {self.expected_width}x{self.expected_height}，"
                f"Bridge 当前为 {depth_width}x{depth_height}"
            )
        depth_x = self._map_pixel(center_x, image_width, depth_width)
        depth_y = self._map_pixel(center_y, image_height, depth_height)
        radius = self.roi_radius_px
        x1, x2 = max(0, depth_x - radius), min(depth_width, depth_x + radius + 1)
        y1, y2 = max(0, depth_y - radius), min(depth_height, depth_y + radius + 1)
        roi = depth[y1:y2, x1:x2]
        valid = roi[(roi >= self.min_depth_mm) & (roi <= self.max_depth_mm)]
        valid_count = int(valid.size)
        if valid_count < self.min_valid_pixels:
            return {
                "z_mm": 0,
                "depth_valid": False,
                "depth_x": depth_x,
                "depth_y": depth_y,
                "valid_pixels": valid_count,
            }
        z_mm = int(round(float(np.percentile(valid.astype(np.float32), self.percentile))))
        return {
            "z_mm": z_mm,
            "depth_valid": True,
            "depth_x": depth_x,
            "depth_y": depth_y,
            "valid_pixels": valid_count,
        }

    def sample_items(
        self,
        classified: ClassifiedDetections,
        depth: "np.ndarray",
    ) -> list[dict[str, Any]]:
        sampled: list[dict[str, Any]] = []
        for item in classified.items:
            depth_info = self.sample_depth(
                depth,
                classified.image_width,
                classified.image_height,
                float(item["center_x"]),
                float(item["center_y"]),
            )
            sampled.append({**item, **depth_info})
        return sampled

    @staticmethod
    def build_external_items(
        sampled: Sequence[Mapping[str, Any]],
        positions_camera: Sequence[Sequence[float]],
    ) -> list[dict[str, Any]]:
        if len(sampled) != len(positions_camera):
            raise ValueError("SDK 三维反投影结果数量与检测目标数量不一致")
        output: list[dict[str, Any]] = []
        for index, (item, position) in enumerate(zip(sampled, positions_camera)):
            if len(position) < 3:
                position = [0.0, 0.0, 0.0]
            if not bool(item.get("depth_valid")):
                position = [0.0, 0.0, 0.0]
            output.append(
                {
                    "id": index,
                    "class_id": int(item.get("class_id", 0)),
                    "confidence": round(float(item.get("confidence", 0.0)), 6),
                    "position_camera": [round(float(position[0]), 3), round(float(position[1]), 3), round(float(position[2]), 3)],
                    "center_px": [round(float(item.get("center_x", 0.0)), 3), round(float(item.get("center_y", 0.0)), 3)],
                }
            )
        return output
