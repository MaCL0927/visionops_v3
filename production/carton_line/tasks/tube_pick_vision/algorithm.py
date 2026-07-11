#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task algorithm for tube-product centers and large-separator presence.

The model is expected to emit two detection classes:

* class 0: tube product. Return image-space center ``x/y`` and aligned depth ``z``.
* class 1: large separator between product layers. Return class information only.

No robot/base-link coordinate conversion is performed in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore


class DetectionFormatError(ValueError):
    """Raised when the runtime result is missing required image metadata."""


@dataclass(frozen=True)
class ClassifiedDetections:
    image_width: int
    image_height: int
    products: list[dict[str, Any]]
    separators: list[dict[str, Any]]
    ignored: list[dict[str, Any]]


def _float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else default
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
    """Decode the bridge 16UC1 PNG whose values are millimetres."""
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
    """Filter model detections and sample D2C-aligned depth around each center."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        depth = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}

        self.product_ids = _int_set(classes.get("product_ids"), [0])
        self.separator_ids = _int_set(classes.get("separator_ids"), [1])
        self.product_names = _name_set(classes.get("product_names"))
        self.separator_names = _name_set(classes.get("separator_names"))
        self.product_min_confidence = float(classes.get("product_min_confidence", 0.50))
        self.separator_min_confidence = float(classes.get("separator_min_confidence", 0.50))
        self.output_order = str(classes.get("output_order", "row_major")).strip().lower()
        if self.output_order not in {"row_major", "column_major", "confidence"}:
            raise ValueError("pick.algorithm.classes.output_order 必须是 row_major/column_major/confidence")

        self.roi_radius_px = max(0, int(depth.get("roi_radius_px", 4)))
        self.percentile = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
        self.min_valid_pixels = max(1, int(depth.get("min_valid_pixels", 3)))
        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))
        self.max_age_ms = max(0, int(depth.get("max_age_ms", 1500)))
        self.fail_on_invalid_depth = bool(depth.get("fail_on_invalid_depth", True))

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

    def _is_product(self, class_id: int | None, class_name: str) -> bool:
        return (class_id in self.product_ids) or (
            bool(self.product_names) and class_name.lower() in self.product_names
        )

    def _is_separator(self, class_id: int | None, class_name: str) -> bool:
        return (class_id in self.separator_ids) or (
            bool(self.separator_names) and class_name.lower() in self.separator_names
        )

    def classify(self, runtime_result: Mapping[str, Any]) -> ClassifiedDetections:
        width, height = self._image_size(runtime_result)
        detections = runtime_result.get("detections")
        detections = detections if isinstance(detections, list) else []
        products: list[dict[str, Any]] = []
        separators: list[dict[str, Any]] = []
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
            detection_id = str(raw.get("id") or f"det-{index}")

            if self._is_product(class_id, class_name):
                center = self._center(raw)
                if score < self.product_min_confidence or center is None:
                    ignored.append({"id": detection_id, "reason": "low_confidence_or_missing_center"})
                    continue
                products.append(
                    {
                        "id": detection_id,
                        "class_id": class_id if class_id is not None else 0,
                        "class_name": class_name or "tube_product",
                        "score": score,
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        # Kept for local debug only; not exposed in the external response.
                        "bbox_xyxy": list(raw.get("bbox_xyxy") or []),
                    }
                )
            elif self._is_separator(class_id, class_name):
                if score < self.separator_min_confidence:
                    ignored.append({"id": detection_id, "reason": "low_confidence"})
                    continue
                separators.append(
                    {
                        "id": detection_id,
                        "class_id": class_id if class_id is not None else 1,
                        "class_name": class_name or "large_separator",
                        "score": score,
                    }
                )
            else:
                ignored.append({"id": detection_id, "reason": "class_not_used", "class_id": class_id})

        if self.output_order == "row_major":
            products.sort(key=lambda item: (float(item["center_y"]), float(item["center_x"])))
        elif self.output_order == "column_major":
            products.sort(key=lambda item: (float(item["center_x"]), float(item["center_y"])))
        else:
            products.sort(key=lambda item: -float(item["score"]))
        separators.sort(key=lambda item: -float(item["score"]))
        return ClassifiedDetections(width, height, products, separators, ignored)

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
                "z_mm": None,
                "valid": False,
                "depth_x": depth_x,
                "depth_y": depth_y,
                "valid_pixels": valid_count,
            }
        z_mm = int(round(float(np.percentile(valid.astype(np.float32), self.percentile))))
        return {
            "z_mm": z_mm,
            "valid": True,
            "depth_x": depth_x,
            "depth_y": depth_y,
            "valid_pixels": valid_count,
        }

    def build_detection_payload(
        self,
        classified: ClassifiedDetections,
        depth: "np.ndarray | None",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return external payload and richer local debug details."""
        external_products: list[dict[str, Any]] = []
        debug_products: list[dict[str, Any]] = []
        invalid_depth_count = 0

        for product in classified.products:
            depth_info: dict[str, Any]
            if depth is None:
                depth_info = {
                    "z_mm": None,
                    "valid": False,
                    "depth_x": None,
                    "depth_y": None,
                    "valid_pixels": 0,
                }
            else:
                depth_info = self.sample_depth(
                    depth,
                    classified.image_width,
                    classified.image_height,
                    float(product["center_x"]),
                    float(product["center_y"]),
                )
            if not depth_info["valid"]:
                invalid_depth_count += 1
            external_products.append(
                {
                    "class_id": int(product["class_id"]),
                    "class_name": str(product["class_name"]),
                    "score": round(float(product["score"]), 6),
                    "center": {
                        "x": round(float(product["center_x"]), 3),
                        "y": round(float(product["center_y"]), 3),
                        "z": depth_info["z_mm"],
                    },
                    "depth_valid": bool(depth_info["valid"]),
                }
            )
            debug_products.append({**product, **depth_info})

        # Separator items deliberately contain no bbox, center or depth fields.
        external_separators = [
            {
                "class_id": int(item["class_id"]),
                "class_name": str(item["class_name"]),
                "score": round(float(item["score"]), 6),
            }
            for item in classified.separators
        ]

        depth_height = int(depth.shape[0]) if depth is not None else 0
        depth_width = int(depth.shape[1]) if depth is not None else 0
        payload = {
            "coordinate_frame": "image_depth_aligned",
            "coordinate_units": {"x": "pixel", "y": "pixel", "z": "mm"},
            "image": {"width": classified.image_width, "height": classified.image_height},
            "depth": {
                "width": depth_width,
                "height": depth_height,
                "encoding": "16UC1",
                "unit": "mm",
                "aligned_to": "color",
                "sampling": "roi_percentile",
                "roi_radius_px": self.roi_radius_px,
                "percentile": self.percentile,
                "required": bool(classified.products),
            },
            "product_detected": bool(external_products),
            "separator_detected": bool(external_separators),
            "product_count": len(external_products),
            "separator_count": len(external_separators),
            "invalid_depth_count": invalid_depth_count,
            "products": external_products,
            "separators": external_separators,
        }
        debug = {
            **payload,
            "products": debug_products,
            "ignored_detections": classified.ignored,
        }
        return payload, debug
