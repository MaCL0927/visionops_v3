#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BBox-centre post-processing for the plastic-bag grasp task.

The detector sees the whole wrapped package.  The robot grasp XY is intentionally
not tied to the transparent knot/head geometry: it is always the centre of the
selected package detection bbox.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class DetectionFormatError(ValueError):
    """Raised when Runtime output cannot satisfy the detection contract."""


@dataclass(frozen=True)
class PlasticBagGraspResult:
    image_width: int
    image_height: int
    items: List[Dict[str, Any]]
    selected: List[Dict[str, Any]]
    ignored: List[Dict[str, Any]]


def _number(value: object, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return output if math.isfinite(output) else default


def _optional_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _int_set(values: object) -> set[int]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    output: set[int] = set()
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            output.add(parsed)
    return output


def _name_set(values: object) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


class PlasticBagGraspAlgorithm:
    def __init__(self, settings: Mapping[str, Any]) -> None:
        image = settings.get("image") if isinstance(settings.get("image"), Mapping) else {}
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        selection = settings.get("selection") if isinstance(settings.get("selection"), Mapping) else {}

        self.expected_width = max(1, int(image.get("width", 640)))
        self.expected_height = max(1, int(image.get("height", 480)))
        self.require_fixed_size = bool(image.get("require_fixed_size", True))
        self.target_ids = _int_set(classes.get("target_ids"))
        self.target_names = _name_set(classes.get("target_names"))
        self.min_confidence = float(classes.get("min_confidence", 0.5))
        self.max_targets = max(1, int(selection.get("max_targets", 1)))
        self.selection_mode = str(selection.get("mode", "confidence")).strip().lower()

    @staticmethod
    def _image_size(runtime_result: Mapping[str, Any]) -> Tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        width = int(_number(image.get("width")))
        height = int(_number(image.get("height")))
        if width <= 0 or height <= 0:
            raise DetectionFormatError("Runtime inference_result 缺少有效 image.width/image.height")
        return width, height

    def _is_target(self, class_id: Optional[int], class_name: str) -> bool:
        lower = class_name.strip().lower()
        if lower and lower in self.target_names:
            return True
        return class_id is not None and class_id in self.target_ids

    @staticmethod
    def _bbox(raw: Mapping[str, Any]) -> Optional[List[float]]:
        bbox = raw.get("bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = [_number(value, float("nan")) for value in bbox[:4]]
            if all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                return [float(x1), float(y1), float(x2), float(y2)]
        return None

    @staticmethod
    def _center(raw: Mapping[str, Any], bbox: Sequence[float]) -> List[float]:
        center = raw.get("center_xy")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            x, y = _number(center[0], float("nan")), _number(center[1], float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                return [float(x), float(y)]
        return [(float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5]

    def evaluate(self, runtime_result: Mapping[str, Any]) -> PlasticBagGraspResult:
        width, height = self._image_size(runtime_result)
        if self.require_fixed_size and (width != self.expected_width or height != self.expected_height):
            raise DetectionFormatError(
                f"plastic_bag_grasp 固定图像尺寸为 {self.expected_width}x{self.expected_height}，"
                f"Runtime 当前为 {width}x{height}"
            )

        detections = runtime_result.get("detections")
        detections = detections if isinstance(detections, list) else []
        candidates: List[Dict[str, Any]] = []
        ignored: List[Dict[str, Any]] = []
        for index, raw in enumerate(detections):
            if not isinstance(raw, Mapping):
                continue
            source_id = str(raw.get("id") or f"det-{index}")
            class_id = _optional_int(raw.get("class_id"))
            class_name = str(raw.get("class_name") or "")
            confidence = _number(raw.get("score"), _number(raw.get("confidence"), 0.0))
            if not self._is_target(class_id, class_name):
                ignored.append({"source_id": source_id, "reason": "class_not_used"})
                continue
            if confidence < self.min_confidence:
                ignored.append({"source_id": source_id, "reason": "low_confidence", "confidence": confidence})
                continue
            bbox = self._bbox(raw)
            if bbox is None:
                ignored.append({"source_id": source_id, "reason": "missing_or_invalid_bbox"})
                continue
            center = self._center(raw, bbox)
            area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            candidates.append(
                {
                    "source_id": source_id,
                    "class_id": class_id if class_id is not None else 0,
                    "class_name": class_name or "plastic_bag",
                    "confidence": max(0.0, min(1.0, confidence)),
                    "bbox_xyxy": bbox,
                    "center_px": center,
                    "bbox_area_px2": area,
                }
            )

        if self.selection_mode == "largest_area":
            candidates.sort(key=lambda item: (-float(item["bbox_area_px2"]), -float(item["confidence"])))
        else:
            candidates.sort(key=lambda item: (-float(item["confidence"]), -float(item["bbox_area_px2"])))
        selected = candidates[: self.max_targets]

        items: List[Dict[str, Any]] = []
        for index, target in enumerate(selected):
            items.append(
                {
                    "id": index,
                    "class_id": int(target["class_id"]),
                    "confidence": round(float(target["confidence"]), 6),
                    "position_camera": [0.0, 0.0, 0.0],
                    "center_px": [round(float(target["center_px"][0]), 3), round(float(target["center_px"][1]), 3)],
                }
            )

        return PlasticBagGraspResult(width, height, items, selected, ignored)
