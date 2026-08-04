"""Segmentation input adapters for PT inference and YOLO polygon labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass
class SegmentationInstance:
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def centroid_uv(self) -> Tuple[float, float]:
        moments = cv2.moments(self.mask.astype(np.uint8), binaryImage=True)
        if abs(moments["m00"]) < 1e-9:
            x1, y1, x2, y2 = self.bbox_xyxy
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def load_yolo_segmentation_labels(
    label_path: Path,
    image_shape: Tuple[int, int],
    class_names: Sequence[str],
) -> List[SegmentationInstance]:
    height, width = image_shape
    instances: List[SegmentationInstance] = []
    if not label_path.exists():
        raise FileNotFoundError("YOLO标签不存在: %s" % label_path)
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
            raise ValueError("YOLO segmentation标签格式错误: %s:%d" % (label_path, line_number))
        class_id = int(float(fields[0]))
        coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0] * width, 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1] * height, 0, height - 1)
        polygon = np.rint(coords).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        instances.append(
            SegmentationInstance(
                instance_id=len(instances),
                class_id=class_id,
                class_name=name,
                confidence=1.0,
                mask=mask.astype(bool),
                bbox_xyxy=_bbox_from_mask(mask),
            )
        )
    return instances


def infer_ultralytics_segmentation(
    image_bgr: np.ndarray,
    model_path: Path,
    confidence: float,
    iou: float,
    image_size: int,
    max_detections: int,
    retina_masks: bool,
    device: str,
) -> List[SegmentationInstance]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("当前Python环境未安装ultralytics，无法加载PT模型") from error

    model = YOLO(str(model_path))
    kwargs: Dict[str, Any] = {
        "source": image_bgr,
        "conf": float(confidence),
        "iou": float(iou),
        "imgsz": int(image_size),
        "max_det": int(max_detections),
        "retina_masks": bool(retina_masks),
        "verbose": False,
    }
    if device and device != "auto":
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        return []
    result = results[0]
    if result.boxes is None or result.masks is None:
        return []
    boxes = result.boxes
    masks_tensor = result.masks.data
    masks = masks_tensor.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
    confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    names_raw = result.names or model.names or {}

    def class_name(class_id: int) -> str:
        if isinstance(names_raw, dict):
            return str(names_raw.get(class_id, class_id))
        if isinstance(names_raw, (list, tuple)) and 0 <= class_id < len(names_raw):
            return str(names_raw[class_id])
        return str(class_id)

    height, width = image_bgr.shape[:2]
    instances: List[SegmentationInstance] = []
    for index in range(min(len(masks), len(classes), len(confidences), len(xyxy))):
        raw = masks[index]
        if raw.shape != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = raw > 0.5
        x1, y1, x2, y2 = xyxy[index].tolist()
        class_id = int(classes[index])
        instances.append(
            SegmentationInstance(
                instance_id=index,
                class_id=class_id,
                class_name=class_name(class_id),
                confidence=float(confidences[index]),
                mask=mask,
                bbox_xyxy=(
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    min(width, int(round(x2))),
                    min(height, int(round(y2))),
                ),
            )
        )
    return instances


@dataclass(frozen=True)
class RuntimeSegmentationAdaptation:
    """Validated Runtime polygon-to-mask conversion result.

    Only real proto-derived masks are accepted by default.  Detection-box
    fallback polygons may be useful for a Web preview, but must not silently
    enter RGB-D geometry.
    """

    instances: List[SegmentationInstance]
    accepted_count: int
    rejected_count: int
    rejected: List[Dict[str, Any]]
    polygon_point_count: int


def _runtime_polygon_contours(value: Any) -> List[np.ndarray]:
    """Normalize Runtime polygon JSON to OpenCV int32 contours."""

    if not isinstance(value, (list, tuple)) or not value:
        return []

    # Runtime currently emits ``[[[x, y], ...], ...]``.  Accept a single
    # unwrapped ``[[x, y], ...]`` as well so older packages remain readable.
    first = value[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) >= 2
        and all(isinstance(item, (int, float)) for item in first[:2])
    ):
        candidates = [value]
    else:
        candidates = list(value)

    contours: List[np.ndarray] = []
    for candidate in candidates:
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 3:
            continue
        points: List[Tuple[float, float]] = []
        for point in candidate:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            points.append((x, y))
        if len(points) >= 3:
            contours.append(np.asarray(points, dtype=np.float32))
    return contours


def runtime_result_to_segmentation_instances(
    result: Mapping[str, Any],
    image_shape: Tuple[int, int],
    *,
    require_proto_mask: bool = True,
    reject_bbox_fallback: bool = True,
    minimum_mask_area_px: int = 1,
    geometry_roi_xyxy: Tuple[int, int, int, int] | None = None,
) -> RuntimeSegmentationAdaptation:
    """Convert one C++ Runtime segmentation result to geometry instances.

    The conversion is deliberately strict because the returned masks are used
    to select aligned depth pixels.  Coordinates are clipped to the exact RGB-D
    frame, and an empty or malformed polygon is rejected instead of being
    replaced by a detection box.
    """

    if str(result.get("message_type") or "") != "inference_result":
        raise ValueError("Runtime结果不是 inference_result")
    if str(result.get("status") or "") != "ok":
        raise ValueError(f"Runtime推理状态异常: {result.get('status')!r}")
    if str(result.get("task_type") or "") != "segmentation":
        raise ValueError(f"Runtime任务不是segmentation: {result.get('task_type')!r}")

    full_height, full_width = int(image_shape[0]), int(image_shape[1])
    if full_height <= 0 or full_width <= 0:
        raise ValueError(f"目标图像尺寸无效: {full_width}x{full_height}")
    image = result.get("image") if isinstance(result.get("image"), Mapping) else {}
    runtime_width = int(image.get("width") or 0)
    runtime_height = int(image.get("height") or 0)
    if runtime_width != full_width or runtime_height != full_height:
        raise ValueError(
            "Runtime结果与精确匹配RGB-D尺寸不一致: "
            f"runtime={runtime_width}x{runtime_height}, rgbd={full_width}x{full_height}"
        )

    if geometry_roi_xyxy is None:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, full_width, full_height
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in geometry_roi_xyxy]
        if not (
            0 <= roi_x1 < roi_x2 <= full_width
            and 0 <= roi_y1 < roi_y2 <= full_height
        ):
            raise ValueError(
                "几何ROI越界: "
                f"roi={(roi_x1, roi_y1, roi_x2, roi_y2)}, "
                f"image={full_width}x{full_height}"
            )
    width = roi_x2 - roi_x1
    height = roi_y2 - roi_y1

    detections = result.get("detections") or []
    if not isinstance(detections, list):
        raise ValueError("Runtime detections字段不是数组")

    instances: List[SegmentationInstance] = []
    rejected: List[Dict[str, Any]] = []
    polygon_point_count = 0
    minimum_area = max(1, int(minimum_mask_area_px))

    for detection_index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            rejected.append({"index": detection_index, "reason": "detection_not_object"})
            continue
        mask_document = detection.get("mask")
        if not isinstance(mask_document, Mapping):
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "mask_missing",
            })
            continue
        source = str(mask_document.get("source") or "")
        if reject_bbox_fallback and source == "bbox_fallback":
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "source": source,
                "reason": "bbox_fallback_rejected",
            })
            continue
        if require_proto_mask and source != "proto":
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "source": source,
                "reason": "non_proto_mask_rejected",
            })
            continue

        mask_size = mask_document.get("size") or []
        if (
            not isinstance(mask_size, (list, tuple))
            or len(mask_size) < 2
            or int(mask_size[0]) != full_height
            or int(mask_size[1]) != full_width
        ):
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "mask_size_mismatch",
                "mask_size": list(mask_size) if isinstance(mask_size, (list, tuple)) else mask_size,
            })
            continue

        contours = _runtime_polygon_contours(mask_document.get("polygon"))
        if not contours:
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "polygon_empty_or_malformed",
            })
            continue

        binary = np.zeros((height, width), dtype=np.uint8)
        clipped_contours: List[np.ndarray] = []
        for contour in contours:
            clipped = contour.copy()
            clipped[:, 0] = np.clip(
                clipped[:, 0] - float(roi_x1),
                0.0,
                float(width - 1),
            )
            clipped[:, 1] = np.clip(
                clipped[:, 1] - float(roi_y1),
                0.0,
                float(height - 1),
            )
            integer = np.rint(clipped).astype(np.int32).reshape(-1, 1, 2)
            if integer.shape[0] >= 3:
                clipped_contours.append(integer)
                polygon_point_count += int(integer.shape[0])
        if not clipped_contours:
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "polygon_empty_after_clipping",
            })
            continue
        cv2.fillPoly(binary, clipped_contours, 1)
        area = int(np.count_nonzero(binary))
        if area < minimum_area:
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "mask_area_too_small",
                "area_px": area,
            })
            continue

        try:
            class_id = int(detection.get("class_id"))
            confidence = float(detection.get("score"))
        except (TypeError, ValueError, OverflowError):
            rejected.append({
                "index": detection_index,
                "id": detection.get("id"),
                "reason": "class_or_score_invalid",
            })
            continue
        class_name = str(detection.get("class_name") or class_id)
        instances.append(
            SegmentationInstance(
                instance_id=len(instances),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                mask=binary.astype(bool),
                bbox_xyxy=_bbox_from_mask(binary),
            )
        )

    return RuntimeSegmentationAdaptation(
        instances=instances,
        accepted_count=len(instances),
        rejected_count=len(rejected),
        rejected=rejected,
        polygon_point_count=polygon_point_count,
    )
