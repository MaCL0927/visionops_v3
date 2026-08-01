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
