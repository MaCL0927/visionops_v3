"""Optional SAM-assisted polygon generation for the built-in annotator.

The service is deliberately lazy: importing the VisionOps server does not import
PyTorch or Ultralytics, and the SAM model is loaded only after the first prompt.
No model file is bundled in the repository.
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


SAM_MODEL_ENV = "VISIONOPS_ANNOTATOR_SAM_MODEL"
SAM_DEVICE_ENV = "VISIONOPS_ANNOTATOR_SAM_DEVICE"
SAM_MAX_POINTS_ENV = "VISIONOPS_ANNOTATOR_SAM_MAX_POINTS"
SAM_SIMPLIFY_RATIO_ENV = "VISIONOPS_ANNOTATOR_SAM_SIMPLIFY_RATIO"
SAM_ALLOW_DOWNLOAD_ENV = "VISIONOPS_ANNOTATOR_SAM_ALLOW_DOWNLOAD"

DEFAULT_MODEL_CANDIDATES = (
    "models/pretrained/sam2.1_s.pt",
    "models/pretrained/mobile_sam.pt",
)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _polygon_area(points: Iterable[Iterable[float]]) -> float:
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _rdp_open(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points[:]
    first = points[0]
    last = points[-1]
    max_distance = -1.0
    max_index = 0
    for index in range(1, len(points) - 1):
        distance = _distance_to_segment(points[index], first, last)
        if distance > max_distance:
            max_distance = distance
            max_index = index
    if max_distance > epsilon:
        left = _rdp_open(points[: max_index + 1], epsilon)
        right = _rdp_open(points[max_index:], epsilon)
        return left[:-1] + right
    return [first, last]


def simplify_closed_polygon(points: Iterable[Iterable[float]], epsilon: float, max_points: int) -> list[list[float]]:
    """Simplify a closed contour without requiring OpenCV.

    SAM contours often contain hundreds of points. VisionOps stores YOLO polygon
    labels, so reducing redundant collinear points substantially improves browser
    editing and label file size while preserving the object boundary.
    """

    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return []

    # Rotate at a point far from the first point so the open RDP seam does not
    # systematically cut through the same part of the contour.
    anchor = pts[0]
    split_index = max(range(1, len(pts)), key=lambda idx: math.hypot(pts[idx][0] - anchor[0], pts[idx][1] - anchor[1]))
    rotated = pts[split_index:] + pts[: split_index + 1]
    simplified = _rdp_open(rotated, max(0.0, float(epsilon)))
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified.pop()

    # Increase epsilon progressively when a very detailed mask still exceeds the
    # UI-safe point count.
    current_epsilon = max(float(epsilon), 0.5)
    while len(simplified) > max(3, int(max_points)):
        current_epsilon *= 1.35
        simplified = _rdp_open(rotated, current_epsilon)
        if len(simplified) > 1 and simplified[0] == simplified[-1]:
            simplified.pop()
        if current_epsilon > 64.0:
            break

    if len(simplified) < 3:
        return []
    return [[round(x, 3), round(y, 3)] for x, y in simplified]


class SamAnnotationEngine:
    """Lazily loaded, process-local SAM inference engine."""

    def __init__(self, project_root: Path, model_factory: Callable[[str], Any] | None = None) -> None:
        self.project_root = Path(project_root)
        self._model_factory = model_factory
        self._model: Any = None
        self._model_path: str = ""
        self._lock = threading.Lock()

    def _configured_model_value(self) -> str:
        return str(os.environ.get(SAM_MODEL_ENV, "")).strip()

    def _candidate_paths(self) -> list[Path]:
        configured = self._configured_model_value()
        if configured:
            path = Path(configured).expanduser()
            return [path if path.is_absolute() else self.project_root / path]
        return [self.project_root / relative for relative in DEFAULT_MODEL_CANDIDATES]

    def resolve_model(self) -> tuple[str, bool]:
        """Return model source and whether it is already available locally."""

        configured = self._configured_model_value()
        for path in self._candidate_paths():
            if path.is_file():
                return str(path.resolve()), True
        if configured and _truthy(os.environ.get(SAM_ALLOW_DOWNLOAD_ENV), default=False):
            # Ultralytics may download a recognised model name when explicitly
            # permitted. Absolute/nonexistent paths remain a configuration error.
            configured_path = Path(configured)
            if not configured_path.is_absolute() and configured_path.parent == Path("."):
                return configured, False
        return "", False

    def status(self) -> dict[str, Any]:
        model_source, local = self.resolve_model()
        candidates = [str(path) for path in self._candidate_paths()]
        return {
            "available": bool(model_source) and importlib.util.find_spec("ultralytics") is not None,
            "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None,
            "model_source": model_source,
            "model_local": local,
            "model_candidates": candidates,
            "device": str(os.environ.get(SAM_DEVICE_ENV, "")).strip() or "auto",
            "loaded": self._model is not None,
            "loaded_model_source": self._model_path,
            "mode": "box_prompt",
        }

    def _load_model_locked(self) -> Any:
        source, _ = self.resolve_model()
        if not source:
            candidates = "、".join(str(path) for path in self._candidate_paths())
            raise RuntimeError(
                "SAM 模型尚未配置。请把 sam2.1_s.pt 或 mobile_sam.pt 放到 models/pretrained，"
                f"或设置 {SAM_MODEL_ENV}。当前候选: {candidates}"
            )
        if self._model is not None and self._model_path == source:
            return self._model
        if importlib.util.find_spec("ultralytics") is None and self._model_factory is None:
            raise RuntimeError("当前服务端 Python 环境未安装 ultralytics，无法使用 SAM 智能框选。")
        if self._model_factory is None:
            from ultralytics import SAM  # type: ignore

            factory: Callable[[str], Any] = SAM
        else:
            factory = self._model_factory
        self._model = factory(source)
        self._model_path = source
        return self._model

    @staticmethod
    def _extract_polygons(results: Any) -> list[list[list[float]]]:
        if results is None:
            return []
        if not isinstance(results, (list, tuple)):
            try:
                results = list(results)
            except TypeError:
                results = [results]
        polygons: list[list[list[float]]] = []
        for result in results:
            masks = getattr(result, "masks", None)
            xy = getattr(masks, "xy", None) if masks is not None else None
            if xy is None:
                continue
            for polygon in xy:
                try:
                    points = polygon.tolist() if hasattr(polygon, "tolist") else list(polygon)
                except Exception:
                    continue
                cleaned: list[list[float]] = []
                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    cleaned.append([float(point[0]), float(point[1])])
                if len(cleaned) >= 3:
                    polygons.append(cleaned)
        return polygons

    def predict_box(self, image_path: Path, bbox: Iterable[float], image_size: tuple[int, int]) -> dict[str, Any]:
        values = [float(value) for value in bbox]
        if len(values) != 4:
            raise ValueError("bbox 必须包含 x1,y1,x2,y2 四个值")
        image_w, image_h = image_size
        x1, x2 = sorted((values[0], values[2]))
        y1, y2 = sorted((values[1], values[3]))
        x1 = max(0.0, min(float(image_w), x1))
        x2 = max(0.0, min(float(image_w), x2))
        y1 = max(0.0, min(float(image_h), y1))
        y2 = max(0.0, min(float(image_h), y2))
        if x2 - x1 < 4.0 or y2 - y1 < 4.0:
            raise ValueError("SAM 提示框过小，请完整框住一个目标")

        device = str(os.environ.get(SAM_DEVICE_ENV, "")).strip()
        with self._lock:
            model = self._load_model_locked()
            kwargs: dict[str, Any] = {
                "source": str(image_path),
                "bboxes": [x1, y1, x2, y2],
                "verbose": False,
            }
            if device:
                kwargs["device"] = device
            try:
                results = model.predict(**kwargs)
            except AttributeError:
                results = model(**kwargs)
            except TypeError:
                # Compatibility with older Ultralytics SAM call signatures.
                source = kwargs.pop("source")
                results = model(source, **kwargs)

        polygons = self._extract_polygons(results)
        if not polygons:
            raise RuntimeError("SAM 没有返回有效掩膜，请扩大提示框或更换模型")
        polygon = max(polygons, key=_polygon_area)
        area_before = _polygon_area(polygon)
        if area_before < 12.0:
            raise RuntimeError("SAM 返回的掩膜面积过小，请重新框选")

        diagonal = math.hypot(float(image_w), float(image_h))
        simplify_ratio = float(os.environ.get(SAM_SIMPLIFY_RATIO_ENV, "0.0015"))
        epsilon = max(0.75, diagonal * max(0.0001, simplify_ratio))
        max_points = max(16, int(os.environ.get(SAM_MAX_POINTS_ENV, "160")))
        simplified = simplify_closed_polygon(polygon, epsilon=epsilon, max_points=max_points)
        if len(simplified) < 3:
            raise RuntimeError("SAM 掩膜轮廓简化后无效，请重新框选")

        # Clamp points because a few model/runtime combinations can return tiny
        # negative or out-of-bound coordinates after resizing.
        simplified = [
            [
                max(0.0, min(float(image_w), float(point[0]))),
                max(0.0, min(float(image_h), float(point[1]))),
            ]
            for point in simplified
        ]
        return {
            "status": "ok",
            "mode": "box_prompt",
            "model_source": self._model_path,
            "bbox": [x1, y1, x2, y2],
            "polygon": simplified,
            "point_count": len(simplified),
            "mask_area_px": round(_polygon_area(simplified), 3),
            "raw_mask_area_px": round(area_before, 3),
        }
