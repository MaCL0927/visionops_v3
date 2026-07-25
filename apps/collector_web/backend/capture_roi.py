"""采集链路 ROI 配置与坐标换算。

本模块只服务于数据采集：
- 采集 ROI 决定手动拍照、定时采图和本地下载时保存哪一块图像；
- 它与 Runtime 的检测结果 ROI 完全独立；
- 配置使用归一化坐标持久化，同时保存绘制时的源分辨率和像素坐标。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from .vision_box_settings import DEFAULT_PROJECT_ROOT

CAPTURE_ROI_CONFIG_PATH = Path(
    os.environ.get(
        "VISIONOPS_CAPTURE_ROI_CONFIG_FILE",
        str(DEFAULT_PROJECT_ROOT / "data" / "capture_roi.json"),
    )
)
MIN_ROI_EDGE_PIXELS = 32
_LOCK = threading.RLock()


def _default_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "message_type": "capture_roi_config",
        "enabled": False,
        "coordinate_space": "runtime_snapshot",
        "source_resolution": {"width": 0, "height": 0},
        "normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
        "pixel_xyxy": [0, 0, 0, 0],
        "crop_resolution": {"width": 0, "height": 0},
        "updated_at_ms": 0,
    }


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是有效数字") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} 不是有限数字")
    return number


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是有效整数") from error
    minimum = 0 if allow_zero else 1
    if number < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return number


def _extract_source_resolution(payload: dict[str, Any]) -> tuple[int, int]:
    source = payload.get("source_resolution") if isinstance(payload.get("source_resolution"), dict) else {}
    width = source.get("width", payload.get("source_width", payload.get("image_width", 0)))
    height = source.get("height", payload.get("source_height", payload.get("image_height", 0)))
    return (
        _positive_int(width, "source_resolution.width", allow_zero=True),
        _positive_int(height, "source_resolution.height", allow_zero=True),
    )


def _extract_normalized(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("normalized_xyxy")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = list(raw)
    else:
        values = [payload.get("x1"), payload.get("y1"), payload.get("x2"), payload.get("y2")]
    if any(value is None for value in values):
        return [0.0, 0.0, 1.0, 1.0]
    return [_finite_number(value, f"normalized_xyxy[{index}]") for index, value in enumerate(values)]


def normalize_capture_roi(payload: dict[str, Any] | None) -> dict[str, Any]:
    """将 API/文件中的 ROI 转换为唯一的持久化格式。"""

    raw = payload if isinstance(payload, dict) else {}
    if isinstance(raw.get("capture_roi"), dict):
        raw = raw["capture_roi"]

    enabled = raw.get("enabled") is True
    source_width, source_height = _extract_source_resolution(raw)
    normalized = _extract_normalized(raw)
    x1, y1, x2, y2 = normalized

    if not enabled:
        pixel = [0, 0, source_width, source_height] if source_width and source_height else [0, 0, 0, 0]
        return {
            "schema_version": "1.0",
            "message_type": "capture_roi_config",
            "enabled": False,
            "coordinate_space": "runtime_snapshot",
            "source_resolution": {"width": source_width, "height": source_height},
            "normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
            "pixel_xyxy": pixel,
            "crop_resolution": {"width": source_width, "height": source_height},
            "updated_at_ms": int(raw.get("updated_at_ms") or 0),
        }

    if source_width <= 0 or source_height <= 0:
        raise ValueError("启用采集 ROI 时必须提供有效的源图像宽高")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("采集 ROI 归一化坐标必须满足 0<=x1<x2<=1 且 0<=y1<y2<=1")

    # 将浏览器浮点坐标固化为源图像像素边界，之后所有采图使用同一边界。
    px1 = max(0, min(source_width - 1, int(round(x1 * source_width))))
    py1 = max(0, min(source_height - 1, int(round(y1 * source_height))))
    px2 = max(px1 + 1, min(source_width, int(round(x2 * source_width))))
    py2 = max(py1 + 1, min(source_height, int(round(y2 * source_height))))
    crop_width = px2 - px1
    crop_height = py2 - py1
    if crop_width < MIN_ROI_EDGE_PIXELS or crop_height < MIN_ROI_EDGE_PIXELS:
        raise ValueError(f"采集 ROI 宽高不能小于 {MIN_ROI_EDGE_PIXELS} 像素")

    return {
        "schema_version": "1.0",
        "message_type": "capture_roi_config",
        "enabled": True,
        "coordinate_space": "runtime_snapshot",
        "source_resolution": {"width": source_width, "height": source_height},
        "normalized_xyxy": [
            px1 / source_width,
            py1 / source_height,
            px2 / source_width,
            py2 / source_height,
        ],
        "pixel_xyxy": [px1, py1, px2, py2],
        "crop_resolution": {"width": crop_width, "height": crop_height},
        "updated_at_ms": int(raw.get("updated_at_ms") or 0),
    }


def load_capture_roi_config(path: Path | None = None) -> dict[str, Any]:
    target = path or CAPTURE_ROI_CONFIG_PATH
    with _LOCK:
        if not target.exists():
            return _default_config()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            return normalize_capture_roi(raw)
        except Exception as error:
            # 配置损坏时不能静默使用错误裁剪区域。将错误暴露给 API/采集调用方。
            raise ValueError(f"读取采集 ROI 配置失败: {error}") from error


def save_capture_roi_config(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = path or CAPTURE_ROI_CONFIG_PATH
    normalized = normalize_capture_roi(payload)
    normalized["updated_at_ms"] = int(time.time_ns() // 1_000_000)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with _LOCK:
        temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    return normalized


def capture_roi_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_capture_roi(config)
    if not normalized["enabled"]:
        # 关闭状态下源分辨率仅用于界面显示，不改变“保存完整图”的采集语义。
        return (False,)
    source = normalized["source_resolution"]
    return (
        True,
        source["width"],
        source["height"],
        *(round(float(value), 9) for value in normalized["normalized_xyxy"]),
    )


def resolve_capture_roi_for_image(
    config: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    allow_scaled_resolution: bool = False,
    aspect_tolerance: float = 0.01,
) -> dict[str, Any]:
    """把持久化 ROI 映射到本次 Runtime 快照的像素坐标。"""

    width = _positive_int(image_width, "image_width")
    height = _positive_int(image_height, "image_height")
    normalized = normalize_capture_roi(config)
    if not normalized["enabled"]:
        return {
            **normalized,
            "source_resolution": {"width": width, "height": height},
            "pixel_xyxy": [0, 0, width, height],
            "crop_resolution": {"width": width, "height": height},
        }

    configured_source = normalized["source_resolution"]
    configured_width = int(configured_source["width"])
    configured_height = int(configured_source["height"])
    if configured_width > 0 and configured_height > 0 and (configured_width != width or configured_height != height):
        if not allow_scaled_resolution:
            raise ValueError(
                "当前快照分辨率与绘制采集 ROI 时不一致，请重新绘制采集 ROI: "
                f"configured={configured_width}x{configured_height}, current={width}x{height}"
            )
        configured_aspect = configured_width / configured_height
        current_aspect = width / height
        if abs(configured_aspect - current_aspect) / configured_aspect > aspect_tolerance:
            raise ValueError(
                "当前快照宽高比与绘制采集 ROI 时不一致: "
                f"configured={configured_width}x{configured_height}, current={width}x{height}"
            )

    x1, y1, x2, y2 = normalized["normalized_xyxy"]
    px1 = max(0, min(width - 1, int(round(x1 * width))))
    py1 = max(0, min(height - 1, int(round(y1 * height))))
    px2 = max(px1 + 1, min(width, int(round(x2 * width))))
    py2 = max(py1 + 1, min(height, int(round(y2 * height))))
    if px2 - px1 < MIN_ROI_EDGE_PIXELS or py2 - py1 < MIN_ROI_EDGE_PIXELS:
        raise ValueError("映射后的采集 ROI 太小，无法保存")
    return {
        **normalized,
        "source_resolution": {"width": width, "height": height},
        "pixel_xyxy": [px1, py1, px2, py2],
        "crop_resolution": {"width": px2 - px1, "height": py2 - py1},
    }
