"""Capture ROI metadata normalization and compatibility helpers.

M32.1 writes ``capture_manifest.json`` on the edge.  M32.2 keeps that
metadata intact through batch ingest, dataset creation, training jobs and the
final ``model.yaml``.  The helpers in this module deliberately do not perform
image processing; they only validate and normalize the metadata contract.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RESIZE_MODE = "letterbox"
DEFAULT_PAD_VALUE = 114
SUPPORTED_COORDINATE_SPACES = {"runtime_snapshot", "bridge_output_image", "image"}


def disabled_input_roi() -> dict[str, Any]:
    return {
        "enabled": False,
    }


def _finite_float(value: Any, name: str) -> float:
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


def _resolution(value: Any, name: str, *, allow_zero: bool = False) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    return {
        "width": _positive_int(raw.get("width", 0), f"{name}.width", allow_zero=allow_zero),
        "height": _positive_int(raw.get("height", 0), f"{name}.height", allow_zero=allow_zero),
    }


def _list4(value: Any, name: str, *, integer: bool) -> list[int] | list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} 必须是长度为 4 的数组")
    if integer:
        return [_positive_int(item, f"{name}[{index}]", allow_zero=True) for index, item in enumerate(value)]
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]


def normalize_input_roi(value: Any, *, allow_disabled: bool = True) -> dict[str, Any]:
    """Normalize a model input ROI to the M32.2 deployment contract."""

    raw = value if isinstance(value, dict) else {}
    if isinstance(raw.get("input_roi"), dict):
        raw = raw["input_roi"]
    if isinstance(raw.get("capture_roi"), dict):
        raw = raw["capture_roi"]

    enabled = raw.get("enabled") is True
    if not enabled:
        if not allow_disabled:
            raise ValueError("input_roi 必须启用")
        return disabled_input_roi()

    coordinate_space = str(raw.get("coordinate_space") or "runtime_snapshot").strip()
    if coordinate_space not in SUPPORTED_COORDINATE_SPACES:
        raise ValueError(f"不支持的 input_roi.coordinate_space: {coordinate_space}")

    source = _resolution(raw.get("source_resolution"), "input_roi.source_resolution")
    pixel = _list4(raw.get("pixel_xyxy"), "input_roi.pixel_xyxy", integer=True)
    normalized = _list4(raw.get("normalized_xyxy"), "input_roi.normalized_xyxy", integer=False)
    x1, y1, x2, y2 = [int(item) for item in pixel]
    nx1, ny1, nx2, ny2 = [float(item) for item in normalized]

    if not (0 <= x1 < x2 <= source["width"] and 0 <= y1 < y2 <= source["height"]):
        raise ValueError(
            "input_roi.pixel_xyxy 必须位于 source_resolution 内，且满足 x1<x2、y1<y2"
        )
    if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
        raise ValueError("input_roi.normalized_xyxy 必须满足 0<=x1<x2<=1 且 0<=y1<y2<=1")

    expected_crop = {"width": x2 - x1, "height": y2 - y1}
    crop_raw = raw.get("crop_resolution")
    if isinstance(crop_raw, dict) and crop_raw:
        crop = _resolution(crop_raw, "input_roi.crop_resolution")
        if crop != expected_crop:
            raise ValueError(
                "input_roi.crop_resolution 与 pixel_xyxy 不一致: "
                f"expected={expected_crop['width']}x{expected_crop['height']}, "
                f"actual={crop['width']}x{crop['height']}"
            )
    else:
        crop = expected_crop

    # Pixel coordinates are authoritative.  Keep the normalized values from
    # the edge package, but verify they are consistent within one source pixel.
    expected_norm = [
        x1 / source["width"],
        y1 / source["height"],
        x2 / source["width"],
        y2 / source["height"],
    ]
    tolerances = [1.0 / source["width"], 1.0 / source["height"], 1.0 / source["width"], 1.0 / source["height"]]
    for index, (actual, expected, tolerance) in enumerate(zip([nx1, ny1, nx2, ny2], expected_norm, tolerances)):
        if abs(actual - expected) > tolerance + 1e-9:
            raise ValueError(
                f"input_roi.normalized_xyxy[{index}] 与 pixel_xyxy/source_resolution 不一致"
            )

    resize_mode = str(raw.get("resize_mode") or DEFAULT_RESIZE_MODE).strip().lower()
    if resize_mode not in {"letterbox", "resize"}:
        raise ValueError(f"不支持的 input_roi.resize_mode: {resize_mode}")
    pad_value = _positive_int(raw.get("pad_value", DEFAULT_PAD_VALUE), "input_roi.pad_value", allow_zero=True)
    if pad_value > 255:
        raise ValueError("input_roi.pad_value 必须位于 0~255")

    return {
        "enabled": True,
        "coordinate_space": coordinate_space,
        "source_resolution": source,
        "pixel_xyxy": [x1, y1, x2, y2],
        "normalized_xyxy": expected_norm,
        "crop_resolution": crop,
        "resize_mode": resize_mode,
        "pad_value": pad_value,
    }


def normalize_capture_metadata(
    capture_manifest: Any = None,
    main_manifest: Any = None,
    *,
    source_name: str = "",
) -> dict[str, Any]:
    """Normalize M32.1 capture metadata, with a legacy full-frame fallback."""

    capture_doc = capture_manifest if isinstance(capture_manifest, dict) else {}
    main_doc = main_manifest if isinstance(main_manifest, dict) else {}

    capture_section = main_doc.get("capture") if isinstance(main_doc.get("capture"), dict) else {}
    roi_candidate: Any = None
    for candidate in (
        capture_doc.get("capture_roi"),
        capture_doc.get("input_roi"),
        capture_section.get("capture_roi"),
        capture_section.get("input_roi"),
        main_doc.get("capture_roi"),
        main_doc.get("input_roi"),
    ):
        if isinstance(candidate, dict):
            roi_candidate = candidate
            break

    explicit_cropped: Any = None
    for candidate in (
        capture_doc.get("images_are_cropped"),
        capture_section.get("images_are_cropped"),
        main_doc.get("images_are_cropped"),
    ):
        if isinstance(candidate, bool):
            explicit_cropped = candidate
            break

    if roi_candidate is None:
        input_roi = disabled_input_roi()
    else:
        input_roi = normalize_input_roi(roi_candidate)

    images_are_cropped = bool(explicit_cropped) if explicit_cropped is not None else bool(input_roi.get("enabled"))
    if images_are_cropped and not input_roi.get("enabled"):
        raise ValueError("上传包声明 images_are_cropped=true，但没有有效的 capture_roi/input_roi")
    if input_roi.get("enabled") and not images_are_cropped:
        raise ValueError("上传包包含启用的 capture_roi，但声明 images_are_cropped=false")

    return {
        "schema_version": "1.0",
        "message_type": "capture_metadata",
        "source_name": str(source_name or ""),
        "images_are_cropped": images_are_cropped,
        "input_roi": input_roi,
    }


def input_roi_signature(value: Any) -> tuple[Any, ...]:
    roi = normalize_input_roi(value)
    if not roi.get("enabled"):
        return (False,)
    source = roi["source_resolution"]
    return (
        True,
        roi.get("coordinate_space"),
        source["width"],
        source["height"],
        *roi["pixel_xyxy"],
        roi.get("resize_mode"),
        roi.get("pad_value"),
    )


def resolve_common_input_roi(values: Iterable[Any], *, context: str = "数据集") -> dict[str, Any]:
    """Return one ROI when every source is compatible, otherwise fail early."""

    normalized = [normalize_input_roi(value) for value in values]
    if not normalized:
        return disabled_input_roi()
    signatures = {input_roi_signature(item) for item in normalized}
    if len(signatures) != 1:
        descriptions = ", ".join(describe_input_roi(item) for item in normalized)
        raise ValueError(
            f"{context}包含不一致的采集 ROI，无法生成唯一的模型输入 ROI: {descriptions}"
        )
    return copy.deepcopy(normalized[0])


def model_preprocess_document(input_roi: Any) -> dict[str, Any]:
    return {"input_roi": normalize_input_roi(input_roi)}


def describe_input_roi(value: Any) -> str:
    roi = normalize_input_roi(value)
    if not roi.get("enabled"):
        return "full-frame"
    source = roi["source_resolution"]
    crop = roi["crop_resolution"]
    pixel = roi["pixel_xyxy"]
    return (
        f"roi={crop['width']}x{crop['height']}@{source['width']}x{source['height']}"
        f"[{pixel[0]},{pixel[1]},{pixel[2]},{pixel[3]}]"
    )


def read_json_object(path: Path, *, strict: bool = False) -> dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        if strict:
            raise ValueError(f"JSON 文件格式错误: {path}: {error}") from error
        return {}
    if not isinstance(value, dict):
        if strict:
            raise ValueError(f"JSON 文件顶层必须是对象: {path}")
        return {}
    return value
