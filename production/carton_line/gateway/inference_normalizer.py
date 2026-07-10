"""Convert the v3 standard inference_result into the proven v2 algorithm payload."""

from __future__ import annotations

from typing import Any, Mapping


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return default


def normalize_inference_result(result: Mapping[str, Any]) -> dict[str, Any]:
    image = result.get("image") if isinstance(result.get("image"), Mapping) else {}
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(detections):
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        box = [_number(value) for value in bbox[:4]]
        center = item.get("center_xy")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            cx, cy = _number(center[0]), _number(center[1])
        else:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        prediction = {
            "id": str(item.get("id") or f"det-{index}"),
            "class_id": item.get("class_id"),
            "class_name": str(item.get("class_name") or ""),
            "confidence": _number(item.get("score")),
            "score": _number(item.get("score")),
            "bbox": box,
            "center": [cx, cy],
            "center_x": cx,
            "center_y": cy,
            "cx": cx,
            "cy": cy,
        }
        obb = item.get("obb")
        if isinstance(obb, Mapping):
            prediction["obb"] = dict(obb)
            for source, target in (("w", "width"), ("h", "height"), ("angle_deg", "angle_deg")):
                if source in obb:
                    prediction[target] = obb[source]
        predictions.append(prediction)

    width = int(_number(image.get("width")))
    height = int(_number(image.get("height")))
    return {
        "schema_version": "1.0",
        "image_width": width,
        "image_height": height,
        "width": width,
        "height": height,
        "predictions": predictions,
        "raw": {
            "image_width": width,
            "image_height": height,
            "predictions": predictions,
            "runtime_result": dict(result),
        },
        "runtime_result": dict(result),
    }
