"""Runtime overlay helper for retained side-ring diagnostics.

This module intentionally contains visualization only. Historical offline replay,
CSV export and dataset traversal code was removed during the M39.3.4.1 production
cleanup. The no-ring_mouth production branch remains disabled until M39.4.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from .segmentation import SegmentationInstance

def _project_polyline(
    points: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = points[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = (
        float(intrinsics["fx"]) * points[valid, 0] / points[valid, 2]
        + float(intrinsics["cx"])
    )
    pixels[valid, 1] = (
        float(intrinsics["fy"]) * points[valid, 1] / points[valid, 2]
        + float(intrinsics["cy"])
    )
    return pixels


def draw_side_ring_fit_overlay(
    rgb_bgr: np.ndarray,
    instances: Sequence[SegmentationInstance],
    fits: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, float],
    selected_instance_id: int | None,
) -> np.ndarray:
    overlay = rgb_bgr.copy()
    instance_map = {int(item.instance_id): item for item in instances}
    for fit in fits:
        instance_id = int(fit.get("ring_instance_id", -1))
        instance = instance_map.get(instance_id)
        if instance is None:
            continue
        processing_status = str(fit.get("processing_status") or "evaluated")
        eligible_value = fit.get("eligible")
        eligible = bool(eligible_value) if eligible_value is not None else False
        selected = selected_instance_id == instance_id
        if processing_status == "deferred":
            color = (128, 128, 128)
        else:
            color = (0, 255, 0) if eligible else (0, 165, 255)
        if selected:
            color = (255, 255, 0)
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
        x1, y1, _, _ = instance.bbox_xyxy
        if processing_status == "deferred":
            label = "S%d D %.2f" % (
                instance_id, float(fit.get("ring_confidence", 0.0))
            )
        else:
            label = "S%d %.2f" % (instance_id, float(fit.get("fit_score", 0.0)))
        cv2.putText(
            overlay,
            label,
            (int(x1), max(12, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        if fit.get("center_uv") is None:
            continue
        center = tuple(int(round(float(value))) for value in fit["center_uv"])
        near = tuple(int(round(float(value))) for value in fit["near_opening_center_uv"])
        far = tuple(int(round(float(value))) for value in fit["far_opening_center_uv"])
        crown = fit.get("near_side_crown") or fit.get("top_arc") or {}
        grasp = tuple(
            int(round(float(value)))
            for value in crown.get("grasp_point_uv", [0, 0])
        )
        legacy_rim = fit.get("near_opening_rim_top_diagnostic") or {}
        legacy_uv_raw = legacy_rim.get("point_uv")
        cv2.line(overlay, far, near, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(overlay, center, near, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(overlay, near, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, far, 4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(
            overlay,
            (grasp[0] - 5, grasp[1] - 5),
            (grasp[0] + 5, grasp[1] + 5),
            (0, 0, 255),
            2,
        )
        if isinstance(legacy_uv_raw, (list, tuple)) and len(legacy_uv_raw) >= 2:
            legacy_uv = tuple(int(round(float(value))) for value in legacy_uv_raw[:2])
            cv2.drawMarker(
                overlay,
                legacy_uv,
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=8,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
        visible_arc = crown.get("visible_arc") or {}
        for key in ("upper_endpoint_uv", "lower_endpoint_uv"):
            endpoint = visible_arc.get(key)
            if isinstance(endpoint, (list, tuple)) and len(endpoint) >= 2:
                point = tuple(int(round(float(value))) for value in endpoint[:2])
                cv2.circle(overlay, point, 3, (255, 0, 0), -1, cv2.LINE_AA)
        debug = fit.get("_debug") or {}
        for key, circle_color in (
            ("near_outer_circle_camera_mm", (0, 255, 255)),
            ("near_inner_circle_camera_mm", (255, 255, 0)),
            ("grasp_outer_circle_camera_mm", (0, 165, 255)),
        ):
            points = debug.get(key)
            if points is None:
                continue
            pixels = _project_polyline(np.asarray(points), intrinsics)
            finite = np.isfinite(pixels).all(axis=1)
            if int(np.count_nonzero(finite)) >= 3:
                contour = np.rint(pixels[finite]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [contour], True, circle_color, 1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "M37.6: hollow-cylinder outer/inner/face joint fit; gray=deferred",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay

