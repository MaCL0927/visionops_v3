"""M39.3.0 offline tilt-evidence detector for Branch-A foam rings.

M39.3.0 deliberately does *not* route production grasps.  It answers a more
limited question before M39.2.9's global near/far depth split is applied:

    Does the raw front annulus contain coherent evidence of a real tilt?

The detector combines two different cues:

1. A raw-annulus plane fit supplied by the caller.  This cue is intentionally
   auxiliary because M39.2.7 showed that a low-residual RANSAC plane can still
   have a false normal.
2. A ring-aware sector gradient.  A boundary-shrunk annulus core is divided
   into angular sectors.  Each sector contributes one robust height sample,
   and a first-harmonic model is fitted around the ring.  A real tilted ring
   should create a coherent approximately sinusoidal height pattern rather
   than one locally dense crescent dominating the fit.

Only consensus between the raw-plane cue and the ring-aware sector cue may
produce ``TILTED``.  Contradictory or severely incoherent data becomes
``UNCERTAIN`` or ``FLAT`` with conflict diagnostics.  No result from this
module changes the M39.2.9 production plane in M39.3.0.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return value / norm


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(np.asarray(a, dtype=np.float64))
    bb = _normalize(np.asarray(b, dtype=np.float64))
    cosine = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _circular_delta_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def build_tilt_core_mask(
    raw_annulus_mask: np.ndarray,
    ring_mask: np.ndarray,
    mouth_mask: np.ndarray,
    *,
    outer_boundary_margin_px: float = 3.0,
    inner_boundary_margin_px: float = 3.0,
) -> np.ndarray:
    """Shrink both physical annulus boundaries before tilt analysis.

    ``raw_annulus_mask`` is the already depth-edge/neighbor-cleaned M38.1 mask.
    M39.3.0 further removes pixels close to the segmented outer foam edge and
    the opening edge.  These are precisely where D2C registration and vertical
    side-wall depth are least reliable.
    """
    raw = np.asarray(raw_annulus_mask, dtype=bool)
    ring = np.asarray(ring_mask, dtype=bool)
    mouth = np.asarray(mouth_mask, dtype=bool)
    if raw.shape != ring.shape or raw.shape != mouth.shape:
        raise ValueError("annulus/ring/mouth masks must have identical shapes")

    ring_u8 = ring.astype(np.uint8)
    outside_mouth_u8 = (~mouth).astype(np.uint8)
    distance_to_outer = cv2.distanceTransform(ring_u8, cv2.DIST_L2, 5)
    distance_from_mouth = cv2.distanceTransform(outside_mouth_u8, cv2.DIST_L2, 5)
    return (
        raw
        & (distance_to_outer >= max(0.0, float(outer_boundary_margin_px)))
        & (distance_from_mouth >= max(0.0, float(inner_boundary_margin_px)))
    )


def _sector_index(
    pixels_xy: np.ndarray,
    center_uv: Tuple[float, float],
    sector_count: int,
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    dx = pixels[:, 0] - float(center_uv[0])
    dy = pixels[:, 1] - float(center_uv[1])
    angles = (np.arctan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)
    indexes = np.floor(angles * float(sector_count) / (2.0 * math.pi)).astype(np.int32)
    return np.clip(indexes, 0, int(sector_count) - 1)


def extract_sector_height_samples(
    points_camera_mm: np.ndarray,
    pixels_xy: np.ndarray,
    center_uv: Tuple[float, float],
    *,
    box_x_camera: Sequence[float],
    box_y_camera: Sequence[float],
    box_z_inside_camera: Sequence[float],
    sector_count: int = 16,
    minimum_sector_points: int = 5,
    local_band_half_width_mm: float = 8.0,
) -> list[Dict[str, Any]]:
    """Collapse the boundary-shrunk annulus to one robust sample per sector.

    No global near/far clustering is performed here.  That is intentional:
    a real ~15 degree tilt across a ~100 mm ring naturally creates about
    25-30 mm of height change, which M39.2.9's global layer split can mistake
    for a separate far layer.
    """
    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) != len(pixels):
        raise ValueError("points and pixels must have the same length")
    if not len(points):
        return []

    sectors = max(8, int(sector_count))
    minimum = max(3, int(minimum_sector_points))
    band = max(1.0, float(local_band_half_width_mm))
    x_axis = _normalize(np.asarray(box_x_camera, dtype=np.float64))
    y_axis = _normalize(np.asarray(box_y_camera, dtype=np.float64))
    z_inside = _normalize(np.asarray(box_z_inside_camera, dtype=np.float64))

    indexes = _sector_index(pixels, center_uv, sectors)
    x_values = points @ x_axis
    y_values = points @ y_axis
    heights = points @ z_inside
    rows: list[Dict[str, Any]] = []
    for sector in range(sectors):
        selection = np.flatnonzero(indexes == sector)
        raw_count = int(len(selection))
        if raw_count < minimum:
            continue
        raw_heights = heights[selection]
        center = float(np.median(raw_heights))
        supported_local = selection[np.abs(raw_heights - center) <= band]
        if len(supported_local) < minimum:
            continue
        supported_heights = heights[supported_local]
        rows.append({
            "sector": int(sector),
            "sector_angle_deg_image": float((float(sector) + 0.5) * 360.0 / float(sectors)),
            "raw_point_count": raw_count,
            "supported_point_count": int(len(supported_local)),
            "height_mm": float(np.median(supported_heights)),
            "height_mad_mm": float(np.median(np.abs(supported_heights - np.median(supported_heights)))),
            "x_box_mm": float(np.median(x_values[supported_local])),
            "y_box_mm": float(np.median(y_values[supported_local])),
            "representative_uv": [
                float(np.median(pixels[supported_local, 0])),
                float(np.median(pixels[supported_local, 1])),
            ],
        })
    return rows


def _fit_first_harmonic(
    sector_samples: Sequence[Mapping[str, Any]],
    *,
    sector_count: int,
    residual_threshold_mm: float,
) -> Optional[Dict[str, Any]]:
    rows = list(sector_samples)
    if len(rows) < 5:
        return None
    sectors = np.asarray([int(row["sector"]) for row in rows], dtype=np.int32)
    heights = np.asarray([float(row["height_mm"]) for row in rows], dtype=np.float64)
    theta = (sectors.astype(np.float64) + 0.5) * 2.0 * math.pi / float(sector_count)
    design = np.column_stack((np.ones(len(rows)), np.cos(theta), np.sin(theta)))
    threshold = max(0.5, float(residual_threshold_mm))

    best: Optional[Tuple[Tuple[int, float], np.ndarray, np.ndarray]] = None
    for triplet in combinations(range(len(rows)), 3):
        matrix = design[list(triplet)]
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            continue
        coefficients = np.linalg.solve(matrix, heights[list(triplet)])
        residuals = np.abs(heights - design @ coefficients)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residuals[inliers])) if count else float("inf")
        score = (count, -median)
        if best is None or score > best[0]:
            best = (score, inliers, coefficients)
    if best is None:
        return None

    inliers = best[1]
    coefficients = np.linalg.lstsq(design[inliers], heights[inliers], rcond=None)[0]
    residuals = np.abs(heights - design @ coefficients)
    inliers = residuals <= threshold
    if int(np.count_nonzero(inliers)) >= 3:
        coefficients = np.linalg.lstsq(design[inliers], heights[inliers], rcond=None)[0]
        residuals = np.abs(heights - design @ coefficients)
        inliers = residuals <= threshold

    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    direction = float((math.degrees(math.atan2(coefficients[2], coefficients[1])) + 360.0) % 360.0)
    xy = np.asarray(
        [[float(row["x_box_mm"]), float(row["y_box_mm"])] for row in rows],
        dtype=np.float64,
    )
    center_xy = np.median(xy, axis=0)
    radius = float(np.median(np.linalg.norm(xy - center_xy[None, :], axis=1)))
    gradient_tilt = float(math.degrees(math.atan2(amplitude, max(radius, 1e-9))))
    flat_center = float(np.median(heights))
    flat_residuals = np.abs(heights - flat_center)

    inlier_sectors = sorted(int(value) for value in sectors[inliers])
    max_gap_sectors = sector_count
    if inlier_sectors:
        wrapped = inlier_sectors[1:] + [inlier_sectors[0] + sector_count]
        max_gap_sectors = max(b - a for a, b in zip(inlier_sectors, wrapped))
    inlier_coverage_deg = float(
        max(0.0, 360.0 - float(max_gap_sectors) * 360.0 / float(sector_count) + 360.0 / float(sector_count))
    )

    used = set(inlier_sectors)
    serialized_rows = []
    for row, residual in zip(rows, residuals.tolist()):
        item = dict(row)
        item["harmonic_residual_mm"] = float(residual)
        item["harmonic_inlier"] = bool(int(item["sector"]) in used)
        serialized_rows.append(item)

    return {
        "model": "h(theta)=h0+a*cos(theta)+b*sin(theta)",
        "h0_mm": float(coefficients[0]),
        "cos_amplitude_mm": float(coefficients[1]),
        "sin_amplitude_mm": float(coefficients[2]),
        "amplitude_mm": amplitude,
        "predicted_peak_to_peak_mm": float(2.0 * amplitude),
        "representative_radius_mm": radius,
        "sector_gradient_tilt_deg": gradient_tilt,
        "gradient_direction_deg_image": direction,
        "valid_sector_count": int(len(rows)),
        "inlier_sector_count": int(np.count_nonzero(inliers)),
        "inlier_sectors": inlier_sectors,
        "inlier_angular_coverage_deg": inlier_coverage_deg,
        "residual_threshold_mm": threshold,
        "residual_median_mm": float(np.median(residuals)),
        "residual_p90_mm": float(np.percentile(residuals, 90)),
        "inlier_residual_median_mm": (
            float(np.median(residuals[inliers])) if int(np.count_nonzero(inliers)) else None
        ),
        "flat_residual_median_mm": float(np.median(flat_residuals)),
        "flat_residual_p90_mm": float(np.percentile(flat_residuals, 90)),
        "sector_samples": serialized_rows,
    }


def _raw_plane_cue(
    raw_plane_normal_toward_camera: Optional[Sequence[float]],
    *,
    box_x_camera: np.ndarray,
    box_y_camera: np.ndarray,
    box_z_inside_camera: np.ndarray,
    raw_plane_inlier_ratio: Optional[float],
    raw_plane_residual_p95_mm: Optional[float],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "tilt_deg": None,
        "gradient_direction_deg_box": None,
        "inlier_ratio": raw_plane_inlier_ratio,
        "residual_p95_mm": raw_plane_residual_p95_mm,
    }
    if raw_plane_normal_toward_camera is None:
        return result
    try:
        normal = _normalize(np.asarray(raw_plane_normal_toward_camera, dtype=np.float64))
    except (TypeError, ValueError):
        return result
    floor_normal = -box_z_inside_camera
    if float(np.dot(normal, floor_normal)) < 0.0:
        normal = -normal
    tilt = _vector_angle_deg(normal, floor_normal)
    nz = float(np.dot(normal, box_z_inside_camera))
    direction = None
    if abs(nz) > 1e-8:
        gx = -float(np.dot(normal, box_x_camera)) / nz
        gy = -float(np.dot(normal, box_y_camera)) / nz
        direction = float((math.degrees(math.atan2(gy, gx)) + 360.0) % 360.0)
    result.update({
        "available": True,
        "normal_toward_camera": [float(value) for value in normal.tolist()],
        "tilt_deg": float(tilt),
        "gradient_direction_deg_box": direction,
    })
    return result


def analyze_tilt_evidence(
    core_points_camera_mm: np.ndarray,
    core_pixels_xy: np.ndarray,
    center_uv: Tuple[float, float],
    *,
    box_x_camera: Sequence[float],
    box_y_camera: Sequence[float],
    box_z_inside_camera: Sequence[float],
    config: Optional[Mapping[str, Any]] = None,
    raw_plane_normal_toward_camera: Optional[Sequence[float]] = None,
    raw_plane_inlier_ratio: Optional[float] = None,
    raw_plane_residual_p95_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """Return FLAT/TILTED/UNCERTAIN evidence without changing grasp geometry."""
    cfg = dict(config or {})
    sector_count = max(8, _safe_int(cfg.get("sector_count"), 16))
    minimum_sector_points = max(3, _safe_int(cfg.get("minimum_sector_points"), 5))
    local_band = max(1.0, _safe_float(cfg.get("sector_local_band_half_width_mm"), 8.0))
    harmonic_threshold = max(0.5, _safe_float(cfg.get("harmonic_ransac_residual_mm"), 4.0))

    x_axis = _normalize(np.asarray(box_x_camera, dtype=np.float64))
    y_axis = _normalize(np.asarray(box_y_camera, dtype=np.float64))
    z_inside = _normalize(np.asarray(box_z_inside_camera, dtype=np.float64))
    samples = extract_sector_height_samples(
        core_points_camera_mm,
        core_pixels_xy,
        center_uv,
        box_x_camera=x_axis,
        box_y_camera=y_axis,
        box_z_inside_camera=z_inside,
        sector_count=sector_count,
        minimum_sector_points=minimum_sector_points,
        local_band_half_width_mm=local_band,
    )
    harmonic = _fit_first_harmonic(
        samples,
        sector_count=sector_count,
        residual_threshold_mm=harmonic_threshold,
    )
    raw_cue = _raw_plane_cue(
        raw_plane_normal_toward_camera,
        box_x_camera=x_axis,
        box_y_camera=y_axis,
        box_z_inside_camera=z_inside,
        raw_plane_inlier_ratio=raw_plane_inlier_ratio,
        raw_plane_residual_p95_mm=raw_plane_residual_p95_mm,
    )

    minimum_valid_sectors = max(6, _safe_int(cfg.get("minimum_valid_sectors"), 10))
    minimum_harmonic_inliers = max(3, _safe_int(cfg.get("minimum_harmonic_inliers"), 5))
    tilted_raw_min = max(0.0, _safe_float(cfg.get("tilted_raw_plane_min_deg"), 12.0))
    tilted_sector_min = max(0.0, _safe_float(cfg.get("tilted_sector_gradient_min_deg"), 10.0))
    tilted_p2p_min = max(0.0, _safe_float(cfg.get("tilted_peak_to_peak_min_mm"), 18.0))
    flat_sector_max = max(0.0, _safe_float(cfg.get("flat_sector_gradient_max_deg"), 8.0))
    flat_p2p_max = max(0.0, _safe_float(cfg.get("flat_peak_to_peak_max_mm"), 12.0))
    flat_raw_max = max(0.0, _safe_float(cfg.get("flat_raw_plane_max_deg"), 6.0))
    severe_incoherence = max(1.0, _safe_float(cfg.get("severe_incoherence_residual_mm"), 40.0))
    low_raw_override_residual = max(
        1.0,
        _safe_float(cfg.get("low_raw_flat_override_max_residual_mm"), 20.0),
    )

    state = "UNCERTAIN"
    confidence = "low"
    reason = "insufficient_tilt_evidence"
    signals: list[str] = []
    if harmonic is None:
        reason = "sector_harmonic_fit_unavailable"
    else:
        valid_sectors = int(harmonic["valid_sector_count"])
        inlier_sectors = int(harmonic["inlier_sector_count"])
        sector_tilt = float(harmonic["sector_gradient_tilt_deg"])
        peak_to_peak = float(harmonic["predicted_peak_to_peak_mm"])
        harmonic_residual = float(harmonic["residual_median_mm"])
        raw_tilt = raw_cue.get("tilt_deg")
        raw_tilt_value = float(raw_tilt) if raw_tilt is not None else None

        if valid_sectors < minimum_valid_sectors:
            reason = "insufficient_valid_sectors"
            signals.append("sector_support_low")
        else:
            strong_tilt = bool(
                raw_tilt_value is not None
                and raw_tilt_value >= tilted_raw_min
                and sector_tilt >= tilted_sector_min
                and peak_to_peak >= tilted_p2p_min
                and inlier_sectors >= minimum_harmonic_inliers
            )
            if strong_tilt:
                state = "TILTED"
                reason = "raw_plane_and_sector_gradient_consensus"
                signals.extend([
                    "raw_plane_tilt_supported",
                    "sector_gradient_tilt_supported",
                    "peak_to_peak_height_supported",
                ])
                confidence = (
                    "strong"
                    if harmonic_residual <= 6.0 and inlier_sectors >= 8
                    else "moderate"
                )
            elif sector_tilt <= flat_sector_max and peak_to_peak <= flat_p2p_max:
                # A small first harmonic means there is no coherent ring-wide
                # gradient.  Scattered bad sectors may still make the overall
                # residual large; only *severe* incoherence is refused.
                if harmonic_residual > severe_incoherence:
                    state = "UNCERTAIN"
                    reason = "small_gradient_but_sector_surface_severely_incoherent"
                    signals.append("sector_surface_incoherent")
                else:
                    state = "FLAT"
                    reason = "sector_gradient_small"
                    confidence = "strong" if raw_tilt_value is not None and raw_tilt_value < 10.0 else "moderate"
                    if raw_tilt_value is not None and raw_tilt_value >= tilted_raw_min:
                        signals.append("raw_plane_false_tilt_suspected")
            elif raw_tilt_value is not None and raw_tilt_value < flat_raw_max and harmonic_residual <= low_raw_override_residual:
                # This protects the flat baseline from a pathological sector-only
                # fit such as the real M39.2.9 flat replay frame where the raw
                # annulus plane stayed near floor but a few sectors implied a huge
                # first harmonic.
                state = "FLAT"
                reason = "raw_plane_flat_sector_gradient_conflict"
                confidence = "low"
                signals.append("sector_gradient_conflicts_with_raw_plane")
            elif raw_tilt_value is not None and raw_tilt_value >= tilted_raw_min and (
                sector_tilt < tilted_sector_min or peak_to_peak < tilted_p2p_min
            ):
                state = "FLAT"
                reason = "raw_plane_tilt_not_supported_by_ring_gradient"
                confidence = "moderate"
                signals.append("raw_plane_false_tilt_suspected")
            else:
                state = "UNCERTAIN"
                reason = "tilt_cues_do_not_reach_consensus"
                signals.append("tilt_cue_conflict")

    direction_disagreement = None
    if harmonic is not None and raw_cue.get("gradient_direction_deg_box") is not None:
        # Harmonic direction is image-angle based, so it must not be directly
        # compared with box-XY direction.  Leave this field null until M39.3.1
        # provides a calibrated sector-to-box directional model.
        direction_disagreement = None

    return {
        "schema_version": "1.0",
        "stage": "M39.3.0_offline_tilt_evidence",
        "mode": "offline_diagnostic_only",
        "production_routing_enabled": False,
        "state": state,
        "confidence": confidence,
        "classification_reason": reason,
        "signals": signals,
        "core_point_count": int(len(np.asarray(core_points_camera_mm).reshape(-1, 3))),
        "center_uv": [float(center_uv[0]), float(center_uv[1])],
        "raw_plane_cue": raw_cue,
        "sector_gradient": harmonic,
        "direction_crosscheck_deg": direction_disagreement,
        "thresholds": {
            "minimum_valid_sectors": minimum_valid_sectors,
            "minimum_harmonic_inliers": minimum_harmonic_inliers,
            "tilted_raw_plane_min_deg": tilted_raw_min,
            "tilted_sector_gradient_min_deg": tilted_sector_min,
            "tilted_peak_to_peak_min_mm": tilted_p2p_min,
            "flat_sector_gradient_max_deg": flat_sector_max,
            "flat_peak_to_peak_max_mm": flat_p2p_max,
            "flat_raw_plane_max_deg": flat_raw_max,
            "severe_incoherence_residual_mm": severe_incoherence,
            "low_raw_flat_override_max_residual_mm": low_raw_override_residual,
        },
    }
