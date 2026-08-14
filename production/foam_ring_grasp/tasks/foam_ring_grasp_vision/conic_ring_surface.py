"""M39.3.3 conic-constrained target-ring front surface reconstruction.

The visible ``ring_mouth`` is treated as the image of a known 3-D inner circle.
Unlike M39.3.2, the outer silhouette of ``foam_ring`` is never interpreted as
front-face outer rim: a tilted 70 mm cylinder exposes side wall, so its global
silhouette is not the front annulus boundary.

Pipeline (diagnostic-only):
1. fit the observed mouth conic/ellipse and use the mouth contour as semantic
   inner-rim anchor;
2. predict a front-annulus sampling band from the known Rout/Rin ratio;
3. extract local, supported depth plateaus only inside the matched foam-ring
   instance and the predicted annulus;
4. enumerate coherent 3-D plane hypotheses;
5. back-project the observed mouth contour onto every plane.  A valid plane must
   turn the 2-D ellipse back into an approximately circular 3-D rim with the
   known 30 mm nominal radius;
6. use that recovered circle to project the nominal 42.5 mm outer circle and
   verify semantic/depth support on the predicted front annulus.

No absolute box-floor depth is used for surface identity.  The calibrated box
axes are used only to express tilt relative to the horizontal reference.
M39.2.9 remains the production grasp path.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


def _f(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _i(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _norm(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 1e-12:
        raise ValueError("zero vector")
    return a / n


def _deproject(u: float, v: float, z_mm: float, intrinsics: Mapping[str, float]) -> np.ndarray:
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    return np.asarray([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm], dtype=np.float64)


def _project(point: Sequence[float], intrinsics: Mapping[str, float]) -> Optional[Tuple[float, float]]:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)) or float(p[2]) <= 1e-6:
        return None
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    return (float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy))


def _angular_coverage(sectors: Sequence[int], sector_count: int) -> float:
    values = sorted(set(int(s) % int(sector_count) for s in sectors))
    if not values:
        return 0.0
    if len(values) == 1:
        return 360.0 / float(sector_count)
    wrapped = values[1:] + [values[0] + int(sector_count)]
    max_gap = max(b - a for a, b in zip(values, wrapped))
    return float(max(0.0, 360.0 - max_gap * 360.0 / sector_count + 360.0 / sector_count))


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_mouth_conic(mouth_mask: np.ndarray) -> Dict[str, Any]:
    contour = _largest_contour(mouth_mask)
    if contour is None or len(contour) < 5:
        return {"available": False, "reason": "mouth_contour_unavailable"}
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (a, b), angle = ellipse
    major = max(float(a), float(b))
    minor = min(float(a), float(b))
    # Downsample the observed contour for inverse-conic/circle validation.
    pts = contour.reshape(-1, 2).astype(np.float64)
    target = min(128, len(pts))
    indexes = np.linspace(0, len(pts) - 1, target).astype(int)
    sampled = pts[indexes]
    return {
        "available": True,
        "center_uv": [float(cx), float(cy)],
        "ellipse_width_px": float(a),
        "ellipse_height_px": float(b),
        "major_px": major,
        "minor_px": minor,
        "axis_ratio": float(minor / max(major, 1e-6)),
        "angle_deg": float(angle),
        "contour_point_count": int(len(pts)),
        "sampled_contour_uv": [[float(p[0]), float(p[1])] for p in sampled],
    }


def _ray_mouth_exit(mask: np.ndarray, center_uv: Tuple[float, float], angle: float, maximum_radius: int) -> Optional[int]:
    mouth = np.asarray(mask, dtype=bool)
    h, w = mouth.shape
    cx, cy = float(center_uv[0]), float(center_uv[1])
    ca, sa = math.cos(angle), math.sin(angle)
    values = []
    for radius in range(1, max(2, int(maximum_radius)) + 1):
        x = int(round(cx + ca * radius))
        y = int(round(cy + sa * radius))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
        if mouth[y, x]:
            values.append(radius)
    return max(values) if values else None


def _cluster_plateaus(
    samples: list[tuple[float, float, float, int, float]],
    *,
    depth_gap_mm: float,
    minimum_points: int,
    minimum_rays: int,
    minimum_radial_span_ratio: float,
    maximum_candidates: int,
) -> list[Dict[str, Any]]:
    if len(samples) < minimum_points:
        return []
    ordered = sorted(samples, key=lambda row: row[0])
    groups: list[list[tuple[float, float, float, int, float]]] = [[ordered[0]]]
    for row in ordered[1:]:
        if float(row[0]) - float(groups[-1][-1][0]) <= depth_gap_mm:
            groups[-1].append(row)
        else:
            groups.append([row])
    out: list[Dict[str, Any]] = []
    for group in groups:
        if len(group) < minimum_points:
            continue
        z = np.asarray([row[0] for row in group], dtype=np.float64)
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med)))
        robust_band = min(10.0, max(2.5, 3.0 * 1.4826 * mad))
        supported = [row for row in group if abs(float(row[0]) - med) <= robust_band]
        if len(supported) < minimum_points:
            continue
        ray_support = len(set(int(row[3]) for row in supported))
        radial = [float(row[4]) for row in supported]
        span = max(radial) - min(radial) if radial else 0.0
        if ray_support < minimum_rays or span < minimum_radial_span_ratio:
            continue
        z2 = np.asarray([float(row[0]) for row in supported], dtype=np.float64)
        out.append({
            "depth_mm": float(np.median(z2)),
            "depth_mad_mm": float(np.median(np.abs(z2 - np.median(z2)))),
            "support_count": int(len(supported)),
            "ray_support_count": int(ray_support),
            "radial_span_ratio": float(span),
            "representative_uv": [
                float(np.median([row[1] for row in supported])),
                float(np.median([row[2] for row in supported])),
            ],
        })
    out.sort(key=lambda row: float(row["depth_mm"]))
    return out[: max(1, int(maximum_candidates))]


def extract_conic_annulus_sector_candidates(
    depth_mm: np.ndarray,
    ring_mask: np.ndarray,
    mouth_mask: np.ndarray,
    center_uv: Tuple[float, float],
    intrinsics: Mapping[str, float],
    *,
    object_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Sample depth inside a front annulus predicted from the known circle ratio.

    The global foam-ring outer silhouette is intentionally ignored as a radial
    endpoint.  It is used only as an ownership/visibility mask.
    """
    depth = np.asarray(depth_mm)
    ring = np.asarray(ring_mask, dtype=bool)
    mouth = np.asarray(mouth_mask, dtype=bool)
    if depth.shape != ring.shape or ring.shape != mouth.shape:
        raise ValueError("depth/ring/mouth shapes must match")

    conic = fit_mouth_conic(mouth)
    if conic.get("available"):
        fitted_center = conic.get("center_uv") or center_uv
        center = (float(fitted_center[0]), float(fitted_center[1]))
    else:
        center = (float(center_uv[0]), float(center_uv[1]))

    rin = 0.5 * _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0)
    rout = 0.5 * _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    ratio = rout / max(rin, 1e-6)
    sectors = max(8, _i(config.get("sector_count"), 16))
    wedge_half = max(1.0, _f(config.get("sector_wedge_half_angle_deg"), 8.0))
    wedge_rays = max(3, _i(config.get("sector_wedge_ray_count"), 9))
    max_radius = max(32, _i(config.get("maximum_radial_search_px"), 120))
    guard = float(np.clip(_f(config.get("inner_edge_guard_wall_ratio"), 0.15), 0.0, 0.5))
    sample_end = float(np.clip(_f(config.get("predicted_annulus_sample_end_ratio"), 0.78), guard + 0.08, 0.95))
    min_depth = _f(config.get("minimum_depth_mm"), 150.0)
    max_depth = _f(config.get("maximum_depth_mm"), 3000.0)
    gap_mm = max(1.0, _f(config.get("plateau_depth_gap_mm"), 4.0))
    min_points = max(4, _i(config.get("plateau_minimum_points"), 6))
    min_rays = max(1, _i(config.get("plateau_minimum_rays"), 2))
    min_span = max(0.02, _f(config.get("plateau_minimum_radial_span_ratio"), 0.10))
    max_candidates = max(1, _i(config.get("maximum_candidates_per_sector"), 3))
    fx_eff = math.sqrt(float(intrinsics["fx"]) * float(intrinsics["fy"]))

    rows: list[Dict[str, Any]] = []
    for sector in range(sectors):
        theta = (sector + 0.5) * 2.0 * math.pi / sectors
        radial_samples: list[tuple[float, float, float, int, float]] = []
        inner_radii, predicted_outer_radii = [], []
        for ray_index, delta_deg in enumerate(np.linspace(-wedge_half, wedge_half, wedge_rays)):
            angle = theta + math.radians(float(delta_deg))
            inner = _ray_mouth_exit(mouth, center, angle, max_radius)
            if inner is None or inner < 2:
                continue
            outer = float(inner) * ratio
            wall = max(1.0, outer - float(inner))
            inner_radii.append(float(inner))
            predicted_outer_radii.append(float(outer))
            ca, sa = math.cos(angle), math.sin(angle)
            sample_count = max(8, int(math.ceil(wall * 2.0)))
            for t in np.linspace(guard, sample_end, sample_count):
                radius = float(inner) + wall * float(t)
                x = int(round(center[0] + ca * radius))
                y = int(round(center[1] + sa * radius))
                if x < 0 or y < 0 or x >= depth.shape[1] or y >= depth.shape[0]:
                    continue
                # Foam-ring mask is semantic ownership only.
                if not ring[y, x]:
                    continue
                z = float(depth[y, x])
                if not np.isfinite(z) or z < min_depth or z > max_depth:
                    continue
                radial_samples.append((z, float(x), float(y), int(ray_index), float(t)))

        row: Dict[str, Any] = {
            "sector": int(sector),
            "sector_angle_deg_image": float((sector + 0.5) * 360.0 / sectors),
            "status": "MISSING",
            "raw_sample_count": int(len(radial_samples)),
            "candidates": [],
        }
        if inner_radii:
            inner_px = float(np.median(inner_radii))
            outer_px = float(np.median(predicted_outer_radii))
            row.update({
                "observed_inner_radius_px": inner_px,
                "predicted_front_outer_radius_px": outer_px,
                "predicted_front_wall_width_px": outer_px - inner_px,
                "outer_rim_source": "known_radius_ratio_from_ring_mouth",
            })
        candidates = _cluster_plateaus(
            radial_samples,
            depth_gap_mm=gap_mm,
            minimum_points=min_points,
            minimum_rays=min_rays,
            minimum_radial_span_ratio=min_span,
            maximum_candidates=max_candidates,
        )
        inner_px = row.get("observed_inner_radius_px")
        for rank, candidate in enumerate(candidates):
            z = float(candidate["depth_mm"])
            inner_est = float(inner_px) * z / fx_eff if inner_px is not None and fx_eff > 1e-9 else None
            scale_error = abs(inner_est - rin) / max(rin, 1e-6) if inner_est is not None else None
            candidate.update({
                "front_rank": int(rank),
                "inner_radius_estimate_mm": inner_est,
                "outer_radius_estimate_mm": (inner_est * ratio if inner_est is not None else None),
                "nominal_scale_error": scale_error,
            })
        row["candidates"] = candidates
        if candidates:
            row["status"] = "CANDIDATES_AVAILABLE"
        rows.append(row)
    conic["nominal_outer_to_inner_radius_ratio"] = float(ratio)
    conic["sampling_center_uv"] = [float(center[0]), float(center[1])]
    return rows, conic


def _candidate_records(
    sector_rows: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, float],
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z: np.ndarray,
    *,
    maximum_nominal_scale_error: float,
) -> list[Dict[str, Any]]:
    records = []
    for row in sector_rows:
        sector = int(row["sector"])
        for index, raw in enumerate(row.get("candidates") or []):
            candidate = dict(raw)
            err = candidate.get("nominal_scale_error")
            if err is not None and float(err) > maximum_nominal_scale_error:
                continue
            uv = candidate.get("representative_uv") or []
            if len(uv) != 2:
                continue
            point = _deproject(float(uv[0]), float(uv[1]), float(candidate["depth_mm"]), intrinsics)
            records.append({
                "sector": sector,
                "candidate_index": int(index),
                "candidate": candidate,
                "point_camera_mm": point,
                "x_box_mm": float(np.dot(point, box_x)),
                "y_box_mm": float(np.dot(point, box_y)),
                "h_box_mm": float(np.dot(point, box_z)),
            })
    return records


def _fit_circle_2d(points: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    if points.shape[0] < 6:
        return None
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        d, e, f = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    center = np.asarray([-0.5 * d, -0.5 * e], dtype=np.float64)
    r2 = float(center @ center - f)
    if not np.isfinite(r2) or r2 <= 1e-6:
        return None
    radius = math.sqrt(r2)
    residuals = np.abs(np.linalg.norm(points - center[None, :], axis=1) - radius)
    return center, float(radius), residuals


def _plane_conic_validation(
    coefficients: Sequence[float],
    conic: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    ring_mask: np.ndarray,
    *,
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z: np.ndarray,
    inner_radius_nominal: float,
    outer_radius_nominal: float,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    contour_uv = np.asarray(conic.get("sampled_contour_uv") or [], dtype=np.float64)
    if contour_uv.ndim != 2 or contour_uv.shape[0] < 8:
        return None
    c = np.asarray(coefficients, dtype=np.float64)
    gx, gy, h0 = float(c[0]), float(c[1]), float(c[2])
    n_raw = box_z - gx * box_x - gy * box_y
    n_len = float(np.linalg.norm(n_raw))
    if n_len <= 1e-9:
        return None
    n = n_raw / n_len

    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    points = []
    for u, v in contour_uv:
        ray = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        denom = float(np.dot(n_raw, ray))
        if abs(denom) < 1e-8:
            continue
        z = h0 / denom
        if not np.isfinite(z) or z <= 100.0 or z >= 3000.0:
            continue
        points.append(ray * z)
    if len(points) < 8:
        return None
    pts = np.asarray(points, dtype=np.float64)
    origin = np.mean(pts, axis=0)
    e1 = box_x - float(np.dot(box_x, n)) * n
    if float(np.linalg.norm(e1)) < 1e-6:
        e1 = np.cross(n, np.asarray([0.0, 0.0, 1.0]))
    e1 = _norm(e1)
    e2 = _norm(np.cross(n, e1))
    xy = np.column_stack([(pts - origin) @ e1, (pts - origin) @ e2])
    fit = _fit_circle_2d(xy)
    if fit is None:
        return None
    center2, radius, residuals = fit
    center3 = origin + center2[0] * e1 + center2[1] * e2
    radius_error = abs(radius - inner_radius_nominal) / max(inner_radius_nominal, 1e-6)

    # Reproject ideal fitted circle and compare it to the observed mouth contour.
    phis = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    projected_inner = []
    projected_outer = []
    semantic_mid_support = []
    ring = np.asarray(ring_mask, dtype=bool)
    h, w = ring.shape
    mid_radius = inner_radius_nominal + 0.50 * (outer_radius_nominal - inner_radius_nominal)
    for phi in phis:
        direction = math.cos(phi) * e1 + math.sin(phi) * e2
        for radius_used, bucket in ((inner_radius_nominal, projected_inner), (outer_radius_nominal, projected_outer)):
            uv = _project(center3 + radius_used * direction, intrinsics)
            if uv is not None:
                bucket.append(uv)
        uv_mid = _project(center3 + mid_radius * direction, intrinsics)
        if uv_mid is not None:
            x = int(round(uv_mid[0])); y = int(round(uv_mid[1]))
            semantic_mid_support.append(bool(0 <= x < w and 0 <= y < h and ring[y, x]))
    if len(projected_inner) < 16:
        return None
    pred = np.asarray(projected_inner, dtype=np.float64)
    obs = contour_uv
    # Symmetric chamfer, small arrays so pairwise computation is cheap.
    dist = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
    d_pred = np.min(dist, axis=1)
    d_obs = np.min(dist, axis=0)
    chamfer = np.concatenate([d_pred, d_obs])
    semantic_ratio = float(np.mean(semantic_mid_support)) if semantic_mid_support else 0.0

    # Depth must support the *whole predicted front annulus*, not only the few
    # sector representatives that seeded the candidate plane. This is the key
    # M39.3.3 disambiguation for a single observed circle: conic geometry alone
    # admits ambiguous plane poses, while dense annulus depth should agree with
    # only the physically visible front surface.
    annulus_sector_count = max(8, _i(config.get("annulus_depth_sector_count"), 16))
    annulus_fractions = config.get("annulus_depth_radial_fractions") or [0.25, 0.50, 0.75]
    try:
        annulus_fractions = [float(v) for v in annulus_fractions]
    except Exception:
        annulus_fractions = [0.25, 0.50, 0.75]
    depth_inlier_mm = max(1.0, _f(config.get("annulus_depth_inlier_mm"), 6.0))
    patch_radius = max(0, _i(config.get("annulus_depth_patch_radius_px"), 1))
    depth_array = np.asarray(config.get("_depth_array")) if config.get("_depth_array") is not None else None
    depth_sector_rows = []
    all_depth_residuals = []
    supported_depth_sectors = []
    if isinstance(depth_array, np.ndarray) and depth_array.ndim == 2:
        plane_d = h0 / n_len
        for sector in range(annulus_sector_count):
            phi = (sector + 0.5) * 2.0 * math.pi / annulus_sector_count
            direction = math.cos(phi) * e1 + math.sin(phi) * e2
            sample_residuals = []
            valid_sample_count = 0
            for frac in annulus_fractions:
                rr = inner_radius_nominal + float(frac) * (outer_radius_nominal - inner_radius_nominal)
                uv = _project(center3 + rr * direction, intrinsics)
                if uv is None:
                    continue
                x0, y0 = int(round(uv[0])), int(round(uv[1]))
                local = []
                for yy in range(y0 - patch_radius, y0 + patch_radius + 1):
                    for xx in range(x0 - patch_radius, x0 + patch_radius + 1):
                        if yy < 0 or xx < 0 or yy >= h or xx >= w or not ring[yy, xx]:
                            continue
                        zz = float(depth_array[yy, xx])
                        if not np.isfinite(zz) or zz < 150.0 or zz > 3000.0:
                            continue
                        pp = _deproject(float(xx), float(yy), zz, intrinsics)
                        local.append(abs(float(np.dot(n, pp)) - plane_d))
                if local:
                    valid_sample_count += 1
                    sample_residuals.append(float(np.median(local)))
            inlier_count = sum(r <= depth_inlier_mm for r in sample_residuals)
            supported = bool(valid_sample_count >= 2 and inlier_count >= max(2, int(math.ceil(0.60 * valid_sample_count))))
            if supported:
                supported_depth_sectors.append(sector)
            all_depth_residuals.extend(sample_residuals)
            depth_sector_rows.append({
                "sector": int(sector),
                "valid_sample_count": int(valid_sample_count),
                "inlier_sample_count": int(inlier_count),
                "supported": supported,
                "residual_median_mm": float(np.median(sample_residuals)) if sample_residuals else None,
            })
    depth_valid_sectors = sum(int(r["valid_sample_count"] > 0) for r in depth_sector_rows)
    depth_support_ratio = float(len(supported_depth_sectors) / max(depth_valid_sectors, 1)) if depth_sector_rows else None
    depth_coverage = _angular_coverage(supported_depth_sectors, annulus_sector_count) if supported_depth_sectors else 0.0
    depth_residual_median = float(np.median(all_depth_residuals)) if all_depth_residuals else None
    depth_residual_p90 = float(np.percentile(all_depth_residuals, 90)) if all_depth_residuals else None
    return {
        "inner_radius_mm": float(radius),
        "inner_radius_error_ratio": float(radius_error),
        "circle_radial_residual_median_mm": float(np.median(residuals)),
        "circle_radial_residual_p90_mm": float(np.percentile(residuals, 90)),
        "reprojection_chamfer_median_px": float(np.median(chamfer)),
        "reprojection_chamfer_p90_px": float(np.percentile(chamfer, 90)),
        "predicted_mid_annulus_semantic_support_ratio": semantic_ratio,
        "predicted_annulus_depth_support": {
            "available": bool(depth_sector_rows),
            "supported_sector_count": int(len(supported_depth_sectors)),
            "valid_sector_count": int(depth_valid_sectors),
            "support_ratio": depth_support_ratio,
            "angular_coverage_deg": float(depth_coverage),
            "residual_median_mm": depth_residual_median,
            "residual_p90_mm": depth_residual_p90,
            "sector_samples": depth_sector_rows,
        },
        "circle_center_camera_mm": [float(v) for v in center3.tolist()],
        "plane_basis_u_camera": [float(v) for v in e1.tolist()],
        "plane_basis_v_camera": [float(v) for v in e2.tolist()],
        "predicted_inner_rim_uv": [[float(u), float(v)] for u, v in projected_inner],
        "predicted_outer_rim_uv": [[float(u), float(v)] for u, v in projected_outer],
    }


def _fit_conic_constrained_plane(
    sector_rows: Sequence[Mapping[str, Any]],
    conic: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    ring_mask: np.ndarray,
    *,
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z: np.ndarray,
    object_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    sector_count = max(8, _i(config.get("sector_count"), 16))
    residual_threshold = max(1.0, _f(config.get("surface_plane_residual_mm"), 5.0))
    minimum_sectors = max(4, _i(config.get("minimum_surface_sectors"), 5))
    minimum_coverage = max(45.0, _f(config.get("minimum_surface_coverage_deg"), 135.0))
    max_tilt = max(5.0, _f(config.get("maximum_reconstruction_tilt_deg"), 50.0))
    max_scale_error = max(0.05, _f(config.get("maximum_sector_inner_radius_error"), 0.55))
    front_window = max(1.0, _f(config.get("front_surface_depth_window_mm"), 22.0))
    rin = 0.5 * _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0)
    rout = 0.5 * _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    max_circle_radius_error = _f(config.get("maximum_conic_inner_radius_error_ratio"), 0.35)
    max_circle_p90 = _f(config.get("maximum_conic_circle_residual_p90_mm"), 5.0)
    max_chamfer_p90 = _f(config.get("maximum_conic_reprojection_p90_px"), 5.0)
    min_semantic_support = _f(config.get("minimum_predicted_annulus_semantic_support_ratio"), 0.35)
    min_depth_support_sectors = max(0, _i(config.get("minimum_predicted_annulus_depth_support_sectors"), 4))
    min_depth_support_ratio = _f(config.get("minimum_predicted_annulus_depth_support_ratio"), 0.25)

    records = _candidate_records(
        sector_rows, intrinsics, box_x, box_y, box_z,
        maximum_nominal_scale_error=max_scale_error,
    )
    if len(records) < minimum_sectors:
        return None
    by_sector: Dict[int, list[Dict[str, Any]]] = {}
    for record in records:
        by_sector.setdefault(int(record["sector"]), []).append(record)

    models = []
    quick_conic_config = dict(config)
    quick_conic_config.pop("_depth_array", None)
    max_seed_models = max(500, _i(config.get("maximum_seed_models"), 4000))
    seen = 0
    for indexes in combinations(range(len(records)), 3):
        if seen >= max_seed_models:
            break
        seed = [records[i] for i in indexes]
        if len({int(row["sector"]) for row in seed}) < 3:
            continue
        matrix = np.asarray([[r["x_box_mm"], r["y_box_mm"], 1.0] for r in seed], dtype=np.float64)
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            continue
        heights = np.asarray([r["h_box_mm"] for r in seed], dtype=np.float64)
        coeff = np.linalg.solve(matrix, heights)
        tilt = math.degrees(math.atan(math.hypot(float(coeff[0]), float(coeff[1]))))
        if tilt > max_tilt:
            continue
        seen += 1

        chosen = []
        for sector, candidates in by_sector.items():
            best = None
            for record in candidates:
                pred = float(coeff[0]) * float(record["x_box_mm"]) + float(coeff[1]) * float(record["y_box_mm"]) + float(coeff[2])
                residual = abs(float(record["h_box_mm"]) - pred)
                if best is None or residual < best[0]:
                    best = (residual, record)
            if best is not None and best[0] <= residual_threshold:
                chosen.append(best)
        sectors = [int(x[1]["sector"]) for x in chosen]
        if len(chosen) < minimum_sectors or _angular_coverage(sectors, sector_count) < minimum_coverage:
            continue
        A = np.asarray([[x[1]["x_box_mm"], x[1]["y_box_mm"], 1.0] for x in chosen], dtype=np.float64)
        hh = np.asarray([x[1]["h_box_mm"] for x in chosen], dtype=np.float64)
        coeff = np.linalg.lstsq(A, hh, rcond=None)[0]
        residuals = np.abs(hh - A @ coeff)
        keep = residuals <= residual_threshold
        if int(np.count_nonzero(keep)) >= minimum_sectors:
            chosen = [x for x, flag in zip(chosen, keep.tolist()) if flag]
            A = A[keep]; hh = hh[keep]
            coeff = np.linalg.lstsq(A, hh, rcond=None)[0]
            residuals = np.abs(hh - A @ coeff)
        sectors = [int(x[1]["sector"]) for x in chosen]
        coverage = _angular_coverage(sectors, sector_count)
        if len(chosen) < minimum_sectors or coverage < minimum_coverage:
            continue
        tilt = math.degrees(math.atan(math.hypot(float(coeff[0]), float(coeff[1]))))
        if tilt > max_tilt:
            continue
        cv = _plane_conic_validation(
            coeff, conic, intrinsics, ring_mask,
            box_x=box_x, box_y=box_y, box_z=box_z,
            inner_radius_nominal=rin, outer_radius_nominal=rout, config=quick_conic_config,
        )
        if cv is None:
            continue
        if float(cv["inner_radius_error_ratio"]) > max_circle_radius_error:
            continue
        if float(cv["circle_radial_residual_p90_mm"]) > max_circle_p90:
            continue
        if float(cv["reprojection_chamfer_p90_px"]) > max_chamfer_p90:
            continue
        if float(cv["predicted_mid_annulus_semantic_support_ratio"]) < min_semantic_support:
            continue
        depth_support = cv.get("predicted_annulus_depth_support") or {}
        if depth_support.get("available"):
            if int(depth_support.get("supported_sector_count") or 0) < min_depth_support_sectors:
                continue
            if float(depth_support.get("support_ratio") or 0.0) < min_depth_support_ratio:
                continue
        depths = [float(x[1]["candidate"]["depth_mm"]) for x in chosen]
        support = sum(int(x[1]["candidate"].get("support_count") or 0) for x in chosen)
        scale_errors = [float(x[1]["candidate"]["nominal_scale_error"]) for x in chosen if x[1]["candidate"].get("nominal_scale_error") is not None]
        conic_score = (
            3.0 * float(cv["inner_radius_error_ratio"])
            + float(cv["circle_radial_residual_p90_mm"]) / max(rin, 1.0)
            + float(cv["reprojection_chamfer_p90_px"]) / 20.0
            + 0.5 * (1.0 - float(cv["predicted_mid_annulus_semantic_support_ratio"]))
            + (
                1.0 - float((cv.get("predicted_annulus_depth_support") or {}).get("support_ratio") or 0.0)
                if (cv.get("predicted_annulus_depth_support") or {}).get("available") else 0.5
            )
            + float((cv.get("predicted_annulus_depth_support") or {}).get("residual_median_mm") or 0.0) / 20.0
        )
        models.append({
            "coefficients": coeff,
            "chosen": chosen,
            "sectors": sorted(sectors),
            "sector_count": int(len(chosen)),
            "angular_coverage_deg": float(coverage),
            "residual_median_mm": float(np.median(residuals)),
            "residual_p90_mm": float(np.percentile(residuals, 90)),
            "median_depth_mm": float(np.median(depths)),
            "support_count": int(support),
            "median_sector_scale_error": float(np.median(scale_errors)) if scale_errors else None,
            "tilt_deg": float(tilt),
            "conic_validation": cv,
            "conic_score": float(conic_score),
        })
    if not models:
        return None

    # Prefer the camera-nearest family. Run the expensive dense annulus-depth
    # disambiguation only on a small shortlist, not on every 3-point seed.
    front_depth = min(float(m["median_depth_mm"]) for m in models)
    front_models = [m for m in models if float(m["median_depth_mm"]) <= front_depth + front_window]
    front_models.sort(key=lambda m: (
        float(m["conic_score"]),
        -int(m["sector_count"]),
        -float(m["angular_coverage_deg"]),
        float(m["residual_median_mm"]),
        -int(m["support_count"]),
    ))
    shortlist_count = max(4, _i(config.get("dense_annulus_shortlist_models"), 16))
    verified = []
    for model in front_models[:shortlist_count]:
        full_cv = _plane_conic_validation(
            model["coefficients"], conic, intrinsics, ring_mask,
            box_x=box_x, box_y=box_y, box_z=box_z,
            inner_radius_nominal=rin, outer_radius_nominal=rout, config=config,
        )
        if full_cv is None:
            continue
        depth_support = full_cv.get("predicted_annulus_depth_support") or {}
        if depth_support.get("available"):
            if int(depth_support.get("supported_sector_count") or 0) < min_depth_support_sectors:
                continue
            if float(depth_support.get("support_ratio") or 0.0) < min_depth_support_ratio:
                continue
        model = dict(model)
        model["conic_validation"] = full_cv
        model["conic_score"] = (
            3.0 * float(full_cv["inner_radius_error_ratio"])
            + float(full_cv["circle_radial_residual_p90_mm"]) / max(rin, 1.0)
            + float(full_cv["reprojection_chamfer_p90_px"]) / 20.0
            + 0.5 * (1.0 - float(full_cv["predicted_mid_annulus_semantic_support_ratio"]))
            + (1.0 - float(depth_support.get("support_ratio") or 0.0) if depth_support.get("available") else 0.5)
            + float(depth_support.get("residual_median_mm") or 0.0) / 20.0
        )
        verified.append(model)
    if not verified:
        return None
    best = min(verified, key=lambda m: (
        float(m["conic_score"]),
        -int(m["sector_count"]),
        -float(m["angular_coverage_deg"]),
        float(m["residual_median_mm"]),
        -int(m["support_count"]),
    ))
    coeff = np.asarray(best["coefficients"], dtype=np.float64)
    gx, gy = float(coeff[0]), float(coeff[1])
    normal_into_box = _norm(box_z - gx * box_x - gy * box_y)
    normal_toward_camera = -normal_into_box
    direction_box = float((math.degrees(math.atan2(gy, gx)) + 360.0) % 360.0)
    predicted_p2p = float(2.0 * rout * math.tan(math.radians(float(best["tilt_deg"]))))

    selected_map = {int(x[1]["sector"]): x[1] for x in best["chosen"]}
    samples = []
    for row in sector_rows:
        item = dict(row)
        selected = selected_map.get(int(row["sector"]))
        if selected is not None:
            item["status"] = "SELECTED_CONIC_FRONT_SURFACE"
            item["selected_candidate"] = dict(selected["candidate"])
            item["point_camera_mm"] = [float(v) for v in selected["point_camera_mm"].tolist()]
            item["x_box_mm"] = float(selected["x_box_mm"])
            item["y_box_mm"] = float(selected["y_box_mm"])
            item["h_box_mm"] = float(selected["h_box_mm"])
        elif row.get("candidates"):
            item["status"] = "REJECTED_NON_CONIC_OR_INCOHERENT"
        samples.append(item)

    return {
        "model": "h_box=h0+gx*x_box+gy*y_box + inverse_conic_circle_constraint",
        "h0_mm": float(coeff[2]),
        "gradient_x": gx,
        "gradient_y": gy,
        "tilt_deg": float(best["tilt_deg"]),
        "tilt_direction_deg_box": direction_box,
        "normal_toward_camera": [float(v) for v in normal_toward_camera.tolist()],
        "predicted_outer_peak_to_peak_mm": predicted_p2p,
        "selected_sector_count": int(best["sector_count"]),
        "selected_sectors": list(best["sectors"]),
        "angular_coverage_deg": float(best["angular_coverage_deg"]),
        "residual_median_mm": float(best["residual_median_mm"]),
        "residual_p90_mm": float(best["residual_p90_mm"]),
        "median_depth_mm": float(best["median_depth_mm"]),
        "support_count": int(best["support_count"]),
        "frontmost_model_depth_mm": float(front_depth),
        "conic_score": float(best["conic_score"]),
        "conic_validation": best["conic_validation"],
        "sector_samples": samples,
    }


def _jackknife(surface: Mapping[str, Any]) -> Dict[str, Any]:
    samples = [r for r in surface.get("sector_samples") or [] if r.get("status") == "SELECTED_CONIC_FRONT_SURFACE"]
    if len(samples) < 6:
        return {"available": False, "tilt_std_deg": None, "direction_resultant": None}
    tilts, dirs = [], []
    for omitted in range(len(samples)):
        subset = [r for i, r in enumerate(samples) if i != omitted]
        A = np.asarray([[float(r["x_box_mm"]), float(r["y_box_mm"]), 1.0] for r in subset], dtype=np.float64)
        h = np.asarray([float(r["h_box_mm"]) for r in subset], dtype=np.float64)
        if len(subset) < 3:
            continue
        c = np.linalg.lstsq(A, h, rcond=None)[0]
        gx, gy = float(c[0]), float(c[1])
        tilts.append(math.degrees(math.atan(math.hypot(gx, gy))))
        dirs.append(math.atan2(gy, gx))
    if not tilts:
        return {"available": False, "tilt_std_deg": None, "direction_resultant": None}
    sx = float(np.mean(np.cos(dirs))); sy = float(np.mean(np.sin(dirs)))
    return {
        "available": True,
        "tilt_std_deg": float(np.std(np.asarray(tilts))),
        "tilt_min_deg": float(np.min(tilts)),
        "tilt_max_deg": float(np.max(tilts)),
        "direction_resultant": float(math.hypot(sx, sy)),
    }


def reconstruct_conic_ring_surface(
    depth_mm: np.ndarray,
    ring_mask: np.ndarray,
    mouth_mask: np.ndarray,
    center_uv: Tuple[float, float],
    intrinsics: Mapping[str, float],
    *,
    box_x_camera: Sequence[float],
    box_y_camera: Sequence[float],
    box_z_inside_camera: Sequence[float],
    object_geometry: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    prior_tilt_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    x_axis, y_axis, z_axis = _norm(box_x_camera), _norm(box_y_camera), _norm(box_z_inside_camera)
    sectors, conic = extract_conic_annulus_sector_candidates(
        depth_mm, ring_mask, mouth_mask, center_uv, intrinsics,
        object_geometry=object_geometry, config=cfg,
    )
    fit_cfg = dict(cfg)
    fit_cfg["_depth_array"] = np.asarray(depth_mm)
    surface = _fit_conic_constrained_plane(
        sectors, conic, intrinsics, ring_mask,
        box_x=x_axis, box_y=y_axis, box_z=z_axis,
        object_geometry=object_geometry, config=fit_cfg,
    )
    base = {
        "schema_version": "1.0",
        "stage": "M39.3.3_conic_constrained_ring_surface_reconstruction",
        "mode": "diagnostic_only",
        "production_routing_enabled": False,
        "semantic_anchor": "matched_ring_mouth_plus_foam_ring_instance",
        "front_outer_rim_source": "known_outer_to_inner_radius_ratio_projected_from_mouth",
        "foam_ring_outer_silhouette_used_as_front_outer_rim": False,
        "uses_absolute_box_floor_depth_for_identity": False,
        "mouth_conic": conic,
        "known_object_geometry": {
            "nominal_inner_diameter_mm": _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0),
            "nominal_outer_diameter_mm": _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0),
            "nominal_wall_thickness_mm": _f(object_geometry.get("nominal_wall_thickness_mm"), 14.0),
            "axial_length_mm": _f(object_geometry.get("axial_length_mm"), 70.0),
        },
        "candidate_sector_count": int(sum(bool(r.get("candidates")) for r in sectors)),
    }
    if surface is None:
        return {**base, "status": "UNCERTAIN", "classification": "UNCERTAIN", "reason": "no_conic_validated_front_surface", "surface": None, "sector_samples": sectors}

    jack = _jackknife(surface)
    cv = surface.get("conic_validation") or {}
    tilt = float(surface["tilt_deg"])
    selected = int(surface["selected_sector_count"])
    coverage = float(surface["angular_coverage_deg"])
    residual = float(surface["residual_median_mm"])
    p2p = float(surface["predicted_outer_peak_to_peak_mm"])
    min_sec = max(5, _i(cfg.get("classification_minimum_sectors"), 6))
    min_cov = _f(cfg.get("classification_minimum_coverage_deg"), 157.5)
    max_res = _f(cfg.get("classification_maximum_residual_mm"), 4.5)
    max_jack = _f(cfg.get("classification_maximum_jackknife_tilt_std_deg"), 5.0)
    max_radius_err = _f(cfg.get("classification_maximum_conic_radius_error_ratio"), 0.24)
    max_circle_p90 = _f(cfg.get("classification_maximum_circle_residual_p90_mm"), 4.0)
    max_chamfer = _f(cfg.get("classification_maximum_reprojection_p90_px"), 4.5)
    min_semantic = _f(cfg.get("classification_minimum_semantic_support_ratio"), 0.45)
    min_depth_sectors = max(0, _i(cfg.get("classification_minimum_annulus_depth_support_sectors"), 6))
    min_depth_ratio = _f(cfg.get("classification_minimum_annulus_depth_support_ratio"), 0.40)
    max_depth_residual = _f(cfg.get("classification_maximum_annulus_depth_residual_median_mm"), 6.0)
    flat_min_depth_sectors = max(0, _i(cfg.get("flat_minimum_annulus_depth_support_sectors"), 5))
    flat_min_depth_ratio = _f(cfg.get("flat_minimum_annulus_depth_support_ratio"), 0.30)
    flat_max_depth_residual = _f(cfg.get("flat_maximum_annulus_depth_residual_median_mm"), 6.0)
    flat_max_radius_error = _f(cfg.get("flat_maximum_conic_radius_error_ratio"), 0.10)
    depth_support = cv.get("predicted_annulus_depth_support") or {}
    flat_max = _f(cfg.get("flat_tilt_max_deg"), 8.0)
    tilt_min = _f(cfg.get("tilted_tilt_min_deg"), 10.0)
    require_independent = bool(cfg.get("require_m39_3_1_tilt_consensus", True))
    prior = dict(prior_tilt_evidence or {})
    prior_state = str(prior.get("state") or prior.get("classification") or "UNAVAILABLE").upper()
    independent_tilt_ok = (not require_independent) or prior_state == "TILTED"
    rout = 0.5 * _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    physical_p2p = _f(cfg.get("maximum_physical_peak_to_peak_mm"), 2.0 * rout * math.tan(math.radians(_f(cfg.get("maximum_reconstruction_tilt_deg"), 50.0))) + 6.0)
    base_stable = bool(
        selected >= min_sec and coverage >= min_cov and residual <= max_res and p2p <= physical_p2p
        and float(cv.get("inner_radius_error_ratio", 99.0)) <= max_radius_err
        and float(cv.get("circle_radial_residual_p90_mm", 99.0)) <= max_circle_p90
        and float(cv.get("reprojection_chamfer_p90_px", 99.0)) <= max_chamfer
        and float(cv.get("predicted_mid_annulus_semantic_support_ratio", 0.0)) >= min_semantic
        and (not jack.get("available") or float(jack.get("tilt_std_deg") or 0.0) <= max_jack)
    )
    dense_tilt_support = bool(
        not depth_support.get("available")
        or (
            int(depth_support.get("supported_sector_count") or 0) >= min_depth_sectors
            and float(depth_support.get("support_ratio") or 0.0) >= min_depth_ratio
            and float(depth_support.get("residual_median_mm") or 99.0) <= max_depth_residual
        )
    )
    flat_dense_support = bool(
        not depth_support.get("available")
        or (
            int(depth_support.get("supported_sector_count") or 0) >= flat_min_depth_sectors
            and float(depth_support.get("support_ratio") or 0.0) >= flat_min_depth_ratio
            and float(depth_support.get("residual_median_mm") or 99.0) <= flat_max_depth_residual
        )
    )
    # Both states require target-specific depth confirmation. This prevents a
    # tilted target from becoming FLAT simply because a sparse/wrong local patch
    # happened to fit a near-horizontal plane.
    if not base_stable:
        cls, reason = "UNCERTAIN", "conic_surface_not_stable_enough_for_tilt_classification"
    elif tilt <= flat_max and flat_dense_support and float(cv.get("inner_radius_error_ratio", 99.0)) <= flat_max_radius_error:
        cls, reason = "FLAT", "conic_plus_annulus_depth_confirmed_flat"
    elif tilt <= flat_max:
        cls, reason = "UNCERTAIN", "flat_plane_lacks_target_annulus_depth_consensus"
    elif tilt >= tilt_min and dense_tilt_support and independent_tilt_ok:
        cls, reason = "TILTED", "conic_depth_and_independent_ring_gradient_confirm_tilt"
    elif tilt >= tilt_min and dense_tilt_support and not independent_tilt_ok:
        cls, reason = "UNCERTAIN", "conic_tilt_conflicts_with_independent_ring_gradient"
    elif tilt >= tilt_min:
        cls, reason = "UNCERTAIN", "tilt_plane_lacks_dense_annulus_depth_consensus"
    else:
        cls, reason = "UNCERTAIN", "conic_validated_tilt_transition_band"
    stable = bool(base_stable and ((tilt <= flat_max and flat_dense_support and float(cv.get("inner_radius_error_ratio", 99.0)) <= flat_max_radius_error) or (tilt >= tilt_min and dense_tilt_support and independent_tilt_ok)))
    return {
        **base,
        "status": "RECONSTRUCTED" if stable else "RECONSTRUCTED_LOW_CONFIDENCE",
        "classification": cls,
        "reason": reason,
        "jackknife": jack,
        "independent_tilt_crosscheck": {
            "required": require_independent,
            "state": prior_state,
            "confidence": prior.get("confidence"),
            "reason": prior.get("classification_reason") or prior.get("reason"),
            "tilt_consensus_ok": bool(independent_tilt_ok),
        },
        "surface": surface,
        "thresholds": {
            "classification_minimum_sectors": min_sec,
            "classification_minimum_coverage_deg": min_cov,
            "classification_maximum_residual_mm": max_res,
            "classification_maximum_jackknife_tilt_std_deg": max_jack,
            "classification_maximum_conic_radius_error_ratio": max_radius_err,
            "classification_maximum_circle_residual_p90_mm": max_circle_p90,
            "classification_maximum_reprojection_p90_px": max_chamfer,
            "classification_minimum_semantic_support_ratio": min_semantic,
            "classification_minimum_annulus_depth_support_sectors": min_depth_sectors,
            "classification_minimum_annulus_depth_support_ratio": min_depth_ratio,
            "classification_maximum_annulus_depth_residual_median_mm": max_depth_residual,
            "flat_minimum_annulus_depth_support_sectors": flat_min_depth_sectors,
            "flat_minimum_annulus_depth_support_ratio": flat_min_depth_ratio,
            "flat_maximum_annulus_depth_residual_median_mm": flat_max_depth_residual,
            "flat_maximum_conic_radius_error_ratio": flat_max_radius_error,
            "require_m39_3_1_tilt_consensus": require_independent,
            "flat_tilt_max_deg": flat_max,
            "tilted_tilt_min_deg": tilt_min,
            "maximum_physical_peak_to_peak_mm": physical_p2p,
        },
    }
