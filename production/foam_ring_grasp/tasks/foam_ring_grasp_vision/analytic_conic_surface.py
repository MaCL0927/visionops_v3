"""M39.3.4 analytic conic pose candidates + dense annulus depth disambiguation.

The visible ``ring_mouth`` is treated as the projection of the known 60 mm
inner circle.  A calibrated-image conic has only a small set of physically
possible circle-plane normals.  M39.3.4 therefore replaces the M39.3.3
3-point/plane enumeration with two analytic conic normals plus one explicit
flat reference.  Full front-annulus depth is then used to select between them.

This module is diagnostic-only.  M39.2.9 remains the production flat path until
front-visible ring handling is frozen.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .conic_ring_surface import fit_mouth_conic


def _f(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


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


def _project(point: Sequence[float], intrinsics: Mapping[str, float]) -> Optional[Tuple[float, float]]:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)) or float(p[2]) <= 1e-6:
        return None
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    return (float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy))


def _fit_circle_2d(points: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    if points.shape[0] < 8:
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
    if not np.isfinite(r2) or r2 <= 1e-10:
        return None
    radius = math.sqrt(r2)
    residuals = np.abs(np.linalg.norm(points - center[None, :], axis=1) - radius)
    return center, float(radius), residuals


def _ellipse_conic_matrix(conic: Mapping[str, Any]) -> Optional[np.ndarray]:
    center = conic.get("center_uv") or []
    if len(center) != 2:
        return None
    # fit_mouth_conic stores major/minor but OpenCV's angle refers to its first
    # fitted diameter.  If raw diameters are unavailable, align the major axis
    # with angle_deg as a stable fallback.
    raw_w = _f(conic.get("ellipse_width_px"), _f(conic.get("major_px"), 0.0))
    raw_h = _f(conic.get("ellipse_height_px"), _f(conic.get("minor_px"), 0.0))
    angle_deg = _f(conic.get("angle_deg"), 0.0)
    if raw_w <= 1e-6 or raw_h <= 1e-6:
        return None
    a = 0.5 * raw_w
    b = 0.5 * raw_h
    th = math.radians(angle_deg)
    R = np.asarray([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float64)
    A = R @ np.diag([1.0 / (a * a), 1.0 / (b * b)]) @ R.T
    c = np.asarray([float(center[0]), float(center[1])], dtype=np.float64)
    C = np.zeros((3, 3), dtype=np.float64)
    C[:2, :2] = A
    C[:2, 2] = -A @ c
    C[2, :2] = C[:2, 2]
    C[2, 2] = float(c @ A @ c - 1.0)
    return C


def analytic_circle_normals(
    conic: Mapping[str, Any], intrinsics: Mapping[str, float]
) -> tuple[list[np.ndarray], Dict[str, Any]]:
    """Return the two calibrated single-circle normal candidates.

    For normalized conic Q with eigenvalues lambda1 >= lambda2 > 0 > lambda3,
    the two circle-plane normal directions are the symmetric combinations of
    the lambda1 and lambda3 eigenvectors.  Sign is oriented toward the camera
    (negative camera Z) afterwards.
    """
    C = _ellipse_conic_matrix(conic)
    if C is None:
        return [], {"available": False, "reason": "ellipse_conic_unavailable"}
    K = np.asarray(
        [[float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
         [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Q = K.T @ C @ K
    Q = 0.5 * (Q + Q.T)
    vals, vecs = np.linalg.eigh(Q)
    # A valid normalized ellipse cone has two eigenvalues of one sign and one
    # of the opposite sign.  Scale is arbitrary, so flip if necessary.
    if int(np.count_nonzero(vals > 0.0)) == 1:
        vals = -vals
        Q = -Q
        vals, vecs = np.linalg.eigh(Q)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    if not (vals[0] < 0.0 < vals[1] <= vals[2]):
        return [], {
            "available": False,
            "reason": "normalized_conic_eigen_signature_invalid",
            "eigenvalues": [float(v) for v in vals.tolist()],
        }
    lam3, lam2, lam1 = float(vals[0]), float(vals[1]), float(vals[2])
    denom = lam1 - lam3
    if denom <= 1e-12:
        return [], {"available": False, "reason": "normalized_conic_eigen_degenerate"}
    alpha2 = (lam1 - lam2) / denom
    beta2 = (lam2 - lam3) / denom
    if alpha2 < -1e-8 or beta2 < -1e-8:
        return [], {"available": False, "reason": "analytic_normal_terms_invalid"}
    alpha = math.sqrt(max(0.0, alpha2))
    beta = math.sqrt(max(0.0, beta2))
    v_high = vecs[:, 2]
    v_neg = vecs[:, 0]
    normals = []
    for sign in (+1.0, -1.0):
        n = alpha * v_high + sign * beta * v_neg
        n = _norm(n)
        if float(n[2]) > 0.0:
            n = -n
        # suppress duplicate near-frontal solutions
        if not any(abs(float(np.dot(n, old))) > 0.999999 for old in normals):
            normals.append(n)
    return normals, {
        "available": bool(normals),
        "eigenvalues": [lam3, lam2, lam1],
        "alpha": float(alpha),
        "beta": float(beta),
        "candidate_count": int(len(normals)),
    }


def _plane_basis(normal: np.ndarray, preferred_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = preferred_axis - float(np.dot(preferred_axis, normal)) * normal
    if float(np.linalg.norm(e1)) < 1e-7:
        seed = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, normal))) > 0.9:
            seed = np.asarray([0.0, 1.0, 0.0])
        e1 = seed - float(np.dot(seed, normal)) * normal
    e1 = _norm(e1)
    e2 = _norm(np.cross(normal, e1))
    return e1, e2


def _candidate_from_normal(
    label: str,
    normal_toward_camera: np.ndarray,
    conic: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    *,
    inner_radius_mm: float,
    outer_radius_mm: float,
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z: np.ndarray,
    plane_d_toward_camera: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    contour = np.asarray(conic.get("sampled_contour_uv") or [], dtype=np.float64)
    if contour.ndim != 2 or contour.shape[0] < 12:
        return None
    n_toward = _norm(normal_toward_camera)
    if float(n_toward[2]) > 0.0:
        n_toward = -n_toward
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    rays = np.column_stack([
        (contour[:, 0] - cx) / fx,
        (contour[:, 1] - cy) / fy,
        np.ones(contour.shape[0], dtype=np.float64),
    ])
    denom = rays @ n_toward
    good = np.abs(denom) > 1e-8
    if int(np.count_nonzero(good)) < 12:
        return None
    rays = rays[good]
    denom = denom[good]
    d_seed = -1.0 if float(np.median(denom)) < 0.0 else 1.0
    if plane_d_toward_camera is None:
        z_unit = d_seed / denom
        good_z = np.isfinite(z_unit) & (z_unit > 0.0)
        if int(np.count_nonzero(good_z)) < 12:
            return None
        points_unit = rays[good_z] * z_unit[good_z, None]
        e1, e2 = _plane_basis(n_toward, box_x)
        origin_unit = np.mean(points_unit, axis=0)
        xy_unit = np.column_stack([
            (points_unit - origin_unit) @ e1,
            (points_unit - origin_unit) @ e2,
        ])
        fit = _fit_circle_2d(xy_unit)
        if fit is None:
            return None
        center2_unit, radius_unit, residual_unit = fit
        if radius_unit <= 1e-9:
            return None
        scale = float(inner_radius_mm / radius_unit)
        d_toward = float(d_seed * scale)
        center3 = (origin_unit + center2_unit[0] * e1 + center2_unit[1] * e2) * scale
        circle_residuals = residual_unit * scale
        recovered_radius = float(inner_radius_mm)
        good_z_final = good_z
    else:
        d_toward = float(plane_d_toward_camera)
        z = d_toward / denom
        good_z_final = np.isfinite(z) & (z > 100.0) & (z < 3000.0)
        if int(np.count_nonzero(good_z_final)) < 12:
            return None
        pts = rays[good_z_final] * z[good_z_final, None]
        e1, e2 = _plane_basis(n_toward, box_x)
        origin = np.mean(pts, axis=0)
        xy = np.column_stack([(pts - origin) @ e1, (pts - origin) @ e2])
        fit = _fit_circle_2d(xy)
        if fit is None:
            return None
        center2, recovered_radius, circle_residuals = fit
        center3 = origin + center2[0] * e1 + center2[1] * e2
    if not np.all(np.isfinite(center3)) or not (100.0 <= float(center3[2]) <= 3000.0):
        return None

    # ideal projected rims
    phis = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    predicted_inner, predicted_outer = [], []
    for phi in phis:
        direction = math.cos(phi) * e1 + math.sin(phi) * e2
        uv_i = _project(center3 + inner_radius_mm * direction, intrinsics)
        uv_o = _project(center3 + outer_radius_mm * direction, intrinsics)
        if uv_i is not None:
            predicted_inner.append(uv_i)
        if uv_o is not None:
            predicted_outer.append(uv_o)
    if len(predicted_inner) < 32 or len(predicted_outer) < 32:
        return None
    obs = contour[good][good_z_final]
    pred = np.asarray(predicted_inner, dtype=np.float64)
    dist = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
    chamfer = np.concatenate([np.min(dist, axis=1), np.min(dist, axis=0)])

    # express plane in box coordinates for downstream reuse
    n_into = -n_toward
    d_into = -d_toward
    nz = float(np.dot(n_into, box_z))
    if nz < 0.0:
        n_into = -n_into
        d_into = -d_into
        nz = -nz
    if nz <= 1e-6:
        return None
    nx = float(np.dot(n_into, box_x)); ny = float(np.dot(n_into, box_y))
    gx, gy = -nx / nz, -ny / nz
    h0 = d_into / nz
    tilt = math.degrees(math.acos(float(np.clip(np.dot(_norm(n_into), box_z), -1.0, 1.0))))
    direction_box = float((math.degrees(math.atan2(gy, gx)) + 360.0) % 360.0)
    return {
        "label": str(label),
        "normal_toward_camera": [float(v) for v in n_toward.tolist()],
        "normal_into_box": [float(v) for v in _norm(n_into).tolist()],
        "plane_d_toward_camera": float(d_toward),
        "coefficients_box": [float(gx), float(gy), float(h0)],
        "circle_center_camera_mm": [float(v) for v in center3.tolist()],
        "plane_basis_u_camera": [float(v) for v in e1.tolist()],
        "plane_basis_v_camera": [float(v) for v in e2.tolist()],
        "inner_radius_mm": float(recovered_radius),
        "inner_radius_error_ratio": float(abs(float(recovered_radius) - float(inner_radius_mm)) / max(float(inner_radius_mm), 1e-6)),
        "circle_radial_residual_median_mm": float(np.median(circle_residuals)),
        "circle_radial_residual_p90_mm": float(np.percentile(circle_residuals, 90)),
        "reprojection_chamfer_median_px": float(np.median(chamfer)),
        "reprojection_chamfer_p90_px": float(np.percentile(chamfer, 90)),
        "tilt_deg": float(tilt),
        "tilt_direction_deg_box": direction_box,
        "predicted_inner_rim_uv": [[float(u), float(v)] for u, v in predicted_inner],
        "predicted_outer_rim_uv": [[float(u), float(v)] for u, v in predicted_outer],
    }


def _mouth_band_plane_consistency(
    normal_toward_camera: np.ndarray,
    depth_mm: np.ndarray,
    ring_mask: np.ndarray,
    mouth_mask: np.ndarray,
    conic: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    *,
    inner_radius_mm: float,
    outer_radius_mm: float,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate one plane normal on the mouth-adjacent annulus.

    Correct orientation turns front-surface depth into an almost constant plane
    constant d=n·X.  Each angular sector independently selects its nearest
    supported local d-cluster, so a globally tilted face is preserved while
    lower rings/box surfaces behind the opening do not dominate.
    """
    depth = np.asarray(depth_mm)
    ring = np.asarray(ring_mask, dtype=bool)
    mouth = np.asarray(mouth_mask, dtype=bool)
    minor = _f(conic.get("minor_px"), 0.0)
    center = conic.get("center_uv") or []
    if minor <= 2.0 or len(center) != 2:
        return None
    radius_px = 0.5 * minor
    wall_ratio = max(0.05, outer_radius_mm / max(inner_radius_mm, 1e-6) - 1.0)
    expected_wall_px = max(3.0, radius_px * wall_ratio)
    guard_px = max(1, _i(config.get("depth_anchor_inner_guard_px"), 2))
    outer_px = max(guard_px + 2, int(round(_f(config.get("depth_anchor_wall_fraction"), 0.70) * expected_wall_px)))
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_px + 1, 2 * outer_px + 1))
    inner_d = cv2.dilate(mouth.astype(np.uint8), k1).astype(bool)
    outer_d = cv2.dilate(mouth.astype(np.uint8), k2).astype(bool)
    band = outer_d & ~inner_d & ring
    valid = band & np.isfinite(depth) & (depth >= _f(config.get("minimum_depth_mm"), 150.0)) & (depth <= _f(config.get("maximum_depth_mm"), 3000.0))
    yy, xx = np.nonzero(valid)
    if len(xx) < max(12, _i(config.get("depth_anchor_minimum_pixels"), 24)):
        return None
    z = depth[yy, xx].astype(np.float64)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"]); cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    n = _norm(normal_toward_camera)
    denom = n[0] * ((xx.astype(np.float64) - cx) / fx) + n[1] * ((yy.astype(np.float64) - cy) / fy) + n[2]
    dvals = z * denom
    good = np.isfinite(dvals) & (np.abs(denom) > 1e-8)
    dvals = dvals[good]; z = z[good]; xx = xx[good]; yy = yy[good]
    if dvals.size < 12:
        return None
    sector_count = max(8, _i(config.get("depth_anchor_sector_count"), 16))
    theta = (np.arctan2(yy.astype(np.float64) - float(center[1]), xx.astype(np.float64) - float(center[0])) + 2.0 * math.pi) % (2.0 * math.pi)
    sector_ids = np.floor(theta * sector_count / (2.0 * math.pi)).astype(np.int32)
    gap = max(1.0, _f(config.get("depth_anchor_cluster_gap_mm"), 3.0))
    min_cluster = max(4, _i(config.get("depth_anchor_cluster_minimum_pixels"), 5))
    reps = []
    rows = []
    for sector in range(sector_count):
        idx = np.flatnonzero(sector_ids == sector)
        if idx.size < min_cluster:
            rows.append({"sector": sector, "available": False})
            continue
        order = idx[np.argsort(dvals[idx])]
        groups = [[int(order[0])]]
        for k in order[1:]:
            if abs(float(dvals[int(k)] - dvals[groups[-1][-1]])) <= gap:
                groups[-1].append(int(k))
            else:
                groups.append([int(k)])
        clusters = []
        for g in groups:
            if len(g) < min_cluster:
                continue
            garr = np.asarray(g, dtype=np.int32)
            clusters.append((float(np.median(z[garr])), -len(g), float(np.median(dvals[garr])), len(g)))
        if not clusters:
            rows.append({"sector": sector, "available": False})
            continue
        zmed, _, dmed, support = sorted(clusters)[0]
        reps.append((sector, dmed, zmed, support))
        rows.append({"sector": sector, "available": True, "plane_d": dmed, "median_depth_mm": zmed, "support_count": support})
    min_sectors = max(4, _i(config.get("depth_anchor_minimum_sectors"), 5))
    if len(reps) < min_sectors:
        return None
    arr = np.asarray([r[1] for r in reps], dtype=np.float64)
    dmed = float(np.median(arr)); residuals = np.abs(arr - dmed)
    sectors = [int(r[0]) for r in reps]
    if len(sectors) == 1:
        coverage = 360.0 / sector_count
    else:
        vals = sorted(set(sectors)); wrapped = vals[1:] + [vals[0] + sector_count]
        max_gap = max(b-a for a,b in zip(vals,wrapped))
        coverage = max(0.0, 360.0 - max_gap * 360.0 / sector_count + 360.0 / sector_count)
    return {
        "plane_d_toward_camera": dmed,
        "median_depth_mm": float(np.median([r[2] for r in reps])),
        "valid_sector_count": int(len(reps)),
        "angular_coverage_deg": float(coverage),
        "residual_median_mm": float(np.median(residuals)),
        "residual_p90_mm": float(np.percentile(residuals, 90)),
        "band_pixel_count": int(len(xx)),
        "guard_px": int(guard_px),
        "outer_px": int(outer_px),
        "sector_samples": rows,
    }


def _polygon_mask(shape: tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask.astype(bool)
    pts = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _dense_annulus_score(
    candidate: Dict[str, Any],
    depth_mm: np.ndarray,
    ring_mask: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    inner_radius_mm: float,
    outer_radius_mm: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    depth = np.asarray(depth_mm)
    ring = np.asarray(ring_mask, dtype=bool)
    h, w = depth.shape
    center3 = np.asarray(candidate["circle_center_camera_mm"], dtype=np.float64)
    e1 = np.asarray(candidate["plane_basis_u_camera"], dtype=np.float64)
    e2 = np.asarray(candidate["plane_basis_v_camera"], dtype=np.float64)
    n = np.asarray(candidate["normal_toward_camera"], dtype=np.float64)
    d = float(candidate["plane_d_toward_camera"])
    wall = outer_radius_mm - inner_radius_mm
    inner_frac = float(np.clip(_f(config.get("dense_annulus_inner_fraction"), 0.20), 0.0, 0.7))
    outer_frac = float(np.clip(_f(config.get("dense_annulus_outer_fraction"), 0.75), inner_frac + 0.08, 0.98))
    r1 = inner_radius_mm + inner_frac * wall
    r2 = inner_radius_mm + outer_frac * wall
    phis = np.linspace(0.0, 2.0 * math.pi, max(96, _i(config.get("dense_annulus_polygon_points"), 160)), endpoint=False)
    inner_uv, outer_uv = [], []
    for phi in phis:
        direction = math.cos(phi) * e1 + math.sin(phi) * e2
        ui = _project(center3 + r1 * direction, intrinsics)
        uo = _project(center3 + r2 * direction, intrinsics)
        if ui is not None: inner_uv.append(ui)
        if uo is not None: outer_uv.append(uo)
    if len(inner_uv) < 32 or len(outer_uv) < 32:
        return {"available": False, "reason": "predicted_dense_annulus_unavailable"}
    outer_mask = _polygon_mask((h, w), outer_uv)
    inner_mask = _polygon_mask((h, w), inner_uv)
    predicted = outer_mask & ~inner_mask
    predicted_count = int(np.count_nonzero(predicted))
    if predicted_count <= 0:
        return {"available": False, "reason": "predicted_dense_annulus_empty"}
    owned = predicted & ring
    owned_count = int(np.count_nonzero(owned))
    semantic_ratio = float(owned_count / max(predicted_count, 1))
    min_depth = _f(config.get("minimum_depth_mm"), 150.0)
    max_depth = _f(config.get("maximum_depth_mm"), 3000.0)
    valid = owned & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    valid_count = int(np.count_nonzero(valid))
    valid_ratio = float(valid_count / max(owned_count, 1))
    if valid_count <= 0:
        return {
            "available": True,
            "semantic_support_ratio": semantic_ratio,
            "valid_depth_ratio": valid_ratio,
            "inlier_ratio": 0.0,
            "supported_sector_count": 0,
            "angular_coverage_deg": 0.0,
            "residual_median_mm": None,
            "residual_p90_mm": None,
            "score": 0.0,
            "predicted_pixel_count": predicted_count,
            "owned_pixel_count": owned_count,
            "valid_pixel_count": valid_count,
        }
    yy, xx = np.nonzero(valid)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    rx = (xx.astype(np.float64) - cx) / fx
    ry = (yy.astype(np.float64) - cy) / fy
    denom = n[0] * rx + n[1] * ry + n[2]
    expected = np.full(denom.shape, np.nan, dtype=np.float64)
    good = np.abs(denom) > 1e-8
    expected[good] = d / denom[good]
    measured = depth[yy, xx].astype(np.float64)
    good &= np.isfinite(expected) & (expected >= min_depth) & (expected <= max_depth)
    residual = np.abs(measured[good] - expected[good])
    if residual.size == 0:
        return {"available": False, "reason": "candidate_expected_depth_invalid"}
    inlier_mm = max(2.0, _f(config.get("dense_depth_inlier_mm"), 8.0))
    inlier = residual <= inlier_mm
    inlier_ratio = float(np.mean(inlier))
    med = float(np.median(residual)); p90 = float(np.percentile(residual, 90))

    # Lightweight angular coverage from predicted annulus samples.  This is not
    # used to fit a plane; it only verifies that depth support spans the ring.
    sector_count = max(8, _i(config.get("dense_depth_sector_count"), 16))
    supported = []
    sector_rows = []
    radial_fracs = config.get("dense_sector_radial_fractions") or [0.30, 0.50, 0.70]
    try:
        radial_fracs = [float(v) for v in radial_fracs]
    except Exception:
        radial_fracs = [0.30, 0.50, 0.70]
    patch = max(0, _i(config.get("dense_sector_patch_radius_px"), 1))
    for sector in range(sector_count):
        phi = (sector + 0.5) * 2.0 * math.pi / sector_count
        direction = math.cos(phi) * e1 + math.sin(phi) * e2
        local_residuals = []
        for frac in radial_fracs:
            rr = inner_radius_mm + float(frac) * wall
            uv = _project(center3 + rr * direction, intrinsics)
            if uv is None: continue
            x0, y0 = int(round(uv[0])), int(round(uv[1]))
            xs = np.arange(max(0, x0 - patch), min(w, x0 + patch + 1))
            ys = np.arange(max(0, y0 - patch), min(h, y0 + patch + 1))
            if xs.size == 0 or ys.size == 0: continue
            gxv, gyv = np.meshgrid(xs, ys)
            own = ring[gyv, gxv]
            zz = depth[gyv, gxv].astype(np.float64)
            vv = own & np.isfinite(zz) & (zz >= min_depth) & (zz <= max_depth)
            if not np.any(vv): continue
            rxv = (gxv.astype(np.float64) - cx) / fx
            ryv = (gyv.astype(np.float64) - cy) / fy
            den = n[0] * rxv + n[1] * ryv + n[2]
            exp = np.where(np.abs(den) > 1e-8, d / den, np.nan)
            rr_local = np.abs(zz - exp)
            vals = rr_local[vv & np.isfinite(exp)]
            if vals.size:
                local_residuals.append(float(np.median(vals)))
        sector_ok = bool(len(local_residuals) >= 2 and sum(v <= inlier_mm for v in local_residuals) >= 2)
        if sector_ok: supported.append(sector)
        sector_rows.append({
            "sector": int(sector),
            "supported": sector_ok,
            "valid_sample_count": int(len(local_residuals)),
            "residual_median_mm": float(np.median(local_residuals)) if local_residuals else None,
        })
    if supported:
        vals = sorted(set(supported)); wrapped = vals[1:] + [vals[0] + sector_count]
        max_gap = max(b - a for a, b in zip(vals, wrapped)) if len(vals) > 1 else sector_count - 1
        coverage = max(0.0, 360.0 - max_gap * 360.0 / sector_count + 360.0 / sector_count)
    else:
        coverage = 0.0

    geometry_quality = math.exp(-max(0.0, _f(candidate.get("circle_radial_residual_p90_mm"), 99.0)) / 5.0)
    reproj_quality = math.exp(-max(0.0, _f(candidate.get("reprojection_chamfer_p90_px"), 99.0)) / 5.0)
    residual_quality = math.exp(-med / max(inlier_mm, 1.0))
    p90_quality = math.exp(-p90 / max(2.0 * inlier_mm, 1.0))
    coverage_ratio = float(coverage / 360.0)
    score = (
        0.34 * inlier_ratio
        + 0.14 * valid_ratio
        + 0.12 * semantic_ratio
        + 0.12 * coverage_ratio
        + 0.12 * residual_quality
        + 0.06 * p90_quality
        + 0.06 * geometry_quality
        + 0.04 * reproj_quality
    )
    return {
        "available": True,
        "semantic_support_ratio": semantic_ratio,
        "valid_depth_ratio": valid_ratio,
        "inlier_ratio": inlier_ratio,
        "supported_sector_count": int(len(supported)),
        "valid_sector_count": int(sum(int(r["valid_sample_count"] > 0) for r in sector_rows)),
        "angular_coverage_deg": float(coverage),
        "residual_median_mm": med,
        "residual_p90_mm": p90,
        "score": float(score),
        "predicted_pixel_count": predicted_count,
        "owned_pixel_count": owned_count,
        "valid_pixel_count": valid_count,
        "sector_samples": sector_rows,
    }


def _candidate_is_usable(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    dense = candidate.get("dense_depth") or {}
    if not dense.get("available"):
        return False
    return bool(
        _f(candidate.get("circle_radial_residual_p90_mm"), 99.0) <= _f(config.get("maximum_circle_residual_p90_mm"), 6.0)
        and _f(candidate.get("reprojection_chamfer_p90_px"), 99.0) <= _f(config.get("maximum_reprojection_p90_px"), 6.0)
        and float(dense.get("semantic_support_ratio") or 0.0) >= _f(config.get("minimum_semantic_support_ratio"), 0.30)
        and float(dense.get("valid_depth_ratio") or 0.0) >= _f(config.get("minimum_valid_depth_ratio"), 0.22)
        and float(dense.get("inlier_ratio") or 0.0) >= _f(config.get("minimum_inlier_ratio"), 0.28)
        and float(dense.get("angular_coverage_deg") or 0.0) >= _f(config.get("minimum_angular_coverage_deg"), 112.5)
        and _f(dense.get("residual_median_mm"), 99.0) <= _f(config.get("maximum_residual_median_mm"), 10.0)
    )


def reconstruct_analytic_conic_surface(
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
    started = time.perf_counter()
    cfg = dict(config or {})
    box_x, box_y, box_z = _norm(box_x_camera), _norm(box_y_camera), _norm(box_z_inside_camera)
    rin = 0.5 * _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0)
    rout = 0.5 * _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    conic = fit_mouth_conic(np.asarray(mouth_mask, dtype=bool))
    timings["fit_mouth_conic_ms"] = (time.perf_counter() - t0) * 1000.0
    if not conic.get("available"):
        return {
            "schema_version": "1.0", "stage": "M39.3.4_analytic_conic_dense_depth",
            "mode": "online_diagnostic_only", "production_routing_enabled": False,
            "classification": "UNCERTAIN", "status": "UNCERTAIN",
            "reason": str(conic.get("reason") or "mouth_conic_unavailable"),
            "mouth_conic": conic, "candidates": [], "selected_candidate": None,
            "timing_ms": {**timings, "total_ms": (time.perf_counter() - started) * 1000.0},
        }

    t0 = time.perf_counter()
    normals, analytic_info = analytic_circle_normals(conic, intrinsics)
    timings["analytic_candidate_generation_ms"] = (time.perf_counter() - t0) * 1000.0
    candidates: list[Dict[str, Any]] = []
    normal_specs = [(f"CONIC_{'A' if idx == 0 else 'B'}", normal) for idx, normal in enumerate(normals)]
    normal_specs.append(("FLAT_REFERENCE", -box_z))
    for label, normal in normal_specs:
        anchor = _mouth_band_plane_consistency(
            normal, depth_mm, ring_mask, mouth_mask, conic, intrinsics,
            inner_radius_mm=rin, outer_radius_mm=rout, config=cfg,
        )
        cand = None
        if anchor is not None:
            cand = _candidate_from_normal(
                label, normal, conic, intrinsics,
                inner_radius_mm=rin, outer_radius_mm=rout,
                box_x=box_x, box_y=box_y, box_z=box_z,
                plane_d_toward_camera=float(anchor["plane_d_toward_camera"]),
            )
        if cand is None:
            cand = _candidate_from_normal(
                label, normal, conic, intrinsics,
                inner_radius_mm=rin, outer_radius_mm=rout,
                box_x=box_x, box_y=box_y, box_z=box_z,
            )
        if cand is not None:
            cand["depth_anchor"] = anchor
            candidates.append(cand)
    # de-duplicate effectively identical frontal analytic/flat candidates
    unique: list[Dict[str, Any]] = []
    for cand in candidates:
        n = np.asarray(cand["normal_toward_camera"], dtype=np.float64)
        if cand["label"] != "FLAT_REFERENCE" and any(
            old["label"] != "FLAT_REFERENCE" and abs(float(np.dot(n, np.asarray(old["normal_toward_camera"])))) > 0.99999
            for old in unique
        ):
            continue
        unique.append(cand)
    candidates = unique

    t0 = time.perf_counter()
    for cand in candidates:
        cand["dense_depth"] = _dense_annulus_score(
            cand, depth_mm, ring_mask, intrinsics,
            inner_radius_mm=rin, outer_radius_mm=rout, config=cfg,
        )
        band = cand.get("depth_anchor") or {}
        band_med = _f(band.get("residual_median_mm"), 99.0)
        band_quality = math.exp(-band_med / max(1.0, _f(cfg.get("mouth_band_score_scale_mm"), 10.0)))
        dense_score = float((cand.get("dense_depth") or {}).get("score") or 0.0)
        cand["evidence_score"] = float(0.68 * band_quality + 0.32 * dense_score)
        # Usability is deliberately less strict than M39.3.3: a high-quality
        # mouth-adjacent plane can compensate for partial outer-annulus depth.
        dense = cand.get("dense_depth") or {}
        cand["usable"] = bool(
            band and int(band.get("valid_sector_count") or 0) >= _i(cfg.get("minimum_mouth_band_sectors"), 5)
            and float(band.get("angular_coverage_deg") or 0.0) >= _f(cfg.get("minimum_mouth_band_coverage_deg"), 112.5)
            and _f(cand.get("circle_radial_residual_p90_mm"), 99.0) <= _f(cfg.get("maximum_circle_residual_p90_mm"), 7.0)
            and _f(cand.get("reprojection_chamfer_p90_px"), 99.0) <= _f(cfg.get("maximum_reprojection_p90_px"), 7.0)
            and (
                _f(band.get("residual_median_mm"), 99.0) <= _f(cfg.get("maximum_mouth_band_residual_median_mm"), 20.0)
                or (dense.get("available") and float(dense.get("inlier_ratio") or 0.0) >= _f(cfg.get("minimum_dense_inlier_ratio_for_rescue"), 0.40))
            )
        )
    timings["dense_annulus_depth_ms"] = (time.perf_counter() - t0) * 1000.0

    usable = [c for c in candidates if c.get("usable")]
    usable.sort(key=lambda c: float(c.get("evidence_score") or 0.0), reverse=True)
    prior = dict(prior_tilt_evidence or {})
    prior_state = str(prior.get("state") or prior.get("classification") or "UNAVAILABLE").upper()
    prior_conf = str(prior.get("confidence") or "").lower()
    flat_max = _f(cfg.get("flat_tilt_max_deg"), 8.0)
    tilt_min = _f(cfg.get("tilted_tilt_min_deg"), 10.0)
    min_tilt_margin = _f(cfg.get("minimum_tilt_winner_margin"), 0.045)
    min_flat_margin = _f(cfg.get("minimum_flat_winner_margin"), 0.015)
    conflict_margin = _f(cfg.get("m3931_conflict_required_margin"), 0.085)
    supportive_margin = _f(cfg.get("m3931_supportive_required_margin"), 0.025)
    physical_max_tilt = _f(cfg.get("maximum_accepted_tilt_deg"), 35.0)

    classification = "UNCERTAIN"
    reason = "no_usable_analytic_candidate"
    selected = usable[0] if usable else None
    runner = usable[1] if len(usable) > 1 else None
    winner_margin = None
    analytic_candidates = [c for c in candidates if c.get("label") != "FLAT_REFERENCE"]
    analytic_min_tilt = min((_f(c.get("tilt_deg"), 999.0) for c in analytic_candidates), default=999.0)
    flat_candidate_all = next((c for c in candidates if c.get("label") == "FLAT_REFERENCE"), None)
    flat_band = (flat_candidate_all or {}).get("depth_anchor") or {}
    flat_band_residual = _f(flat_band.get("residual_median_mm"), 99.0)
    strong_flat_band = bool(
        flat_candidate_all is not None
        and int(flat_band.get("valid_sector_count") or 0) >= _i(cfg.get("strong_flat_minimum_sectors"), 8)
        and flat_band_residual <= _f(cfg.get("strong_flat_maximum_mouth_band_residual_mm"), 4.5)
    )
    center_c = conic.get("center_uv") or [float(intrinsics["cx"]), float(intrinsics["cy"])]
    off_axis_angle_deg = math.degrees(math.atan(math.hypot(
        (float(center_c[0]) - float(intrinsics["cx"])) / float(intrinsics["fx"]),
        (float(center_c[1]) - float(intrinsics["cy"])) / float(intrinsics["fy"]),
    )))
    definite_tilt_min = _f(cfg.get("definite_analytic_tilt_min_deg"), 12.0)
    near_flat_analytic_max = _f(cfg.get("near_flat_analytic_max_deg"), 10.0)

    # Stage 1: classify FLAT vs clearly non-flat. A very coherent flat mouth
    # band can override a spurious conic decomposition (important for old flat
    # counterexamples). Otherwise if both analytic solutions are clearly tilted,
    # do not fall back to FLAT merely because the flat plane locally fits a few
    # pixels.
    if strong_flat_band:
        selected = flat_candidate_all
        classification, reason = "FLAT", "strong_mouth_band_confirms_operational_flat_reference"
    elif analytic_min_tilt <= near_flat_analytic_max and flat_candidate_all is not None:
        flat_usable = bool(flat_candidate_all.get("usable"))
        if flat_usable or prior_state != "TILTED":
            selected = flat_candidate_all
            classification, reason = "FLAT", "analytic_geometry_contains_near_flat_solution"

    if classification == "UNCERTAIN" and analytic_min_tilt >= definite_tilt_min:
        # Far off the optical axis, segmentation/conic bias can create a false
        # tilted analytic solution. If the independent ring gradient says FLAT
        # and the explicit flat reference has at least as much direct depth
        # evidence as every tilt candidate, keep the proven M39.2.9 flat path.
        if prior_state == "FLAT" and off_axis_angle_deg >= _f(cfg.get("off_axis_flat_protection_deg"), 12.0) and flat_candidate_all is not None and flat_candidate_all.get("usable"):
            best_tilt_all = max((float(c.get("evidence_score") or 0.0) for c in candidates if c.get("label") != "FLAT_REFERENCE"), default=0.0)
            if float(flat_candidate_all.get("evidence_score") or 0.0) >= best_tilt_all:
                selected = flat_candidate_all
                classification, reason = "FLAT", "off_axis_conic_ambiguity_prefers_depth_confirmed_flat_reference"
        tilt_usable = [c for c in usable if c.get("label") != "FLAT_REFERENCE" and _f(c.get("tilt_deg"), 999.0) <= physical_max_tilt] if classification == "UNCERTAIN" else []
        tilt_usable.sort(key=lambda c: float(c.get("evidence_score") or 0.0), reverse=True)
        if tilt_usable:
            best_tilt = tilt_usable[0]
            second_tilt = tilt_usable[1] if len(tilt_usable) > 1 else None
            tscore = float(best_tilt.get("evidence_score") or 0.0)
            second_tscore = float(second_tilt.get("evidence_score") or 0.0) if second_tilt is not None else 0.0
            tilt_margin = tscore - second_tscore
            # A/B direction ambiguity is the only reason to retain UNCERTAIN at
            # this point. M39.3.1 may lower, but never remove, the required margin.
            req = _f(cfg.get("minimum_analytic_ab_margin"), 0.035)
            if prior_state == "TILTED":
                req = min(req, _f(cfg.get("minimum_analytic_ab_margin_with_m3931"), 0.020))
            if second_tilt is None or tilt_margin >= req:
                selected = best_tilt
                classification, reason = "TILTED", "analytic_geometry_nonflat_and_depth_selects_pose_branch"
                winner_margin = float(tilt_margin)
            else:
                selected = best_tilt
                classification, reason = "UNCERTAIN", "analytic_ab_direction_ambiguity_remains"
                winner_margin = float(tilt_margin)
        else:
            classification, reason = "UNCERTAIN", "analytic_nonflat_but_depth_support_insufficient"

    if classification == "UNCERTAIN" and analytic_min_tilt < definite_tilt_min and selected is not None:
        # 10-12 degree transition band: use the best evidence candidate and the
        # lightweight M39.3.1 state as a tie-break, not a hard veto.
        best_score = float(selected.get("evidence_score") or 0.0)
        second_score = float(runner.get("evidence_score") or 0.0) if runner is not None else 0.0
        winner_margin = best_score - second_score
        if selected.get("label") == "FLAT_REFERENCE" and prior_state != "TILTED":
            classification, reason = "FLAT", "transition_band_prefers_flat_reference"
        elif selected.get("label") != "FLAT_REFERENCE" and prior_state == "TILTED" and winner_margin >= supportive_margin:
            classification, reason = "TILTED", "transition_band_depth_plus_m3931_support"
        else:
            classification, reason = "UNCERTAIN", "tilt_transition_band_not_decisive"
    # User-facing selected surface is the winner actually used by the router.
    selected_surface = None
    if selected is not None:
        selected_surface = {
            "candidate_label": selected["label"],
            "tilt_deg": selected["tilt_deg"],
            "plane_d_toward_camera": selected.get("plane_d_toward_camera"),
            "plane_basis_u_camera": selected.get("plane_basis_u_camera"),
            "plane_basis_v_camera": selected.get("plane_basis_v_camera"),
            "tilt_direction_deg_box": selected["tilt_direction_deg_box"],
            "normal_toward_camera": selected["normal_toward_camera"],
            "coefficients_box": selected["coefficients_box"],
            "circle_center_camera_mm": selected["circle_center_camera_mm"],
            "inner_radius_mm": selected["inner_radius_mm"],
            "circle_radial_residual_p90_mm": selected["circle_radial_residual_p90_mm"],
            "reprojection_chamfer_p90_px": selected["reprojection_chamfer_p90_px"],
            "predicted_inner_rim_uv": selected["predicted_inner_rim_uv"],
            "predicted_outer_rim_uv": selected["predicted_outer_rim_uv"],
            "dense_depth": selected["dense_depth"],
            "mouth_band_consistency": selected.get("depth_anchor"),
            "evidence_score": selected.get("evidence_score"),
        }
    timings["total_ms"] = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": "1.0",
        "stage": "M39.3.4_analytic_conic_pose_plus_dense_annulus_depth",
        "mode": ("online_production_surface_source" if bool(cfg.get("production_routing_enabled", False)) else "online_diagnostic_only"),
        "production_routing_enabled": bool(cfg.get("production_routing_enabled", False)),
        "status": "RESOLVED" if classification in {"FLAT", "TILTED"} else "UNCERTAIN",
        "classification": classification,
        "reason": reason,
        "mouth_conic": conic,
        "analytic_solver": analytic_info,
        "known_object_geometry": {
            "nominal_inner_diameter_mm": 2.0 * rin,
            "nominal_outer_diameter_mm": 2.0 * rout,
        },
        "prior_m39_3_1": {
            "state": prior_state,
            "confidence": prior.get("confidence"),
            "reason": prior.get("classification_reason") or prior.get("reason"),
            "hard_veto": False,
        },
        "candidate_count": int(len(candidates)),
        "usable_candidate_count": int(len(usable)),
        "winner_margin": winner_margin,
        "analytic_min_tilt_deg": None if analytic_min_tilt >= 900.0 else float(analytic_min_tilt),
        "mouth_off_axis_angle_deg": float(off_axis_angle_deg),
        "strong_flat_band": bool(strong_flat_band),
        "selected_candidate": selected_surface,
        "candidates": candidates,
        "timing_ms": timings,
        "thresholds": {
            "flat_tilt_max_deg": flat_max,
            "tilted_tilt_min_deg": tilt_min,
            "minimum_tilt_winner_margin": min_tilt_margin,
            "minimum_flat_winner_margin": min_flat_margin,
            "m3931_conflict_required_margin": conflict_margin,
            "m3931_supportive_required_margin": supportive_margin,
            "maximum_accepted_tilt_deg": physical_max_tilt,
        },
    }
