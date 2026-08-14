"""M39.3.2 semantic + topology + depth ring-prior surface reconstruction.

This module deliberately does not route production grasps.  It reconstructs
only the visible front annulus of a matched ``ring_mouth`` + ``foam_ring``
instance by using what is already known about the object:

* the mouth and foam-ring instance are semantically matched;
* the object is an annulus with known nominal inner/outer radius;
* along each angular sector, valid front-face depth must lie between the mouth
  boundary and the matched outer ring boundary;
* inner/outer depth-edge guards are ignored;
* a real front-face sample must form a locally supported radial plateau;
* sector representatives must jointly form the camera-nearest coherent annular
  plane with physically plausible ring scale and tilt.

No absolute box-floor depth is used to decide surface identity.  The calibrated
box axes are used only to express the reconstructed plane and measure tilt.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.asarray([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm], dtype=np.float64)


def _angular_coverage(sectors: Sequence[int], sector_count: int) -> float:
    values = sorted(set(int(s) % int(sector_count) for s in sectors))
    if not values:
        return 0.0
    if len(values) == 1:
        return 360.0 / float(sector_count)
    wrapped = values[1:] + [values[0] + int(sector_count)]
    max_gap = max(b - a for a, b in zip(values, wrapped))
    return float(max(0.0, 360.0 - max_gap * 360.0 / sector_count + 360.0 / sector_count))


def _ray_ring_run(
    mouth_mask: np.ndarray,
    ring_mask: np.ndarray,
    center_uv: Tuple[float, float],
    angle_rad: float,
    *,
    maximum_radius_px: int,
    maximum_mask_gap_px: int,
) -> Optional[Tuple[int, int, int]]:
    """Return mouth exit, first ring pixel, last contiguous matched-ring pixel."""
    mouth = np.asarray(mouth_mask, dtype=bool)
    ring = np.asarray(ring_mask, dtype=bool)
    h, w = ring.shape
    cx, cy = float(center_uv[0]), float(center_uv[1])
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    samples = []
    for radius in range(max(1, int(maximum_radius_px)) + 1):
        x = int(round(cx + ca * radius))
        y = int(round(cy + sa * radius))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
        samples.append((radius, bool(mouth[y, x]), bool(ring[y, x])))
    mouth_r = [r for r, is_mouth, _ in samples if is_mouth]
    if not mouth_r:
        return None
    mouth_exit = max(mouth_r)
    start = None
    for r, _is_mouth, is_ring in samples:
        if r <= mouth_exit:
            continue
        if is_ring:
            start = r
            break
        if r - mouth_exit > max(2, int(maximum_mask_gap_px) + 2):
            break
    if start is None:
        return None
    end = start
    gap = 0
    allowed_gap = max(1, int(maximum_mask_gap_px))
    for r, _is_mouth, is_ring in samples:
        if r < start:
            continue
        if is_ring:
            end = r
            gap = 0
        else:
            gap += 1
            if gap > allowed_gap:
                break
    if end <= start:
        return None
    return int(mouth_exit), int(start), int(end)


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
    candidates: list[Dict[str, Any]] = []
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
        radial_span = max(radial) - min(radial) if radial else 0.0
        if ray_support < minimum_rays or radial_span < minimum_radial_span_ratio:
            continue
        z2 = [float(row[0]) for row in supported]
        candidates.append({
            "depth_mm": float(np.median(z2)),
            "depth_mad_mm": float(np.median(np.abs(np.asarray(z2) - np.median(z2)))),
            "support_count": int(len(supported)),
            "ray_support_count": int(ray_support),
            "radial_span_ratio": float(radial_span),
            "representative_uv": [
                float(np.median([row[1] for row in supported])),
                float(np.median([row[2] for row in supported])),
            ],
        })
    candidates.sort(key=lambda row: float(row["depth_mm"]))
    return candidates[: max(1, int(maximum_candidates))]


def extract_mouth_anchored_sector_candidates(
    depth_mm: np.ndarray,
    ring_mask: np.ndarray,
    mouth_mask: np.ndarray,
    center_uv: Tuple[float, float],
    intrinsics: Mapping[str, float],
    *,
    object_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Extract topology-constrained local depth plateaus for every ring sector."""
    depth = np.asarray(depth_mm)
    ring = np.asarray(ring_mask, dtype=bool)
    mouth = np.asarray(mouth_mask, dtype=bool)
    if depth.shape != ring.shape or ring.shape != mouth.shape:
        raise ValueError("depth/ring/mouth shapes must match")

    sectors = max(8, _i(config.get("sector_count"), 16))
    wedge_half = max(1.0, _f(config.get("sector_wedge_half_angle_deg"), 8.0))
    wedge_rays = max(3, _i(config.get("sector_wedge_ray_count"), 9))
    maximum_radius = max(32, _i(config.get("maximum_radial_search_px"), 120))
    maximum_mask_gap = max(0, _i(config.get("maximum_ring_mask_gap_px"), _i(object_geometry.get("maximum_ring_mask_gap_px"), 2)))
    sample_start = float(np.clip(_f(config.get("sample_wall_start_ratio"), 0.12), 0.0, 0.9))
    sample_end = float(np.clip(_f(config.get("sample_wall_end_ratio"), 0.78), sample_start + 0.05, 1.0))
    min_wall_px = max(3.0, _f(config.get("minimum_visible_wall_width_px"), 6.0))
    min_depth = _f(config.get("minimum_depth_mm"), 150.0)
    max_depth = _f(config.get("maximum_depth_mm"), 3000.0)
    gap_mm = max(1.0, _f(config.get("plateau_depth_gap_mm"), 4.0))
    min_points = max(4, _i(config.get("plateau_minimum_points"), 6))
    min_rays = max(1, _i(config.get("plateau_minimum_rays"), 2))
    min_span = max(0.02, _f(config.get("plateau_minimum_radial_span_ratio"), 0.10))
    max_candidates = max(1, _i(config.get("maximum_candidates_per_sector"), 3))

    inner_radius_nominal = 0.5 * _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0)
    outer_radius_nominal = 0.5 * _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    fx_eff = math.sqrt(float(intrinsics["fx"]) * float(intrinsics["fy"]))

    rows: list[Dict[str, Any]] = []
    for sector in range(sectors):
        theta = (sector + 0.5) * 2.0 * math.pi / float(sectors)
        radial_samples: list[tuple[float, float, float, int, float]] = []
        runs = []
        for ray_index, delta_deg in enumerate(np.linspace(-wedge_half, wedge_half, wedge_rays)):
            angle = theta + math.radians(float(delta_deg))
            run = _ray_ring_run(
                mouth,
                ring,
                center_uv,
                angle,
                maximum_radius_px=maximum_radius,
                maximum_mask_gap_px=maximum_mask_gap,
            )
            if run is None:
                continue
            mouth_exit, start, end = run
            width = float(end - start + 1)
            if width < min_wall_px:
                continue
            runs.append((mouth_exit, start, end, width))
            ca, sa = math.cos(angle), math.sin(angle)
            sample_count = max(8, int(math.ceil(width * 2.0)))
            for t in np.linspace(sample_start, sample_end, sample_count):
                radius = float(start) + (width - 1.0) * float(t)
                x = int(round(float(center_uv[0]) + ca * radius))
                y = int(round(float(center_uv[1]) + sa * radius))
                if x < 0 or y < 0 or x >= depth.shape[1] or y >= depth.shape[0] or not ring[y, x]:
                    continue
                z = float(depth[y, x])
                if not np.isfinite(z) or z < min_depth or z > max_depth:
                    continue
                radial_samples.append((z, float(x), float(y), int(ray_index), float(t)))

        row: Dict[str, Any] = {
            "sector": int(sector),
            "sector_angle_deg_image": float((sector + 0.5) * 360.0 / sectors),
            "status": "MISSING",
            "ray_run_count": int(len(runs)),
            "raw_sample_count": int(len(radial_samples)),
            "candidates": [],
        }
        if runs:
            row["mouth_exit_radius_px"] = float(np.median([run[0] for run in runs]))
            row["ring_inner_radius_px"] = float(np.median([run[1] for run in runs]))
            row["ring_outer_radius_px"] = float(np.median([run[2] for run in runs]))
            row["visible_wall_width_px"] = float(np.median([run[3] for run in runs]))
        candidates = _cluster_plateaus(
            radial_samples,
            depth_gap_mm=gap_mm,
            minimum_points=min_points,
            minimum_rays=min_rays,
            minimum_radial_span_ratio=min_span,
            maximum_candidates=max_candidates,
        )
        inner_px = row.get("ring_inner_radius_px")
        outer_px = row.get("ring_outer_radius_px")
        for rank, candidate in enumerate(candidates):
            z = float(candidate["depth_mm"])
            size_error = None
            inner_est = None
            outer_est = None
            wall_est = None
            if inner_px is not None and outer_px is not None and fx_eff > 1e-9:
                inner_est = float(inner_px) * z / fx_eff
                outer_est = float(outer_px) * z / fx_eff
                wall_est = outer_est - inner_est
                size_error = (
                    abs(inner_est - inner_radius_nominal) / max(inner_radius_nominal, 1e-6)
                    + abs(outer_est - outer_radius_nominal) / max(outer_radius_nominal, 1e-6)
                )
            candidate.update({
                "front_rank": int(rank),
                "inner_radius_estimate_mm": inner_est,
                "outer_radius_estimate_mm": outer_est,
                "wall_thickness_estimate_mm": wall_est,
                "nominal_scale_error": size_error,
            })
        row["candidates"] = candidates
        if candidates:
            row["status"] = "CANDIDATES_AVAILABLE"
        rows.append(row)
    return rows


def _candidate_records(
    sector_rows: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, float],
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z_inside: np.ndarray,
    *,
    maximum_nominal_scale_error: float,
) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for row in sector_rows:
        sector = int(row["sector"])
        for index, candidate_raw in enumerate(row.get("candidates") or []):
            candidate = dict(candidate_raw)
            size_error = candidate.get("nominal_scale_error")
            if size_error is not None and float(size_error) > maximum_nominal_scale_error:
                continue
            uv = candidate.get("representative_uv") or []
            if len(uv) != 2:
                continue
            z = float(candidate["depth_mm"])
            point = _deproject(float(uv[0]), float(uv[1]), z, intrinsics)
            records.append({
                "sector": sector,
                "candidate_index": int(index),
                "candidate": candidate,
                "point_camera_mm": point,
                "x_box_mm": float(np.dot(point, box_x)),
                "y_box_mm": float(np.dot(point, box_y)),
                "h_box_mm": float(np.dot(point, box_z_inside)),
            })
    return records


def _fit_frontmost_coherent_plane(
    sector_rows: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, float],
    *,
    box_x: np.ndarray,
    box_y: np.ndarray,
    box_z_inside: np.ndarray,
    object_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    sector_count = max(8, _i(config.get("sector_count"), 16))
    residual_threshold = max(1.0, _f(config.get("surface_plane_residual_mm"), 5.0))
    minimum_sectors = max(4, _i(config.get("minimum_surface_sectors"), 5))
    minimum_coverage = max(45.0, _f(config.get("minimum_surface_coverage_deg"), 135.0))
    max_tilt = max(5.0, _f(config.get("maximum_reconstruction_tilt_deg"), 35.0))
    max_size_error = max(0.05, _f(config.get("maximum_nominal_scale_error"), 0.45))
    preferred_size_error = max(0.02, _f(config.get("preferred_nominal_scale_error"), 0.25))
    front_window = max(1.0, _f(config.get("front_surface_depth_window_mm"), 15.0))
    records = _candidate_records(
        sector_rows,
        intrinsics,
        box_x,
        box_y,
        box_z_inside,
        maximum_nominal_scale_error=max_size_error,
    )
    if len(records) < minimum_sectors:
        return None

    by_sector: Dict[int, list[Dict[str, Any]]] = {}
    for record in records:
        by_sector.setdefault(int(record["sector"]), []).append(record)

    models: list[Dict[str, Any]] = []
    for indexes in combinations(range(len(records)), 3):
        seed = [records[index] for index in indexes]
        if len({int(row["sector"]) for row in seed}) < 3:
            continue
        matrix = np.asarray([[row["x_box_mm"], row["y_box_mm"], 1.0] for row in seed], dtype=np.float64)
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            continue
        heights = np.asarray([row["h_box_mm"] for row in seed], dtype=np.float64)
        coefficients = np.linalg.solve(matrix, heights)
        tilt = math.degrees(math.atan(math.hypot(float(coefficients[0]), float(coefficients[1]))))
        if tilt > max_tilt:
            continue

        chosen: list[tuple[float, Dict[str, Any]]] = []
        for sector, candidates in by_sector.items():
            best = None
            for record in candidates:
                predicted = (
                    float(coefficients[0]) * float(record["x_box_mm"])
                    + float(coefficients[1]) * float(record["y_box_mm"])
                    + float(coefficients[2])
                )
                residual = abs(float(record["h_box_mm"]) - predicted)
                if best is None or residual < best[0]:
                    best = (residual, record)
            if best is not None and best[0] <= residual_threshold:
                chosen.append(best)
        sectors = [int(row[1]["sector"]) for row in chosen]
        if len(chosen) < minimum_sectors or _angular_coverage(sectors, sector_count) < minimum_coverage:
            continue

        design = np.asarray([[row[1]["x_box_mm"], row[1]["y_box_mm"], 1.0] for row in chosen], dtype=np.float64)
        h = np.asarray([row[1]["h_box_mm"] for row in chosen], dtype=np.float64)
        coefficients = np.linalg.lstsq(design, h, rcond=None)[0]
        residuals = np.abs(h - design @ coefficients)
        keep = residuals <= residual_threshold
        if int(np.count_nonzero(keep)) >= minimum_sectors:
            design = design[keep]
            h = h[keep]
            chosen = [row for row, flag in zip(chosen, keep.tolist()) if flag]
            coefficients = np.linalg.lstsq(design, h, rcond=None)[0]
            residuals = np.abs(h - design @ coefficients)
        sectors = [int(row[1]["sector"]) for row in chosen]
        coverage = _angular_coverage(sectors, sector_count)
        if len(chosen) < minimum_sectors or coverage < minimum_coverage:
            continue
        gx, gy = float(coefficients[0]), float(coefficients[1])
        tilt = math.degrees(math.atan(math.hypot(gx, gy)))
        if tilt > max_tilt:
            continue
        depths = [float(row[1]["candidate"]["depth_mm"]) for row in chosen]
        size_errors = [
            float(row[1]["candidate"]["nominal_scale_error"])
            for row in chosen
            if row[1]["candidate"].get("nominal_scale_error") is not None
        ]
        support = sum(int(row[1]["candidate"].get("support_count") or 0) for row in chosen)
        models.append({
            "coefficients": coefficients,
            "chosen": chosen,
            "sector_count": int(len(chosen)),
            "sectors": sorted(sectors),
            "angular_coverage_deg": float(coverage),
            "residual_median_mm": float(np.median(residuals)),
            "residual_p90_mm": float(np.percentile(residuals, 90)),
            "median_depth_mm": float(np.median(depths)),
            "median_nominal_scale_error": float(np.median(size_errors)) if size_errors else None,
            "preferred_scale_sector_count": int(sum(err <= preferred_size_error for err in size_errors)),
            "support_count": int(support),
            "tilt_deg": float(tilt),
        })
    if not models:
        return None

    # Semantic visibility prior: the target's visible front annulus must be the
    # camera-nearest coherent annular surface.  A floor or lower ring may be a
    # smoother/larger plane, but it cannot occlude a valid nearer matched-ring
    # front surface.  Allow a small depth window for noise, then rank by support.
    front_depth = min(float(model["median_depth_mm"]) for model in models)
    front_models = [model for model in models if float(model["median_depth_mm"]) <= front_depth + front_window]
    best = max(
        front_models,
        key=lambda model: (
            int(model["sector_count"]),
            float(model["angular_coverage_deg"]),
            int(model["preferred_scale_sector_count"]),
            -float(model["median_nominal_scale_error"] if model["median_nominal_scale_error"] is not None else 99.0),
            -float(model["residual_median_mm"]),
            int(model["support_count"]),
        ),
    )

    coefficients = np.asarray(best["coefficients"], dtype=np.float64)
    gx, gy = float(coefficients[0]), float(coefficients[1])
    normal_into_box = _norm(box_z_inside - gx * box_x - gy * box_y)
    normal_toward_camera = -normal_into_box
    direction_box = float((math.degrees(math.atan2(gy, gx)) + 360.0) % 360.0)
    outer_diameter = _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
    predicted_p2p = float(outer_diameter * math.tan(math.radians(float(best["tilt_deg"]))))

    selected_samples = []
    selected_map = {int(record[1]["sector"]): record[1] for record in best["chosen"]}
    for row in sector_rows:
        item = dict(row)
        selected = selected_map.get(int(row["sector"]))
        if selected is not None:
            candidate = dict(selected["candidate"])
            item["status"] = "SELECTED_FRONT_SURFACE"
            item["selected_candidate"] = candidate
            item["point_camera_mm"] = [float(v) for v in selected["point_camera_mm"].tolist()]
            item["x_box_mm"] = float(selected["x_box_mm"])
            item["y_box_mm"] = float(selected["y_box_mm"])
            item["h_box_mm"] = float(selected["h_box_mm"])
        elif row.get("candidates"):
            item["status"] = "REJECTED_NON_FRONT_OR_INCOHERENT"
        selected_samples.append(item)

    return {
        "model": "h_box=h0+gx*x_box+gy*y_box",
        "h0_mm": float(coefficients[2]),
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
        "median_nominal_scale_error": best["median_nominal_scale_error"],
        "preferred_scale_sector_count": int(best["preferred_scale_sector_count"]),
        "support_count": int(best["support_count"]),
        "frontmost_model_depth_mm": float(front_depth),
        "sector_samples": selected_samples,
    }


def _jackknife(surface: Mapping[str, Any]) -> Dict[str, Any]:
    samples = [row for row in surface.get("sector_samples") or [] if row.get("status") == "SELECTED_FRONT_SURFACE"]
    if len(samples) < 6:
        return {"available": False, "tilt_std_deg": None, "direction_resultant": None}
    # The selected plane is already robustly fitted.  Refit in box coordinates
    # after removing each sector to expose one-sector domination.
    tilts, directions = [], []
    for omitted in range(len(samples)):
        subset = [row for index, row in enumerate(samples) if index != omitted]
        A = np.asarray([[float(row["x_box_mm"]), float(row["y_box_mm"]), 1.0] for row in subset], dtype=np.float64)
        h = np.asarray([float(row["h_box_mm"]) for row in subset], dtype=np.float64)
        if len(subset) < 3:
            continue
        c = np.linalg.lstsq(A, h, rcond=None)[0]
        gx, gy = float(c[0]), float(c[1])
        tilts.append(math.degrees(math.atan(math.hypot(gx, gy))))
        directions.append(math.atan2(gy, gx))
    if not tilts:
        return {"available": False, "tilt_std_deg": None, "direction_resultant": None}
    sx = float(np.mean(np.cos(directions)))
    sy = float(np.mean(np.sin(directions)))
    return {
        "available": True,
        "tilt_std_deg": float(np.std(np.asarray(tilts, dtype=np.float64))),
        "tilt_min_deg": float(np.min(tilts)),
        "tilt_max_deg": float(np.max(tilts)),
        "direction_resultant": float(math.hypot(sx, sy)),
    }


def reconstruct_ring_prior_surface(
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
) -> Dict[str, Any]:
    cfg = dict(config or {})
    x_axis = _norm(box_x_camera)
    y_axis = _norm(box_y_camera)
    z_inside = _norm(box_z_inside_camera)
    sectors = extract_mouth_anchored_sector_candidates(
        depth_mm,
        ring_mask,
        mouth_mask,
        center_uv,
        intrinsics,
        object_geometry=object_geometry,
        config=cfg,
    )
    surface = _fit_frontmost_coherent_plane(
        sectors,
        intrinsics,
        box_x=x_axis,
        box_y=y_axis,
        box_z_inside=z_inside,
        object_geometry=object_geometry,
        config=cfg,
    )
    if surface is None:
        return {
            "schema_version": "1.0",
            "stage": "M39.3.2_ring_prior_surface_reconstruction",
            "mode": "diagnostic_only",
            "production_routing_enabled": False,
            "status": "UNCERTAIN",
            "classification": "UNCERTAIN",
            "reason": "no_coherent_mouth_anchored_front_surface",
            "sector_samples": sectors,
            "surface": None,
        }

    jackknife = _jackknife(surface)
    tilt = float(surface["tilt_deg"])
    selected_count = int(surface["selected_sector_count"])
    coverage = float(surface["angular_coverage_deg"])
    residual = float(surface["residual_median_mm"])
    p2p = float(surface["predicted_outer_peak_to_peak_mm"])
    min_class_sectors = max(5, _i(cfg.get("classification_minimum_sectors"), 7))
    min_class_coverage = _f(cfg.get("classification_minimum_coverage_deg"), 180.0)
    max_class_residual = _f(cfg.get("classification_maximum_residual_mm"), 4.0)
    max_jackknife_std = _f(cfg.get("classification_maximum_jackknife_tilt_std_deg"), 5.0)
    max_median_scale_error = _f(cfg.get("classification_maximum_median_scale_error"), 0.30)
    min_preferred_scale_sectors = max(0, _i(cfg.get("classification_minimum_preferred_scale_sectors"), 3))
    flat_max = _f(cfg.get("flat_tilt_max_deg"), 8.0)
    tilted_min = _f(cfg.get("tilted_tilt_min_deg"), 10.0)
    physical_max_p2p = _f(
        cfg.get("maximum_physical_peak_to_peak_mm"),
        _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0)
        * math.tan(math.radians(_f(cfg.get("maximum_reconstruction_tilt_deg"), 35.0)))
        + 8.0,
    )
    stable = bool(
        selected_count >= min_class_sectors
        and coverage >= min_class_coverage
        and residual <= max_class_residual
        and p2p <= physical_max_p2p
        and (surface.get("median_nominal_scale_error") is None or float(surface.get("median_nominal_scale_error")) <= max_median_scale_error)
        and int(surface.get("preferred_scale_sector_count") or 0) >= min_preferred_scale_sectors
        and (
            not jackknife.get("available")
            or float(jackknife.get("tilt_std_deg") or 0.0) <= max_jackknife_std
        )
    )
    if not stable:
        classification = "UNCERTAIN"
        reason = "ring_prior_surface_not_stable_enough_for_tilt_classification"
    elif tilt <= flat_max:
        classification = "FLAT"
        reason = "ring_prior_surface_near_floor_parallel"
    elif tilt >= tilted_min:
        classification = "TILTED"
        reason = "ring_prior_surface_coherent_tilt"
    else:
        classification = "UNCERTAIN"
        reason = "ring_prior_surface_tilt_in_transition_band"

    return {
        "schema_version": "1.0",
        "stage": "M39.3.2_ring_prior_surface_reconstruction",
        "mode": "diagnostic_only",
        "production_routing_enabled": False,
        "status": "RECONSTRUCTED" if stable else "RECONSTRUCTED_LOW_CONFIDENCE",
        "classification": classification,
        "reason": reason,
        "semantic_anchor": "matched_ring_mouth_plus_foam_ring_instance",
        "surface_identity_policy": "camera_nearest_coherent_topology_constrained_annular_surface",
        "uses_absolute_box_floor_depth_for_identity": False,
        "known_object_geometry": {
            "nominal_inner_diameter_mm": _f(object_geometry.get("nominal_inner_diameter_mm"), 60.0),
            "nominal_outer_diameter_mm": _f(object_geometry.get("nominal_outer_diameter_mm"), 85.0),
            "nominal_wall_thickness_mm": _f(object_geometry.get("nominal_wall_thickness_mm"), 14.0),
            "axial_length_mm": _f(object_geometry.get("axial_length_mm"), 70.0),
        },
        "candidate_sector_count": int(sum(bool(row.get("candidates")) for row in sectors)),
        "jackknife": jackknife,
        "surface": surface,
        "thresholds": {
            "classification_minimum_sectors": min_class_sectors,
            "classification_minimum_coverage_deg": min_class_coverage,
            "classification_maximum_residual_mm": max_class_residual,
            "classification_maximum_jackknife_tilt_std_deg": max_jackknife_std,
            "classification_maximum_median_scale_error": max_median_scale_error,
            "classification_minimum_preferred_scale_sectors": min_preferred_scale_sectors,
            "flat_tilt_max_deg": flat_max,
            "tilted_tilt_min_deg": tilted_min,
            "maximum_physical_peak_to_peak_mm": physical_max_p2p,
        },
    }
