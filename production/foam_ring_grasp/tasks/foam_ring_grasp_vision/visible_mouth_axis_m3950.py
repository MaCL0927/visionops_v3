"""M39.5.2.2 Visible-Mouth Signed 3D Axis + Camera-Near-Rim routing support.

Axis/shape stage used by M39.5.1 production routing.  This module itself does
not mutate ``robot_candidate``; the dedicated M39.5.1 production stage consumes
its output.  The stage decouples semantic mouth visibility from robot READY pose:

* UPRIGHT_VISIBLE      -> mouth visible and cylinder axis near box +Z/-Z;
* TILTED_VISIBLE_SIDE  -> mouth visible but cylinder axis clearly non-vertical;
* PURE_SIDE            -> no usable mouth or projected mouth is almost edge-on;
* UNCERTAIN            -> evidence is insufficient / transition band.

For a tilted visible mouth, the known-circle conic produces the usual A/B 3-D
normal ambiguity.  M39.5.0 resolves the *signed* axis with the semantic image
vector ``foam_ring centroid -> ring_mouth centroid``.  Exact depth remains a
quality/consistency signal inside the reused M39.3.4 conic reconstruction; it no
longer owns the sign decision.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.analytic_conic_surface import (
    reconstruct_analytic_conic_surface,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    _associate_ring_mouths_detailed,
    _box_reference_axes_camera,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _f(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _norm(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return a / n


def _json_vec(v: np.ndarray | Sequence[float] | None) -> Optional[list[float]]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)):
        return None
    return [float(x) for x in a.tolist()]


def _mask_axis_ratio(mask: np.ndarray) -> Optional[float]:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (_center, axes, _angle) = cv2.fitEllipse(contour)
    major = max(float(axes[0]), float(axes[1]))
    minor = min(float(axes[0]), float(axes[1]))
    if major <= 1e-6:
        return None
    return float(minor / major)


def _project(point_camera_mm: Sequence[float], intrinsics: Mapping[str, float]) -> Optional[np.ndarray]:
    p = np.asarray(point_camera_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)) or p[2] <= 1e-6:
        return None
    fx = _f(intrinsics.get("fx"), 0.0)
    fy = _f(intrinsics.get("fy"), 0.0)
    cx = _f(intrinsics.get("cx"), 0.0)
    cy = _f(intrinsics.get("cy"), 0.0)
    if fx <= 0.0 or fy <= 0.0:
        return None
    return np.asarray([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy], dtype=np.float64)


def _angle2d_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    aa = np.asarray(a, dtype=np.float64).reshape(2)
    bb = np.asarray(b, dtype=np.float64).reshape(2)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-6 or nb <= 1e-6:
        return None
    c = float(np.dot(aa / na, bb / nb))
    return float(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))


def _angle3d_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    aa, bb = _norm(a), _norm(b)
    if not np.any(aa) or not np.any(bb):
        return None
    return float(math.degrees(math.acos(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))))


def _find_scene_item(scene: Mapping[str, Any], ring_id: int, mouth_id: Optional[int]) -> Optional[Mapping[str, Any]]:
    for item in scene.get("instances") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            if int(item.get("ring_instance_id")) != int(ring_id):
                continue
        except Exception:
            continue
        if mouth_id is not None and item.get("mouth_instance_id") is not None:
            try:
                if int(item.get("mouth_instance_id")) != int(mouth_id):
                    continue
            except Exception:
                continue
        return item
    return None


def _select_ring_and_mouth(
    scene: Mapping[str, Any],
    instances: Sequence[SegmentationInstance],
    geometry_config: GeometryConfig,
) -> Tuple[Optional[SegmentationInstance], Optional[SegmentationInstance], Dict[str, Any]]:
    rings = [x for x in instances if str(x.class_name) == "foam_ring"]
    mouths = [x for x in instances if str(x.class_name) == "ring_mouth"]
    if not rings:
        return None, None, {"reason": "no_foam_ring"}

    by_id = {int(x.instance_id): x for x in instances}
    selected_id = scene.get("selected_ring_instance_id")
    selected_ring = None
    try:
        candidate = by_id.get(int(selected_id)) if selected_id is not None else None
        if candidate is not None and str(candidate.class_name) == "foam_ring":
            selected_ring = candidate
    except Exception:
        selected_ring = None
    if selected_ring is None:
        selected_ring = max(rings, key=lambda x: (float(x.confidence), int(x.area_px)))

    matches, _unmatched_rings, _unmatched_mouths, assoc_debug = _associate_ring_mouths_detailed(
        rings, mouths, geometry_config
    )
    selected_mouth = None
    metrics: Dict[str, Any] = {}
    for ring, mouth, assoc in matches:
        if int(ring.instance_id) == int(selected_ring.instance_id):
            selected_mouth = mouth
            metrics = dict(assoc)
            break
    return selected_ring, selected_mouth, {
        "association": metrics,
        "association_debug": assoc_debug,
        "selected_ring_instance_id": int(selected_ring.instance_id),
        "selected_mouth_instance_id": int(selected_mouth.instance_id) if selected_mouth is not None else None,
    }


def _upright_opening_center(
    scene: Mapping[str, Any],
    ring_id: int,
    mouth_id: int,
) -> Optional[np.ndarray]:
    item = _find_scene_item(scene, ring_id, mouth_id)
    if item is None:
        return None
    for path in (
        ("ring_center_camera_mm",),
        ("plane", "centroid_camera_mm"),
        ("pose", "centroid_camera_mm"),
    ):
        value: Any = item
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
            a = np.asarray(value, dtype=np.float64)
            if np.all(np.isfinite(a)) and a[2] > 1.0:
                return a
    return None


def _projected_circle_axis_ratio(
    center: Optional[np.ndarray],
    normal: np.ndarray,
    radius_mm: float,
    intrinsics: Mapping[str, float],
) -> Optional[float]:
    """Expected image ellipse ratio for a known 3-D circle orientation.

    This is intentionally geometry-only: no dense depth scoring is required.
    M39.5.1 uses it to compare the observed mouth ellipse with the calibrated
    *flat* (box-Z) reference at the same image location.
    """
    if center is None:
        return None
    c = np.asarray(center, dtype=np.float64).reshape(3)
    n = _norm(normal)
    if not np.any(n) or not np.all(np.isfinite(c)) or c[2] <= 1.0:
        return None
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, n))) > 0.85:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = _norm(np.cross(n, reference))
    v = _norm(np.cross(n, u))
    if not np.any(u) or not np.any(v):
        return None
    uv_rows = []
    for theta in np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False):
        point = c + float(radius_mm) * (math.cos(theta) * u + math.sin(theta) * v)
        uv = _project(point, intrinsics)
        if uv is not None and np.all(np.isfinite(uv)):
            uv_rows.append(uv)
    if len(uv_rows) < 8:
        return None
    pts = np.asarray(uv_rows, dtype=np.float32).reshape(-1, 1, 2)
    (_center, axes, _angle) = cv2.fitEllipse(pts)
    major = max(float(axes[0]), float(axes[1]))
    minor = min(float(axes[0]), float(axes[1]))
    if major <= 1e-6:
        return None
    return float(minor / major)


def _camera_near_rim_geometry(
    center: Optional[np.ndarray],
    axis_out: Optional[np.ndarray],
    inner_radius_mm: float,
    outer_radius_mm: float,
) -> Dict[str, Any]:
    """Point on the wall radial midline that is closest to the camera.

    The user's intended "upper arc" is not image-up.  It is the camera-facing
    portion of the opening rim.  For a circle in the opening plane, the exact
    Euclidean closest direction is the centre->camera vector projected into the
    opening plane.
    """
    if center is None or axis_out is None:
        return {"available": False, "reason": "opening_center_or_axis_unavailable"}
    c = np.asarray(center, dtype=np.float64).reshape(3)
    axis = _norm(axis_out)
    if not np.any(axis) or not np.all(np.isfinite(c)):
        return {"available": False, "reason": "axis_or_center_invalid"}
    toward_camera = -c
    radial = toward_camera - float(np.dot(toward_camera, axis)) * axis
    radial = _norm(radial)
    if not np.any(radial):
        return {"available": False, "reason": "camera_direction_parallel_to_opening_axis"}
    contact_radius = 0.5 * (float(inner_radius_mm) + float(outer_radius_mm))
    point = c + contact_radius * radial
    return {
        "available": True,
        "definition": "opening_plane_projection_of_center_to_camera_at_wall_radial_midline",
        "camera_near_radial_direction_camera": _json_vec(radial),
        "contact_radius_mm": float(contact_radius),
        "camera_near_rim_midpoint_camera_mm": _json_vec(point),
        "opening_center_camera_mm": _json_vec(c),
        "axis_out_of_opening_camera": _json_vec(axis),
        "axis_into_opening_camera": _json_vec(-axis),
    }


def _vector_angle_image_deg(v: np.ndarray) -> Optional[float]:
    a = np.asarray(v, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(a)) <= 1e-6:
        return None
    return float((math.degrees(math.atan2(float(a[1]), float(a[0]))) + 360.0) % 360.0)


def _circular_angle_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))

def attach_m3950_visible_mouth_axis_validation(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    raw_config: Mapping[str, Any],
    geometry_config: GeometryConfig,
) -> Dict[str, Any]:
    """Attach the M39.5.x visible-mouth shape/axis result.

    M39.5.1 changes two semantics compared with M39.5.0:
      1. UPRIGHT is no longer inferred merely from a circular/concentric mouth.
         The observed ellipse is compared with the calibrated box-Z flat-circle
         projection.  A small transition band is arbitrated with ring-depth
         sector-gradient evidence.
      2. The red rim target is camera-nearest, not image-up.

    This function still does not mutate ``robot_candidate``.  M39.5.1's
    dedicated production stage consumes this result after the legacy stages.
    """

    cfg = raw_config.get("m39_5_0_visible_mouth_axis_validation") or {}
    cfg = cfg if isinstance(cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", True))
    result: Dict[str, Any] = {
        "schema_version": "1.1",
        "stage": "M39.5.1_visible_mouth_signed_3d_axis_camera_near_rim",
        "mode": "production_axis_source",
        "production_routing_enabled": bool((raw_config.get("m39_5_1_tilted_visible_grasp") or {}).get("enabled", False)),
        "enabled": enabled,
        "classification": "UNCERTAIN",
        "recommended_ready_pose": "NONE",
        "reason": "disabled" if not enabled else "not_evaluated",
        "axis_solution_reliable": False,
        "selected_ring_instance_id": None,
        "selected_mouth_instance_id": None,
    }
    if not enabled:
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    axes = _box_reference_axes_camera(geometry_config)
    if axes is None:
        result.update(reason="calibrated_box_reference_unavailable")
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result
    box_x, box_y_down, box_z_inside = axes

    ring, mouth, association = _select_ring_and_mouth(scene, instances, geometry_config)
    result["association"] = association
    if ring is None:
        result.update(classification="UNCERTAIN", reason="no_foam_ring")
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result
    rid = int(ring.instance_id)
    result["selected_ring_instance_id"] = rid

    if mouth is None:
        result.update(
            classification="PURE_SIDE",
            recommended_ready_pose="SIDE_INITIAL",
            reason="no_matched_ring_mouth",
            mouth_visible=False,
            signed_axis_source="m39_4_pure_side_recovery_required",
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result
    mid = int(mouth.instance_id)
    result["selected_mouth_instance_id"] = mid
    result["mouth_visible"] = True

    ring_center = np.asarray(ring.centroid_uv, dtype=np.float64)
    mouth_center = np.asarray(mouth.centroid_uv, dtype=np.float64)
    semantic_vec = mouth_center - ring_center
    semantic_offset_px = float(np.linalg.norm(semantic_vec))
    x1, y1, x2, y2 = ring.bbox_xyxy
    ring_major_px = max(1.0, float(max(x2 - x1, y2 - y1)))
    semantic_ratio = semantic_offset_px / ring_major_px
    mouth_axis_ratio = _mask_axis_ratio(mouth.mask)

    scene_item = _find_scene_item(scene, rid, mid)
    prior = None
    if isinstance(scene_item, Mapping):
        branch = scene_item.get("m38_branch_a") if isinstance(scene_item.get("m38_branch_a"), Mapping) else {}
        p = branch.get("m39_3_1_tilt_evidence") if isinstance(branch, Mapping) else None
        prior = p if isinstance(p, Mapping) else None
    prior = prior or {}
    sector = prior.get("sector_gradient") if isinstance(prior.get("sector_gradient"), Mapping) else {}
    sector_tilt = _f(sector.get("sector_gradient_tilt_deg"), -1.0)
    sector_peak_to_peak = _f(sector.get("predicted_peak_to_peak_mm"), -1.0)
    sector_gradient_direction = sector.get("gradient_direction_deg_image")
    try:
        sector_gradient_direction = float(sector_gradient_direction) if sector_gradient_direction is not None else None
    except Exception:
        sector_gradient_direction = None

    result["semantic_axis_2d"] = {
        "ring_centroid_uv": [float(v) for v in ring_center.tolist()],
        "mouth_centroid_uv": [float(v) for v in mouth_center.tolist()],
        "body_to_mouth_vector_uv": [float(v) for v in semantic_vec.tolist()],
        "offset_px": semantic_offset_px,
        "ring_major_px": ring_major_px,
        "offset_ratio": float(semantic_ratio),
        "mouth_minor_major_ratio": mouth_axis_ratio,
    }
    result["prior_tilt_evidence"] = {
        "state": prior.get("state"),
        "confidence": prior.get("confidence"),
        "sector_gradient_tilt_deg": None if sector_tilt < 0 else float(sector_tilt),
        "sector_peak_to_peak_mm": None if sector_peak_to_peak < 0 else float(sector_peak_to_peak),
        "sector_gradient_direction_deg_image": sector_gradient_direction,
        "valid_sector_count": sector.get("valid_sector_count"),
        "inlier_sector_count": sector.get("inlier_sector_count"),
    }

    pure_side_ratio_max = _f(cfg.get("pure_side_mouth_axis_ratio_max"), 0.34)
    flat_upright_max = _f(cfg.get("flat_reference_axis_ratio_deficit_upright_max"), 0.070)
    flat_tilted_min = _f(cfg.get("flat_reference_axis_ratio_deficit_tilted_min"), 0.090)
    transition_sector_tilt_min = _f(cfg.get("transition_sector_tilt_min_deg"), 10.0)
    transition_peak_min = _f(cfg.get("transition_peak_to_peak_min_mm"), 14.0)
    transition_sector_flat_max = _f(cfg.get("transition_sector_flat_max_deg"), 8.0)
    transition_peak_flat_max = _f(cfg.get("transition_peak_to_peak_flat_max_mm"), 12.0)
    side_ready_tilt_min = _f(cfg.get("side_ready_axis_tilt_min_deg"), 30.0)
    min_semantic_px = _f(cfg.get("minimum_semantic_offset_px_for_sign"), 6.0)
    max_align = _f(cfg.get("maximum_semantic_conic_alignment_deg"), 30.0)
    min_semantic_branch_sep = _f(cfg.get("minimum_semantic_branch_separation_deg"), 60.0)
    max_gradient_align = _f(cfg.get("maximum_sector_gradient_alignment_deg"), 85.0)
    min_gradient_branch_sep = _f(cfg.get("minimum_sector_gradient_branch_separation_deg"), 25.0)
    min_evidence_margin = _f(cfg.get("minimum_conic_evidence_margin_for_axis"), 0.020)
    max_reproj = _f(cfg.get("maximum_axis_reprojection_chamfer_p90_px"), 5.5)
    max_radial = _f(cfg.get("maximum_axis_circle_radial_residual_p90_mm"), 4.0)
    min_band_sectors = int(_f(cfg.get("minimum_axis_mouth_band_valid_sectors"), 10.0))
    axis_probe_mm = _f(cfg.get("axis_projection_probe_mm"), 60.0)

    object_cfg = raw_config.get("object_geometry") or {}
    inner_radius = 0.5 * _f(object_cfg.get("nominal_inner_diameter_mm"), 60.0)
    outer_radius = 0.5 * _f(object_cfg.get("nominal_outer_diameter_mm"), 85.0)

    threshold_doc = {
        "pure_side_mouth_axis_ratio_max": pure_side_ratio_max,
        "flat_reference_axis_ratio_deficit_upright_max": flat_upright_max,
        "flat_reference_axis_ratio_deficit_tilted_min": flat_tilted_min,
        "transition_sector_tilt_min_deg": transition_sector_tilt_min,
        "transition_peak_to_peak_min_mm": transition_peak_min,
        "transition_sector_flat_max_deg": transition_sector_flat_max,
        "transition_peak_to_peak_flat_max_mm": transition_peak_flat_max,
        "side_ready_axis_tilt_min_deg": side_ready_tilt_min,
        "maximum_semantic_conic_alignment_deg": max_align,
        "minimum_semantic_offset_px_for_sign": min_semantic_px,
        "maximum_sector_gradient_alignment_deg": max_gradient_align,
        "minimum_sector_gradient_branch_separation_deg": min_gradient_branch_sep,
        "minimum_conic_evidence_margin_for_axis": min_evidence_margin,
    }

    if mouth_axis_ratio is not None and mouth_axis_ratio <= pure_side_ratio_max:
        result.update(
            classification="PURE_SIDE",
            recommended_ready_pose="SIDE_INITIAL",
            reason="matched_mouth_is_edge_on_pseudo_mouth",
            signed_axis_source="m39_4_pure_side_recovery_required",
            thresholds=threshold_doc,
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    reference_center = _upright_opening_center(scene, rid, mid)
    expected_flat_ratio = _projected_circle_axis_ratio(
        reference_center,
        -_norm(box_z_inside),
        inner_radius,
        intrinsics,
    )
    flat_deficit = None
    if expected_flat_ratio is not None and mouth_axis_ratio is not None:
        flat_deficit = float(expected_flat_ratio - mouth_axis_ratio)
    shape_state = "TRANSITION"
    shape_reason = "flat_reference_projection_unavailable"
    if flat_deficit is not None:
        if flat_deficit <= flat_upright_max:
            shape_state = "UPRIGHT"
            shape_reason = "observed_mouth_matches_calibrated_flat_projection"
        elif flat_deficit >= flat_tilted_min:
            shape_state = "TILTED"
            shape_reason = "observed_mouth_deviates_from_calibrated_flat_projection"
        else:
            tilted_transition = bool(
                (sector_tilt >= transition_sector_tilt_min)
                or (sector_peak_to_peak >= transition_peak_min)
            )
            upright_transition = bool(
                str(prior.get("state") or "").upper() == "FLAT"
                and 0.0 <= sector_tilt <= transition_sector_flat_max
                and 0.0 <= sector_peak_to_peak <= transition_peak_flat_max
            )
            if tilted_transition:
                shape_state = "TILTED"
                shape_reason = "flat_shape_transition_resolved_tilted_by_sector_depth_gradient"
            elif upright_transition:
                shape_state = "UPRIGHT"
                shape_reason = "flat_shape_transition_resolved_upright_by_sector_depth_gradient"
            else:
                shape_state = "TRANSITION"
                shape_reason = "flat_shape_transition_requires_conic_axis"
    result["flat_reference_shape_test"] = {
        "available": bool(expected_flat_ratio is not None and mouth_axis_ratio is not None),
        "observed_mouth_axis_ratio": mouth_axis_ratio,
        "expected_flat_axis_ratio": expected_flat_ratio,
        "axis_ratio_deficit_expected_minus_observed": flat_deficit,
        "shape_state": shape_state,
        "reason": shape_reason,
    }

    if shape_state == "UPRIGHT":
        axis_out = -_norm(box_z_inside)
        center = reference_center
        near = _camera_near_rim_geometry(center, axis_out, inner_radius, outer_radius)
        result.update(
            classification="UPRIGHT_VISIBLE",
            recommended_ready_pose="VISIBLE_INITIAL",
            production_grasp_policy="VISIBLE_CLOCK3",
            geometric_pose_classification="UPRIGHT",
            reason=shape_reason,
            axis_solution_reliable=True,
            signed_axis_source="calibrated_box_z_flat_reference",
            axis_out_of_opening_camera=_json_vec(axis_out),
            axis_into_opening_camera=_json_vec(-axis_out),
            axis_tilt_from_box_z_deg=0.0,
            opening_center_camera_mm=_json_vec(center),
            camera_near_rim=near,
            thresholds=threshold_doc,
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    # TILTED or transition: reconstruct both analytic conic branches.  Unlike
    # M39.5.0, a short body->mouth semantic vector does not make the *shape*
    # uncertain.  It only changes how A/B is disambiguated.
    conic_cfg = raw_config.get("m39_3_4_analytic_conic_surface") or {}
    conic_cfg = dict(conic_cfg) if isinstance(conic_cfg, Mapping) else {}
    conic_cfg["production_routing_enabled"] = False
    try:
        conic = reconstruct_analytic_conic_surface(
            depth_mm,
            np.asarray(ring.mask, dtype=bool),
            np.asarray(mouth.mask, dtype=bool),
            (float(mouth_center[0]), float(mouth_center[1])),
            intrinsics,
            box_x_camera=box_x,
            box_y_camera=box_y_down,
            box_z_inside_camera=box_z_inside,
            object_geometry=object_cfg,
            config=conic_cfg,
            prior_tilt_evidence=prior,
        )
    except Exception as exc:
        result.update(
            classification="UNCERTAIN",
            recommended_ready_pose="NONE",
            production_grasp_policy="NONE",
            geometric_pose_classification=("TILTED_SHAPE_AXIS_UNRESOLVED" if shape_state == "TILTED" else "TRANSITION_AXIS_UNRESOLVED"),
            reason=f"conic_reconstruction_error:{type(exc).__name__}:{exc}",
            axis_solution_reliable=False,
            thresholds=threshold_doc,
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    conic_rows = []
    for cand in conic.get("candidates") or []:
        if not isinstance(cand, Mapping) or not str(cand.get("label") or "").startswith("CONIC_"):
            continue
        center_raw = cand.get("circle_center_camera_mm")
        axis_raw = cand.get("normal_toward_camera")
        if center_raw is None or axis_raw is None:
            continue
        center = np.asarray(center_raw, dtype=np.float64).reshape(3)
        axis = _norm(np.asarray(axis_raw, dtype=np.float64).reshape(3))
        if not np.any(axis):
            continue
        uv0 = _project(center, intrinsics)
        uv1 = _project(center + axis_probe_mm * axis, intrinsics)
        projected = None if uv0 is None or uv1 is None else (uv1 - uv0)
        semantic_alignment = _angle2d_deg(semantic_vec, projected) if projected is not None else None
        projected_angle = _vector_angle_image_deg(projected) if projected is not None else None
        gradient_alignment = _circular_angle_diff_deg(projected_angle, sector_gradient_direction)
        depth_anchor = cand.get("mouth_band_consistency") if isinstance(cand.get("mouth_band_consistency"), Mapping) else cand.get("depth_anchor")
        depth_anchor = depth_anchor if isinstance(depth_anchor, Mapping) else {}
        row = {
            "candidate_label": cand.get("label"),
            "axis_out_of_opening_camera": _json_vec(axis),
            "circle_center_camera_mm": _json_vec(center),
            "tilt_deg": _f(cand.get("tilt_deg"), 999.0),
            "semantic_projection_alignment_deg": semantic_alignment,
            "projected_axis_angle_deg_image": projected_angle,
            "sector_gradient_alignment_deg": gradient_alignment,
            "projected_axis_vector_uv": [float(v) for v in projected.tolist()] if projected is not None else None,
            "usable_by_m3934": bool(cand.get("usable", False)),
            "evidence_score": cand.get("evidence_score"),
            "dense_depth": cand.get("dense_depth"),
            "mouth_band_consistency": depth_anchor,
            "reprojection_chamfer_p90_px": cand.get("reprojection_chamfer_p90_px"),
            "circle_radial_residual_p90_mm": cand.get("circle_radial_residual_p90_mm"),
        }
        conic_rows.append((cand, row))

    result["conic_disambiguation"] = {
        "source_stage": conic.get("stage"),
        "source_classification": conic.get("classification"),
        "source_reason": conic.get("reason"),
        "selection_policy": "semantic_then_sector_gradient_then_m3934_then_evidence_margin",
        "candidates": [row for _cand, row in conic_rows],
    }
    if not conic_rows:
        result.update(
            classification="UNCERTAIN",
            recommended_ready_pose="NONE",
            production_grasp_policy="NONE",
            geometric_pose_classification=("TILTED_SHAPE_AXIS_UNRESOLVED" if shape_state == "TILTED" else "TRANSITION_AXIS_UNRESOLVED"),
            reason="no_analytic_conic_axis_candidates",
            axis_solution_reliable=False,
            thresholds=threshold_doc,
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    chosen = None
    chosen_row = None
    source = None
    selection_diag: Dict[str, Any] = {}

    # 1) Semantic body->mouth direction, when long enough.  The 16-frame field
    # set showed the correct branch remains obvious with 21-22 deg alignment,
    # so M39.5.1 uses 30 deg instead of M39.5.0's 20 deg.
    sem_sorted = sorted(
        [(float(row["semantic_projection_alignment_deg"]), cand, row)
         for cand, row in conic_rows if row.get("semantic_projection_alignment_deg") is not None],
        key=lambda x: x[0],
    )
    if semantic_offset_px >= min_semantic_px and sem_sorted:
        best = sem_sorted[0]
        runner = sem_sorted[1][0] if len(sem_sorted) > 1 else 180.0
        if best[0] <= max_align and (runner - best[0]) >= min_semantic_branch_sep:
            chosen, chosen_row, source = best[1], best[2], "semantic_body_to_mouth_projection"
            selection_diag.update(semantic_best_deg=best[0], semantic_runner_deg=runner)

    # 2) Depth-sector gradient direction is an independent signed cue.  It is
    # especially useful for mild tilt where body/mouth centroids remain almost
    # concentric.  Only branch *direction* is consumed here; magnitude is not.
    if chosen is None and sector_gradient_direction is not None:
        grad_sorted = sorted(
            [(float(row["sector_gradient_alignment_deg"]), cand, row)
             for cand, row in conic_rows if row.get("sector_gradient_alignment_deg") is not None],
            key=lambda x: x[0],
        )
        if grad_sorted:
            best = grad_sorted[0]
            runner = grad_sorted[1][0] if len(grad_sorted) > 1 else 180.0
            if best[0] <= max_gradient_align and (runner - best[0]) >= min_gradient_branch_sep:
                chosen, chosen_row, source = best[1], best[2], "sector_depth_gradient_direction"
                selection_diag.update(gradient_best_deg=best[0], gradient_runner_deg=runner)

    # 3) Reuse M39.3.4's depth winner when it is explicitly resolved.
    if chosen is None:
        selected = conic.get("selected_candidate") if isinstance(conic.get("selected_candidate"), Mapping) else None
        label = str((selected or {}).get("label") or "")
        if label.startswith("CONIC_"):
            for cand, row in conic_rows:
                if str(cand.get("label") or "") == label:
                    chosen, chosen_row, source = cand, row, "m39_3_4_resolved_depth_winner"
                    break

    # 4) Final bounded fallback: if the two evidence scores are sufficiently
    # separated, choose the stronger branch.  Shape classification is already
    # independently TILTED at this point.
    if chosen is None:
        score_sorted = sorted(
            [(float(row.get("evidence_score") or 0.0), cand, row) for cand, row in conic_rows],
            key=lambda x: x[0], reverse=True,
        )
        if score_sorted:
            best = score_sorted[0]
            runner = score_sorted[1][0] if len(score_sorted) > 1 else 0.0
            if (best[0] - runner) >= min_evidence_margin:
                chosen, chosen_row, source = best[1], best[2], "conic_evidence_margin"
                selection_diag.update(evidence_best=best[0], evidence_runner=runner)

    if chosen is None or chosen_row is None:
        result.update(
            classification="UNCERTAIN",
            recommended_ready_pose="NONE",
            production_grasp_policy="NONE",
            geometric_pose_classification=("TILTED_SHAPE_AXIS_UNRESOLVED" if shape_state == "TILTED" else "TRANSITION_AXIS_UNRESOLVED"),
            reason=("tilted_shape_but_signed_axis_unresolved" if shape_state == "TILTED" else "transition_shape_and_axis_unresolved"),
            axis_solution_reliable=False,
            axis_selection=selection_diag,
            thresholds=threshold_doc,
        )
        scene["m39_5_0_visible_mouth_axis_validation"] = result
        return result

    axis_out = _norm(np.asarray(chosen.get("normal_toward_camera"), dtype=np.float64))
    center = np.asarray(chosen.get("circle_center_camera_mm"), dtype=np.float64)
    tilt = _f(chosen.get("tilt_deg"), 999.0)
    near = _camera_near_rim_geometry(center, axis_out, inner_radius, outer_radius)
    reproj = _f(chosen.get("reprojection_chamfer_p90_px"), 999.0)
    radial = _f(chosen.get("circle_radial_residual_p90_mm"), 999.0)
    anchor = chosen.get("mouth_band_consistency") if isinstance(chosen.get("mouth_band_consistency"), Mapping) else chosen.get("depth_anchor")
    anchor = anchor if isinstance(anchor, Mapping) else {}
    valid_band = int(anchor.get("valid_sector_count") or 0)
    geometry_quality_pass = bool(reproj <= max_reproj and radial <= max_radial and valid_band >= min_band_sectors and near.get("available") is True)

    # M39.5.2 decouples physical tilt detection from robot READY selection.
    # A mildly tilted visible mouth (<30 deg) intentionally keeps the proven
    # VISIBLE_INITIAL + clock-3 grasp; only >=30 deg uses SIDE_INITIAL and the
    # camera-nearest-rim grasp.  Axis geometry quality remains mandatory only
    # for the SIDE camera-near grasp because that branch consumes the recovered
    # 3-D axis directly.
    if tilt < side_ready_tilt_min:
        classification, ready = "UPRIGHT_VISIBLE", "VISIBLE_INITIAL"
        grasp_policy = "VISIBLE_CLOCK3_MILD_TILT"
        geometric_pose = "MILD_TILT_VISIBLE" if tilt > 1.0 else "UPRIGHT"
        reason = "signed_axis_tilt_below_30deg_use_visible_clock3"
    else:
        classification, ready = "TILTED_VISIBLE_SIDE", "SIDE_INITIAL"
        grasp_policy = "SIDE_CAMERA_NEAR_RIM"
        geometric_pose = "TILTED_VISIBLE"
        reason = shape_reason if geometry_quality_pass else "tilted_shape_axis_geometry_quality_insufficient"

    result.update(
        classification=classification,
        recommended_ready_pose=ready,
        production_grasp_policy=grasp_policy,
        geometric_pose_classification=geometric_pose,
        reason=reason,
        axis_solution_reliable=bool(geometry_quality_pass),
        signed_axis_source=f"analytic_conic_A_B_selected_by_{source}",
        axis_selection={"source": source, **selection_diag},
        selected_conic_candidate=chosen_row,
        semantic_conic_alignment_deg=chosen_row.get("semantic_projection_alignment_deg"),
        sector_gradient_conic_alignment_deg=chosen_row.get("sector_gradient_alignment_deg"),
        axis_out_of_opening_camera=_json_vec(axis_out),
        axis_into_opening_camera=_json_vec(-axis_out),
        axis_tilt_from_box_z_deg=float(tilt),
        opening_center_camera_mm=_json_vec(center),
        camera_near_rim=near,
        axis_geometry_quality={
            "pass": bool(geometry_quality_pass),
            "reprojection_chamfer_p90_px": reproj,
            "maximum_reprojection_chamfer_p90_px": max_reproj,
            "circle_radial_residual_p90_mm": radial,
            "maximum_circle_radial_residual_p90_mm": max_radial,
            "mouth_band_valid_sector_count": valid_band,
            "minimum_mouth_band_valid_sector_count": min_band_sectors,
        },
        thresholds=threshold_doc,
    )
    scene["m39_5_0_visible_mouth_axis_validation"] = result
    return result

def draw_m3950_visible_mouth_axis_overlay(
    image_bgr: np.ndarray,
    result: Mapping[str, Any],
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    """Draw semantic direction, conic A/B, signed axis and camera-near rim target."""

    out = image_bgr
    if not isinstance(result, Mapping) or not bool(result.get("enabled", False)):
        return out

    sem = result.get("semantic_axis_2d") if isinstance(result.get("semantic_axis_2d"), Mapping) else {}
    rc = sem.get("ring_centroid_uv")
    mc = sem.get("mouth_centroid_uv")
    if isinstance(rc, Sequence) and isinstance(mc, Sequence) and len(rc) == 2 and len(mc) == 2:
        p0 = (int(round(float(rc[0]))), int(round(float(rc[1]))))
        p1 = (int(round(float(mc[0]))), int(round(float(mc[1]))))
        cv2.circle(out, p0, 4, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p1, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(out, p0, p1, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.18)

    conic = result.get("conic_disambiguation") if isinstance(result.get("conic_disambiguation"), Mapping) else {}
    for idx, row in enumerate(conic.get("candidates") or []):
        if not isinstance(row, Mapping):
            continue
        c = row.get("circle_center_camera_mm")
        a = row.get("axis_out_of_opening_camera")
        if not (isinstance(c, Sequence) and isinstance(a, Sequence) and len(c) == 3 and len(a) == 3):
            continue
        center = np.asarray(c, dtype=np.float64)
        axis = _norm(np.asarray(a, dtype=np.float64))
        uv0 = _project(center, intrinsics)
        uv1 = _project(center + 55.0 * axis, intrinsics)
        if uv0 is None or uv1 is None:
            continue
        color = (255, 0, 255) if idx == 0 else (255, 128, 0)
        cv2.arrowedLine(out, tuple(np.rint(uv0).astype(int)), tuple(np.rint(uv1).astype(int)), color, 1, cv2.LINE_AA, tipLength=0.15)
        label = f"{row.get('candidate_label')} {float(row.get('tilt_deg') or 0):.1f}d / {float(row.get('semantic_projection_alignment_deg') or 0):.1f}a"
        q = tuple(np.rint(uv1 + np.asarray([4.0, -4.0])).astype(int))
        cv2.putText(out, label, q, cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

    center = result.get("opening_center_camera_mm")
    axis = result.get("axis_out_of_opening_camera")
    if isinstance(center, Sequence) and isinstance(axis, Sequence) and len(center) == 3 and len(axis) == 3:
        c = np.asarray(center, dtype=np.float64)
        a = _norm(np.asarray(axis, dtype=np.float64))
        uv0 = _project(c, intrinsics)
        uv1 = _project(c + 75.0 * a, intrinsics)
        if uv0 is not None:
            cv2.circle(out, tuple(np.rint(uv0).astype(int)), 5, (0, 200, 255), 2, cv2.LINE_AA)
        if uv0 is not None and uv1 is not None:
            cv2.arrowedLine(out, tuple(np.rint(uv0).astype(int)), tuple(np.rint(uv1).astype(int)), (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.18)

    near = result.get("camera_near_rim") if isinstance(result.get("camera_near_rim"), Mapping) else {}
    point = near.get("camera_near_rim_midpoint_camera_mm")
    if isinstance(point, Sequence) and len(point) == 3:
        uv = _project(point, intrinsics)
        if uv is not None:
            p = tuple(np.rint(uv).astype(int))
            cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
            cv2.putText(out, "CAMERA-NEAR-RIM", (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    classification = str(result.get("classification") or "UNCERTAIN")
    ready = str(result.get("recommended_ready_pose") or "NONE")
    policy = str(result.get("production_grasp_policy") or "NONE")
    tilt = result.get("axis_tilt_from_box_z_deg")
    tilt_text = "n/a" if tilt is None else f"{float(tilt):.1f}deg"
    text = f"M39.5.2 {classification} | axis={tilt_text} | READY={ready}"
    detail = f"policy={policy} | signed-axis={'OK' if bool(result.get('axis_solution_reliable', False)) else 'UNRESOLVED'}"
    # This is the authoritative current-version banner.  It is intentionally
    # opaque and two lines tall because older diagnostic stages may also draw
    # text near the top-left of the same image.
    cv2.rectangle(out, (6, 6), (min(out.shape[1] - 6, 760), 58), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, detail, (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)
    return out
