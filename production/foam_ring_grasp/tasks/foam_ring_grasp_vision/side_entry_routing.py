"""M39.4.2.2 short-pregrasp side-grasp routing for pure side-lying foam rings.

Consumes M39.4.1 opening reconstruction, validates the complete OPEN insertion
path plus the CLOSED grasp environment, and authorizes the first production
side-rim-pinch cycle when all gates pass.  Box clearance is evaluated with a
side-entry-specific, component-wise physical-clearance policy so that the
coarse mounting-disk cylinder and manual box model do not impose the historical
16-18 mm virtual margin on otherwise valid field poses.

Geometry contract (existing m38_6_visual_grasp frame):
- +X: closing axis, inner finger -> outer finger / camera-facing wall
- +Y: lateral tangent
- +Z: approach axis, selected opening -> cylinder interior / TCP forward

The gripper origin is NOT the cylinder opening centre.  Rim pinch requires the
fingertip midpoint to be centred on the target wall thickness, therefore ENTRY
is reconstructed at the camera-facing rim midpoint and GRASP is the same point
inserted axially into the cylinder.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .box_model_3d import BoxModel3D, load_box_model
from .gripper_model_3d import (
    build_static_gripper_model,
    check_full_gripper_static_final_pose,
    sample_component_surface_local,
)
from .partial_opening_cylinder import _deproject_mask, _dilate, _erode, _safe_float, _safe_int, _unit
from .robot_pose_transform import _rotation_to_quaternion_xyzw
from .segmentation import SegmentationInstance

_EPS = 1e-9


def _vec(value: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in value], dtype=np.float64).reshape(3)


def _json_vector(value: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def _rotation_from_frame(frame: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _unit(frame.get("x_closing_axis_camera") or [1.0, 0.0, 0.0])
    z = _unit(frame.get("z_approach_axis_camera") or [0.0, 0.0, 1.0])
    x = _unit(x - z * float(np.dot(x, z)))
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    return x, y, z, np.column_stack((x, y, z))


def _project(point: np.ndarray, intrinsics: Mapping[str, float]) -> List[float]:
    x, y, z = [float(v) for v in point]
    if z <= 1e-6:
        return [float("nan"), float("nan")]
    return [
        float(intrinsics["fx"]) * x / z + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * y / z + float(intrinsics["cy"]),
    ]


def _load_calibrated_box(raw_config: Mapping[str, Any]) -> Tuple[Optional[BoxModel3D], Optional[str]]:
    section = raw_config.get("box_wall") or {}
    if not bool(section.get("enabled", True)):
        return None, "box_wall_disabled"
    name = str(section.get("calibration_file") or "box_model.json")
    path = Path(name)
    if not path.is_absolute():
        package_root = Path(__file__).resolve().parents[2]
        path = package_root / "config" / path.name
    try:
        return load_box_model(path), None
    except Exception as error:
        return None, f"box_model_load_failed: {error}"


def _target_cylinder_coordinates(
    points_camera: np.ndarray,
    opening_center: np.ndarray,
    insertion_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    relative = np.asarray(points_camera, dtype=np.float64) - opening_center[None, :]
    axial = relative @ insertion_axis
    radial_vec = relative - axial[:, None] * insertion_axis[None, :]
    radial = np.linalg.norm(radial_vec, axis=1)
    return axial, radial


def _prepare_neighbor_clouds_side(
    all_rings: Sequence[SegmentationInstance],
    target_ring: SegmentationInstance,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
    *,
    opening_center: np.ndarray,
    insertion_axis: np.ndarray,
    outer_radius_mm: float,
    axial_length_mm: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    section = raw_config.get("neighbor_3d") or {}
    depth_cfg = raw_config.get("depth") or {}
    if not bool(section.get("enabled", True)):
        return [], {"status": "disabled", "instances": []}
    minimum_mm = _safe_float(section.get("minimum_depth_mm"), _safe_float(depth_cfg.get("minimum_mm"), 150.0))
    maximum_mm = _safe_float(section.get("maximum_depth_mm"), _safe_float(depth_cfg.get("maximum_mm"), 3000.0))
    erode_px = _safe_int(section.get("mask_erode_px"), 1)
    stride = max(1, _safe_int(section.get("point_stride"), 2))
    minimum_points = max(1, _safe_int(section.get("minimum_points_per_instance"), 12))
    maximum_points = max(minimum_points, _safe_int(section.get("maximum_points_per_instance"), 3000))
    target_overlap = _dilate(target_ring.mask, _safe_int(section.get("target_surface_exclusion_dilate_px"), 2))
    shell_tol = max(4.0, _safe_float(section.get("target_surface_exclusion_mm"), 12.0))

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for item in all_rings:
        if int(item.instance_id) == int(target_ring.instance_id):
            continue
        mask = _erode(item.mask, erode_px)
        points, pixels = _deproject_mask(depth_mm, mask, intrinsics, minimum_mm, maximum_mm)
        if stride > 1 and len(points):
            points = points[::stride]
            pixels = pixels[::stride]
        raw_count = int(len(points))
        removed_target = 0
        if len(points):
            xs = pixels[:, 0]
            ys = pixels[:, 1]
            overlap = target_overlap[ys, xs]
            if np.any(overlap):
                axial, radial = _target_cylinder_coordinates(points, opening_center, insertion_axis)
                looks_like_target = (
                    overlap
                    & (axial >= -8.0)
                    & (axial <= axial_length_mm + 8.0)
                    & (radial <= outer_radius_mm + shell_tol)
                )
                removed_target = int(np.count_nonzero(looks_like_target))
                keep = ~looks_like_target
                points = points[keep]
                pixels = pixels[keep]
        if len(points) > maximum_points:
            idx = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
            points = points[idx]
            pixels = pixels[idx]
        status = "ready" if len(points) >= minimum_points else "insufficient"
        row = {
            "instance_id": int(item.instance_id),
            "status": status,
            "raw_point_count": raw_count,
            "retained_point_count": int(len(points)),
            "removed_target_overlap_points": removed_target,
            "points_camera": points,
            "pixels_uv": pixels,
        }
        rows.append(row)
        summary_rows.append({k: v for k, v in row.items() if k not in {"points_camera", "pixels_uv"}})
    ready = sum(1 for row in rows if row["status"] == "ready")
    return rows, {
        "status": "clear_no_neighbors" if not rows else ("ready" if ready else "insufficient_depth_support"),
        "neighbor_instance_count": len(rows),
        "ready_instance_count": ready,
        "instances": summary_rows,
    }


def _component_points_camera(
    primitive: Any,
    origin: np.ndarray,
    rotation: np.ndarray,
    collision_cfg: Mapping[str, Any],
) -> np.ndarray:
    local = sample_component_surface_local(
        primitive,
        obb_resolution=_safe_int(collision_cfg.get("obb_surface_resolution"), 4),
        cylinder_radial_samples=_safe_int(collision_cfg.get("cylinder_radial_samples"), 24),
        cylinder_axial_samples=_safe_int(collision_cfg.get("cylinder_axial_samples"), 5),
    )
    return local @ rotation.T + origin[None, :]


def _target_self_clearance_at_pose(
    origin: np.ndarray,
    rotation: np.ndarray,
    opening_mm: float,
    opening_center: np.ndarray,
    insertion_axis: np.ndarray,
    inner_radius_mm: float,
    outer_radius_mm: float,
    axial_length_mm: float,
    geometry_cfg: Mapping[str, Any],
    collision_cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    model = build_static_gripper_model(opening_mm, geometry_cfg, collision_cfg)
    hole_margin = max(0.0, _safe_float(stage_cfg.get("inner_hole_clearance_margin_mm"), 2.0))
    outer_margin = max(0.0, _safe_float(stage_cfg.get("outer_finger_clearance_margin_mm"), 2.0))
    material_tol = max(0.0, _safe_float(stage_cfg.get("target_material_intersection_tolerance_mm"), 1.5))
    axial_tol = max(0.0, _safe_float(stage_cfg.get("target_axial_tolerance_mm"), 1.5))

    inner_points_inside: List[np.ndarray] = []
    outer_points_inside: List[np.ndarray] = []
    other_material_hits = 0
    material_hits_by_component: Dict[str, int] = {}
    component_rows: List[Dict[str, Any]] = []
    for primitive in model.components:
        points = _component_points_camera(primitive, origin, rotation, collision_cfg)
        axial, radial = _target_cylinder_coordinates(points, opening_center, insertion_axis)
        inside_axial = (axial >= -axial_tol) & (axial <= axial_length_mm + axial_tol)
        material = inside_axial & (radial >= inner_radius_mm - material_tol) & (radial <= outer_radius_mm + material_tol)
        hit_count = int(np.count_nonzero(material))
        name = str(primitive.name)
        if name.startswith("inner_"):
            relevant = points[inside_axial]
            if len(relevant):
                inner_points_inside.append(relevant)
        elif name.startswith("outer_"):
            relevant = points[inside_axial]
            if len(relevant):
                outer_points_inside.append(relevant)
        else:
            other_material_hits += hit_count
        if hit_count:
            material_hits_by_component[name] = hit_count
        component_rows.append({
            "name": name,
            "group": str(primitive.group),
            "inside_target_axial_sample_count": int(np.count_nonzero(inside_axial)),
            "target_material_intersection_sample_count": hit_count,
        })

    inner_all = np.vstack(inner_points_inside) if inner_points_inside else np.empty((0, 3), dtype=np.float64)
    outer_all = np.vstack(outer_points_inside) if outer_points_inside else np.empty((0, 3), dtype=np.float64)
    inner_clearance = None
    outer_clearance = None
    inner_pass = True
    outer_pass = True
    if len(inner_all):
        _, inner_radial = _target_cylinder_coordinates(inner_all, opening_center, insertion_axis)
        inner_clearance = float(inner_radius_mm - np.max(inner_radial))
        inner_pass = bool(inner_clearance >= hole_margin)
    if len(outer_all):
        _, outer_radial = _target_cylinder_coordinates(outer_all, opening_center, insertion_axis)
        outer_clearance = float(np.min(outer_radial) - outer_radius_mm)
        outer_pass = bool(outer_clearance >= outer_margin)

    # Material hits on the two fingers are already represented by the radial
    # containment/clearance tests.  All other gripper components must remain
    # entirely outside the target material during entry-only validation.
    status = "clear"
    reasons: List[str] = []
    if not inner_pass:
        status = "intersects"
        reasons.append("inner_finger_hole_envelope_failed")
    if not outer_pass:
        status = "intersects"
        reasons.append("outer_finger_clearance_failed")
    if other_material_hits > 0:
        status = "intersects"
        reasons.append("nonfinger_gripper_target_material_collision")
    return {
        "status": status,
        "opening_mm": float(opening_mm),
        "inner_finger_sample_count_inside_target_axial_range": int(len(inner_all)),
        "inner_finger_minimum_hole_clearance_mm": inner_clearance,
        "inner_finger_required_clearance_mm": float(hole_margin),
        "inner_finger_hole_envelope_pass": bool(inner_pass),
        "outer_finger_sample_count_inside_target_axial_range": int(len(outer_all)),
        "outer_finger_minimum_outer_clearance_mm": outer_clearance,
        "outer_finger_required_clearance_mm": float(outer_margin),
        "outer_finger_clearance_pass": bool(outer_pass),
        "nonfinger_target_material_intersection_samples": int(other_material_hits),
        "material_intersection_samples_by_component": material_hits_by_component,
        "rejection_reasons": reasons,
        "components": component_rows if bool(stage_cfg.get("include_target_component_details_in_json", False)) else [],
    }


def _m39421_side_box_policy(full: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Field-calibrated box policy for side insertion.

    Generic full-gripper collision includes box safety margins plus component
    inflation.  For side insertion this previously produced an effective
    16-18 mm mounting-disk keep-out and rejected field-verified poses.  This
    policy uses *physical* clearance per component.  The mounting disk is a
    conservative 70 mm cylinder, so a small modeled overlap is tolerated as
    calibration/model uncertainty; fingers/palm/fittings retain positive
    physical clearance requirements.
    """
    default_min = _safe_float(stage_cfg.get("default_component_minimum_box_clearance_mm"), 2.0)
    thresholds = dict(stage_cfg.get("component_minimum_box_clearance_mm") or {})
    rows = full.get("component_summaries") if isinstance(full.get("component_summaries"), list) else []
    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clearance = row.get("box_clearance_mm")
        if clearance is None:
            continue
        group = str(row.get("group") or row.get("name") or "unknown")
        threshold = _safe_float(thresholds.get(group), default_min)
        value = float(clearance)
        record = {
            "name": row.get("name"),
            "group": group,
            "clearance_mm": value,
            "required_minimum_mm": float(threshold),
            "generic_box_status": row.get("box_status"),
        }
        if value < threshold:
            violations.append(record)
        elif str(row.get("box_status") or "clear") != "clear":
            warnings.append(record)
    return {
        "mode": "component_physical_clearance_field_calibrated",
        "status": "rejected" if violations else ("warning" if warnings else "clear"),
        "hard_reject": bool(violations),
        "violations": violations,
        "warnings": warnings,
        "thresholds_mm": {str(k): float(v) for k, v in thresholds.items()},
        "default_minimum_mm": float(default_min),
    }


def _evaluate_open_pose(
    label: str,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    opening_mm: float,
    opening_center: np.ndarray,
    inner_radius_mm: float,
    outer_radius_mm: float,
    axial_length_mm: float,
    box_model: Optional[BoxModel3D],
    neighbor_clouds: Sequence[Mapping[str, Any]],
    raw_config: Mapping[str, Any],
    intrinsics: Mapping[str, float],
) -> Dict[str, Any]:
    geometry_cfg = dict(raw_config.get("gripper_geometry_3d") or {})
    collision_cfg = dict(raw_config.get("full_gripper_static_collision") or {})
    stage_cfg = raw_config.get("m39_4_2_side_entry_validation") or {}
    # M39.4.2 "full gripper" means the complete tool installed below the
    # flange: moving fingers/contact blocks/palm/mounting disk/fittings.  The
    # generic robot_wrist cylinder is an arm-envelope approximation rather than
    # gripper geometry and would reject almost every horizontal entry in a box.
    # Keep it available as an explicit opt-in diagnostic; robot arm/wrist
    # reachability/collision remains the controller/planner's responsibility.
    if not bool(stage_cfg.get("include_robot_wrist_in_tool_collision", False)):
        wrist = dict(geometry_cfg.get("robot_wrist") or {})
        wrist["enabled"] = False
        geometry_cfg["robot_wrist"] = wrist
    collision_cfg["enabled"] = True
    # M39.4.2.2 retains the field-calibrated component-wise physical box-clearance policy
    # below. Keep the generic checker informative, but do not let its large
    # safety-margin model hard-reject the pose before field-calibrated policy.
    collision_cfg["hard_reject_box_intersection"] = False
    collision_cfg["hard_reject_box_clearance"] = False
    if stage_cfg.get("minimum_environment_clearance_mm") is not None:
        collision_cfg["minimum_clearance_mm"] = float(stage_cfg.get("minimum_environment_clearance_mm"))
    full = check_full_gripper_static_final_pose(
        origin,
        x_axis,
        y_axis,
        z_axis,
        opening_mm,
        box_model,
        neighbor_clouds,
        geometry_cfg,
        collision_cfg,
        intrinsics=intrinsics,
    )
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    target = _target_self_clearance_at_pose(
        origin,
        rotation,
        opening_mm,
        opening_center,
        z_axis,
        inner_radius_mm,
        outer_radius_mm,
        axial_length_mm,
        geometry_cfg,
        collision_cfg,
        stage_cfg,
    )
    side_box_policy = _m39421_side_box_policy(full, stage_cfg)
    hard_environment = bool(side_box_policy.get("hard_reject")) or bool(full.get("hard_reject_neighbor")) or str(full.get("box_status")) == "unconfigured"
    clear = (not hard_environment) and str(target.get("status")) == "clear"
    return {
        "label": label,
        "origin_camera_mm": _json_vector(origin),
        "opening_mm": float(opening_mm),
        "status": "clear" if clear else "rejected",
        "environment": full,
        "side_box_policy": side_box_policy,
        "target_ring": target,
        "hard_environment_reject": bool(hard_environment),
        "hard_target_reject": str(target.get("status")) != "clear",
    }


def _evaluate_closed_environment_pose(
    label: str,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    opening_mm: float,
    box_model: Optional[BoxModel3D],
    neighbor_clouds: Sequence[Mapping[str, Any]],
    raw_config: Mapping[str, Any],
    intrinsics: Mapping[str, float],
) -> Dict[str, Any]:
    """Validate box/neighbour environment at the CLOSED grasp gap.

    Target-ring contact is intentionally not evaluated here because closing is
    supposed to compress the target wall.
    """
    geometry_cfg = dict(raw_config.get("gripper_geometry_3d") or {})
    collision_cfg = dict(raw_config.get("full_gripper_static_collision") or {})
    stage_cfg = raw_config.get("m39_4_2_side_entry_validation") or {}
    if not bool(stage_cfg.get("include_robot_wrist_in_tool_collision", False)):
        wrist = dict(geometry_cfg.get("robot_wrist") or {})
        wrist["enabled"] = False
        geometry_cfg["robot_wrist"] = wrist
    collision_cfg["enabled"] = True
    collision_cfg["hard_reject_box_intersection"] = False
    collision_cfg["hard_reject_box_clearance"] = False
    if stage_cfg.get("minimum_environment_clearance_mm") is not None:
        collision_cfg["minimum_clearance_mm"] = float(stage_cfg.get("minimum_environment_clearance_mm"))
    full = check_full_gripper_static_final_pose(
        origin, x_axis, y_axis, z_axis, opening_mm, box_model, neighbor_clouds,
        geometry_cfg, collision_cfg, intrinsics=intrinsics,
    )
    side_box_policy = _m39421_side_box_policy(full, stage_cfg)
    hard = bool(side_box_policy.get("hard_reject")) or bool(full.get("hard_reject_neighbor")) or str(full.get("box_status")) == "unconfigured"
    return {
        "label": label,
        "origin_camera_mm": _json_vector(origin),
        "opening_mm": float(opening_mm),
        "status": "rejected" if hard else "clear",
        "environment": full,
        "side_box_policy": side_box_policy,
        "hard_environment_reject": bool(hard),
        "target_contact_expected": True,
    }


def _linear_samples(a: np.ndarray, b: np.ndarray, count: int) -> List[np.ndarray]:
    return [(1.0 - f) * a + f * b for f in np.linspace(0.0, 1.0, max(2, int(count)))]


def _evaluate_path(
    pregrasp: np.ndarray,
    entry: np.ndarray,
    grasp: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    opening_mm: float,
    opening_center: np.ndarray,
    inner_radius_mm: float,
    outer_radius_mm: float,
    axial_length_mm: float,
    box_model: Optional[BoxModel3D],
    neighbor_clouds: Sequence[Mapping[str, Any]],
    raw_config: Mapping[str, Any],
    intrinsics: Mapping[str, float],
) -> Dict[str, Any]:
    cfg = raw_config.get("m39_4_2_side_entry_validation") or {}
    # ENTRY remains a geometric diagnostic/keyframe, but the robot does not stop
    # there.  The executable approach is one coaxial PREGRASP -> GRASP move.
    segments = [
        ("PREGRASP_TO_GRASP", pregrasp, grasp, _safe_int(cfg.get("pregrasp_to_grasp_samples"), 9)),
    ]
    stage_rows: List[Dict[str, Any]] = []
    worst_rows: List[Tuple[str, int, Dict[str, Any]]] = []
    for name, start, end, count in segments:
        samples = _linear_samples(start, end, count)
        pose_rows: List[Dict[str, Any]] = []
        for index, origin in enumerate(samples):
            pose = _evaluate_open_pose(
                f"{name}_{index}", origin, x_axis, y_axis, z_axis, opening_mm,
                opening_center, inner_radius_mm, outer_radius_mm, axial_length_mm,
                box_model, neighbor_clouds, raw_config, intrinsics,
            )
            pose["sample_index"] = int(index)
            pose["sample_count"] = int(len(samples))
            pose_rows.append(pose)
            if pose["status"] != "clear":
                worst_rows.append((name, index, pose))
                if bool(cfg.get("stop_on_first_hard_reject", True)):
                    break
        stage_status = "clear" if all(row["status"] == "clear" for row in pose_rows) and len(pose_rows) == len(samples) else "rejected"
        stage_rows.append({
            "stage": name,
            "status": stage_status,
            "sample_count_requested": int(len(samples)),
            "sample_count_evaluated": int(len(pose_rows)),
            "poses": pose_rows if bool(cfg.get("include_path_pose_details_in_json", False)) else [],
            "worst_pose": next((row for row in pose_rows if row["status"] != "clear"), None),
        })
        if stage_status != "clear" and bool(cfg.get("stop_on_first_hard_reject", True)):
            break
    status = "clear" if len(stage_rows) == 1 and all(row["status"] == "clear" for row in stage_rows) else "rejected"
    return {
        "status": status,
        "motion_scope": "PREGRASP_to_GRASP_open_gripper_direct; ENTRY_is_diagnostic_only",
        "gripper_closing_checked": False,
        "post_grasp_lift_checked": False,
        "stages": stage_rows,
        "first_reject": (
            {"stage": worst_rows[0][0], "sample_index": int(worst_rows[0][1]), "pose": worst_rows[0][2]}
            if worst_rows else None
        ),
    }


def build_m3942_side_entry_candidate(
    ring: SegmentationInstance,
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    opening_fit: Mapping[str, Any],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    cfg = raw_config.get("m39_4_2_side_entry_validation") or {}
    object_cfg = raw_config.get("object_geometry") or {}
    gripper_cfg = raw_config.get("gripper") or {}
    reasons: List[str] = []
    warnings: List[str] = []

    if not bool(opening_fit.get("reliable", False)):
        return {"status": "opening_not_reliable", "ready": False, "rejection_reasons": ["m3942_m3941_opening_not_reliable"]}
    frame = opening_fit.get("opening_frame_camera") or {}
    if not isinstance(frame, Mapping):
        return {"status": "opening_frame_missing", "ready": False, "rejection_reasons": ["m3942_opening_frame_missing"]}
    opening_center = _vec(opening_fit.get("opening_center_camera_mm") or [0.0, 0.0, 0.0])
    x_axis, y_axis, z_axis, rotation = _rotation_from_frame(frame)

    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)
    inner_radius = 0.5 * _safe_float(object_cfg.get("nominal_inner_diameter_mm"), 60.0)
    axial_length = _safe_float(object_cfg.get("axial_length_mm"), 70.0)
    if not (outer_radius > inner_radius > 0.0 and axial_length > 0.0):
        reasons.append("m3942_invalid_ring_size_prior")
    wall = outer_radius - inner_radius
    rim_radius = 0.5 * (outer_radius + inner_radius)

    minimum_opening = _safe_float(gripper_cfg.get("minimum_opening_mm"), 10.0)
    maximum_opening = _safe_float(gripper_cfg.get("maximum_opening_mm"), 75.0)
    preopen_clearance = _safe_float(gripper_cfg.get("preopen_clearance_mm"), 6.0)
    approach_opening = wall + 2.0 * preopen_clearance
    approach_opening = float(np.clip(approach_opening, minimum_opening + 0.5, maximum_opening - 3.0))

    compression = _safe_float(gripper_cfg.get("wall_compression_mm"), 1.5)
    future_target_gap = float(np.clip(wall - 2.0 * compression, minimum_opening + 0.5, maximum_opening - 0.5))

    insertion_depth = _safe_float(cfg.get("grasp_insertion_depth_mm"), _safe_float((raw_config.get("m39_4_1_side_opening_reconstruction") or {}).get("preview_insertion_depth_mm"), 18.0))
    pregrasp_offset = _safe_float(cfg.get("pregrasp_offset_from_entry_mm"), 35.0)
    if insertion_depth <= 0.0 or insertion_depth >= axial_length - 5.0:
        reasons.append("m3942_invalid_insertion_depth")
    if pregrasp_offset <= 10.0:
        reasons.append("m3942_invalid_pregrasp_offset")

    entry = opening_center + rim_radius * x_axis
    grasp = entry + insertion_depth * z_axis
    pregrasp = entry - pregrasp_offset * z_axis

    box_model, box_error = _load_calibrated_box(raw_config)
    if box_error:
        reasons.append("m3942_box_model_unavailable")
        warnings.append(box_error)

    neighbor_clouds, neighbor_summary = _prepare_neighbor_clouds_side(
        all_rings, ring, depth_mm, intrinsics, raw_config,
        opening_center=opening_center,
        insertion_axis=z_axis,
        outer_radius_mm=outer_radius,
        axial_length_mm=axial_length,
    )

    keyframes: Dict[str, Any] = {}
    if not reasons:
        for label, origin in (("PREGRASP", pregrasp), ("ENTRY", entry), ("GRASP", grasp)):
            keyframes[label.lower()] = _evaluate_open_pose(
                label, origin, x_axis, y_axis, z_axis, approach_opening,
                opening_center, inner_radius, outer_radius, axial_length,
                box_model, neighbor_clouds, raw_config, intrinsics,
            )
        path = _evaluate_path(
            pregrasp, entry, grasp, x_axis, y_axis, z_axis, approach_opening,
            opening_center, inner_radius, outer_radius, axial_length,
            box_model, neighbor_clouds, raw_config, intrinsics,
        )
        closed_grasp_environment = _evaluate_closed_environment_pose(
            "GRASP_CLOSED", grasp, x_axis, y_axis, z_axis, future_target_gap,
            box_model, neighbor_clouds, raw_config, intrinsics,
        )
    else:
        path = {"status": "skipped_prerequisite_failed", "rejection_reasons": list(reasons)}
        closed_grasp_environment = {"status": "skipped_prerequisite_failed"}

    for label in ("pregrasp", "entry", "grasp"):
        row = keyframes.get(label)
        if isinstance(row, Mapping) and str(row.get("status")) != "clear":
            reasons.append(f"m3942_{label}_collision_or_clearance_failed")
    if str(path.get("status")) != "clear":
        reasons.append("m3942_open_insertion_path_failed")
    if str(closed_grasp_environment.get("status")) != "clear":
        reasons.append("m39421_closed_grasp_environment_failed")

    ready = len(reasons) == 0
    production_enabled = bool(cfg.get("production_grasp_enabled", False))
    closing_enabled = bool(cfg.get("gripper_closing_enabled", False))
    production_ready = bool(ready and production_enabled and closing_enabled)
    quaternion = _rotation_to_quaternion_xyzw(rotation)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = grasp
    interface_cfg = raw_config.get("robot_interface") or {}
    grasp_frame = {
        "coordinate_frame": str(interface_cfg.get("camera_frame_id") or "camera_color_optical_frame"),
        "camera_optical_convention": "+X right, +Y down, +Z forward",
        "length_unit": str(interface_cfg.get("length_unit") or "mm"),
        "origin_camera_mm": _json_vector(grasp),
        "x_closing_axis_camera": _json_vector(x_axis),
        "y_lateral_axis_camera": _json_vector(y_axis),
        "z_approach_axis_camera": _json_vector(z_axis),
        "rotation_matrix_rows": [[float(v) for v in row] for row in rotation],
        "quaternion_xyzw": _json_vector(quaternion),
        "T_camera_grasp_rows": [[float(v) for v in row] for row in T],
        "orientation_policy": "M39.4.2_side_opening_axis_plus_camera_facing_arc",
        "orientation_diagnostics": {
            "source_axis_stage": "M39.4.0.1",
            "source_opening_stage": "M39.4.1",
            "visual_plus_z_semantics": "selected_opening_to_ring_interior",
            "visual_plus_x_semantics": "hole_to_camera_facing_outer_wall",
        },
    }

    target_gap = keyframes.get("grasp", {}).get("target_ring") if isinstance(keyframes.get("grasp"), Mapping) else {}
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.4.2.2_side_grasp_production_routing",
        "status": "side_grasp_production_ready" if production_ready else ("entry_geometry_ready" if ready else "entry_validation_rejected"),
        "ready": bool(ready),
        "entry_only_robot_validation_ready": bool(ready),
        "production_grasp_ready": bool(production_ready),
        "gripper_closing_enabled": bool(closing_enabled),
        "ring_instance_id": int(ring.instance_id),
        "entry_endpoint": opening_fit.get("entry_endpoint"),
        "entry_selection_rule": opening_fit.get("entry_selection_rule"),
        "opening_center_camera_mm": _json_vector(opening_center),
        "opening_center_uv": _project(opening_center, intrinsics),
        "rim_midpoint_radius_mm": float(rim_radius),
        "rim_midpoint_entry_camera_mm": _json_vector(entry),
        "entry_center_camera_mm": _json_vector(entry),
        "entry_center_uv": _project(entry, intrinsics),
        "grasp_center_camera_mm": _json_vector(grasp),
        "grasp_center_uv": _project(grasp, intrinsics),
        "pregrasp_center_camera_mm": _json_vector(pregrasp),
        "pregrasp_center_uv": _project(pregrasp, intrinsics),
        "insertion_depth_mm": float(insertion_depth),
        "pregrasp_offset_from_entry_mm": float(pregrasp_offset),
        "robot_motion_contract": {
            "approach": "SIDE_AVOIDANCE -> SIDE_PREGRASP -> SIDE_GRASP (direct; ENTRY diagnostic only)",
            "return_after_close": "SIDE_GRASP -> SIDE_AVOIDANCE -> SIDE_INITIAL",
            "entry_robot_stop_enabled": False,
        },
        "nominal_inner_radius_mm": float(inner_radius),
        "nominal_outer_radius_mm": float(outer_radius),
        "nominal_wall_thickness_mm": float(wall),
        "approach_opening_mm": float(approach_opening),
        "future_target_closing_gap_mm": float(future_target_gap),
        "closing_axis_camera": _json_vector(x_axis),
        "lateral_axis_camera": _json_vector(y_axis),
        "approach_vector_camera": _json_vector(z_axis),
        "grasp_frame_camera": grasp_frame,
        "neighbor_point_clouds": neighbor_summary,
        "keyframe_checks": keyframes,
        "path_collision": path,
        "closed_grasp_environment": closed_grasp_environment,
        "inner_finger_hole_envelope": {
            "status": (target_gap or {}).get("status"),
            "pass": (target_gap or {}).get("inner_finger_hole_envelope_pass"),
            "minimum_clearance_mm": (target_gap or {}).get("inner_finger_minimum_hole_clearance_mm"),
            "required_clearance_mm": (target_gap or {}).get("inner_finger_required_clearance_mm"),
        },
        "outer_finger_clearance": {
            "status": (target_gap or {}).get("status"),
            "pass": (target_gap or {}).get("outer_finger_clearance_pass"),
            "minimum_clearance_mm": (target_gap or {}).get("outer_finger_minimum_outer_clearance_mm"),
            "required_clearance_mm": (target_gap or {}).get("outer_finger_required_clearance_mm"),
        },
        "rejection_reasons": sorted(set(reasons)),
        "warnings": warnings,
        "robot_candidate": None,
        "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
    }

    if production_ready:
        result["robot_candidate"] = {
            "schema_version": "1.0",
            "message_type": "foam_ring_side_grasp_production_candidate",
            "status": "side_grasp_production_ready",
            "robot_ready": True,
            "entry_only_robot_validation_ready": True,
            "production_grasp_ready": True,
            "reason": "M39.4.2.2 short-pregrasp direct side rim-pinch production grasp",
            "grasp_branch": "m39_4_2_2_side_grasp_production",
            "target": {
                "ring_instance_id": int(ring.instance_id),
                "entry_endpoint": opening_fit.get("entry_endpoint"),
                "entry_selection_rule": opening_fit.get("entry_selection_rule"),
                "inner_finger_hole_envelope_pass": True,
                "outer_finger_clearance_pass": True,
                "path_collision_status": path.get("status"),
            },
            "grasp_frame_camera": grasp_frame,
            "pregrasp_center_camera_mm": _json_vector(pregrasp),
            "entry_center_camera_mm": _json_vector(entry),
            "grasp_center_camera_mm": _json_vector(grasp),
            "side_entry_validation": {
                "opening_center_camera_mm": _json_vector(opening_center),
                "rim_midpoint_entry_camera_mm": _json_vector(entry),
                "pregrasp_center_camera_mm": _json_vector(pregrasp),
                "entry_center_camera_mm": _json_vector(entry),
                "grasp_center_camera_mm": _json_vector(grasp),
                "insertion_depth_mm": float(insertion_depth),
                "approach_opening_mm": float(approach_opening),
                "inner_finger_hole_envelope": result["inner_finger_hole_envelope"],
                "outer_finger_clearance": result["outer_finger_clearance"],
                "keyframe_checks": keyframes,
                "path_collision": path,
                "closed_grasp_environment": closed_grasp_environment,
            },
            "gripper_command": {
                "mode": "SIDE_RIM_PINCH_CLOSE_AT_GRASP",
                "opening_mm": float(approach_opening),
                "target_close_opening_mm": float(future_target_gap),
                "close_allowed": True,
            },
        }
    return result


def attach_m3942_side_entry_validation(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    cfg = raw_config.get("m39_4_2_side_entry_validation") or {}
    enabled = bool(cfg.get("enabled", True))
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.4.2.2_side_grasp_production_routing",
        "enabled": enabled,
        "mode": str(cfg.get("mode") or "side_grasp_production"),
        "executed": False,
        "status": "disabled" if not enabled else "not_applicable",
        "entry_only_robot_validation_ready": False,
        "production_grasp_ready": False,
        "gripper_closing_enabled": bool(cfg.get("gripper_closing_enabled", False)),
    }
    if not enabled:
        scene["m39_4_2_side_entry_validation"] = summary
        return summary
    source = scene.get("m39_4_1_side_opening_reconstruction")
    selected = source.get("selected") if isinstance(source, Mapping) and isinstance(source.get("selected"), Mapping) else None
    if not isinstance(selected, Mapping) or not bool(selected.get("reliable", False)):
        summary["status"] = "no_reliable_m3941_opening"
        scene["m39_4_2_side_entry_validation"] = summary
        return summary
    ring_id = int(selected.get("ring_instance_id"))
    rings = [item for item in instances if item.class_name == "foam_ring"]
    ring = next((item for item in rings if int(item.instance_id) == ring_id), None)
    if ring is None:
        summary["status"] = "selected_ring_instance_missing"
        scene["m39_4_2_side_entry_validation"] = summary
        return summary

    result = build_m3942_side_entry_candidate(ring, rings, depth_mm, intrinsics, selected, raw_config)
    ready = bool(result.get("ready", False))
    production_ready = bool(result.get("production_grasp_ready", False))
    summary.update({
        "executed": True,
        "status": "side_grasp_production_ready" if production_ready else ("entry_geometry_ready" if ready else "entry_validation_rejected"),
        "entry_only_robot_validation_ready": ready,
        "production_grasp_ready": production_ready,
        "gripper_closing_enabled": bool(result.get("gripper_closing_enabled", False)),
        "selected_ring_instance_id": ring_id,
        "selected": result if production_ready else None,
        "diagnostic": result,
        "selected_grasp_branch": "m39_4_2_2_side_grasp_production" if production_ready else ("m39_4_2_2_side_geometry_ready" if ready else "m39_4_2_side_entry_rejected"),
        "terminal_reject": not production_ready,
        "reason": None if production_ready else ((result.get("rejection_reasons") or ["m39421_side_grasp_not_ready"])[0] if not ready else "m39421_production_grasp_disabled"),
        "display_reason_short": "M39.4.2.2 SIDE GRASP READY" if production_ready else ("M39.4.2.2 SIDE GEOMETRY READY - PRODUCTION DISABLED" if ready else "REJECT: M39.4.2.2 SIDE COLLISION/CLEARANCE"),
        "operator_action": "side_grasp_production_allowed" if production_ready else "inspect_m39_4_2_collision_debug",
        "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
    })
    if production_ready and isinstance(result.get("robot_candidate"), Mapping):
        # M39.4.2 is intentionally allowed to replace the no-mouth diagnostic
        # rejection with an ENTRY-only candidate.  It never replaces a normal
        # M39.3 visible-mouth production candidate.
        current = scene.get("robot_candidate")
        if not isinstance(current, Mapping):
            scene["robot_candidate"] = dict(result["robot_candidate"])
            scene["selected_grasp_branch"] = "m39_4_2_2_side_grasp_production"
            scene["operator_action"] = "side_grasp_production_allowed"
    else:
        if str(scene.get("selected_grasp_branch") or "").startswith("m39_4_1"):
            scene["operator_action"] = "inspect_m39_4_2_collision_debug"
    scene["m39_4_2_side_entry_validation"] = summary
    return summary


def draw_m3942_side_entry_overlay(image_bgr: np.ndarray, summary: Mapping[str, Any]) -> np.ndarray:
    output = image_bgr.copy()
    row = summary.get("selected") if isinstance(summary.get("selected"), Mapping) else summary.get("diagnostic")
    if not isinstance(row, Mapping):
        return output
    ready = bool(row.get("ready", False))
    points = []
    for key, color, label in (
        ("pregrasp_center_uv", (255, 255, 255), "P"),
        ("entry_center_uv", (0, 255, 255), "E"),
        ("grasp_center_uv", (0, 128, 255), "G"),
    ):
        uv = row.get(key)
        if isinstance(uv, list) and len(uv) == 2 and all(math.isfinite(float(v)) for v in uv):
            p = tuple(int(round(float(v))) for v in uv)
            cv2.circle(output, p, 7, color, 2, cv2.LINE_AA)
            cv2.putText(output, label, (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            points.append(p)
    if len(points) >= 2:
        for a, b in zip(points[:-1], points[1:]):
            cv2.arrowedLine(output, a, b, (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.15)
    title = "M39.4.2.2 SIDE GRASP READY" if bool(row.get("production_grasp_ready", False)) else ("M39.4.2.2 GEOMETRY READY" if ready else "M39.4.2.2 SIDE REJECT")
    color = (0, 220, 0) if ready else (0, 0, 255)
    cv2.putText(output, title, (14, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    hole = row.get("inner_finger_hole_envelope") if isinstance(row.get("inner_finger_hole_envelope"), Mapping) else {}
    outer = row.get("outer_finger_clearance") if isinstance(row.get("outer_finger_clearance"), Mapping) else {}
    detail = (
        f"inner={float(hole.get('minimum_clearance_mm') or 0.0):.1f}mm "
        f"outer={float(outer.get('minimum_clearance_mm') or 0.0):.1f}mm "
        f"path={str((row.get('path_collision') or {}).get('status') or 'n/a')}"
    )
    cv2.putText(output, detail, (14, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return output
