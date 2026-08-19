"""M39.5.1 production grasp for tilted rings with a visible mouth.

Consumes the signed 3-D opening axis from M39.5.x and uses the *camera-nearest*
opening-wall midpoint as the pinch location.  The actual collision/clearance
validation reuses the field-validated M39.4.2 side-entry checker so the new
branch does not bypass any gripper, target-ring, neighbor, or box checks.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

import numpy as np  # type: ignore

from .robot_pose_transform import _rotation_to_quaternion_xyzw
from .segmentation import SegmentationInstance
from .side_entry_routing import build_m3942_side_entry_candidate


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return a / n


def _rows(R: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(R, dtype=np.float64)]


def _vec(v: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(v, dtype=np.float64).reshape(-1)]


def _find_ring(instances: Sequence[SegmentationInstance], ring_id: int) -> SegmentationInstance | None:
    for inst in instances:
        if str(inst.class_name) == "foam_ring" and int(inst.instance_id) == int(ring_id):
            return inst
    return None


def attach_m3951_tilted_visible_grasp_production(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = raw_config.get("m39_5_1_tilted_visible_grasp") or {}
    cfg = cfg if isinstance(cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", True))
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.5.1_tilted_visible_camera_near_rim_production",
        "enabled": enabled,
        "executed": False,
        "production_grasp_ready": False,
        "terminal_reject": False,
        "status": "disabled" if not enabled else "not_applicable",
        "reason": "disabled" if not enabled else "M39.5.x classification is not TILTED_VISIBLE_SIDE",
    }
    scene["m39_5_1_tilted_visible_grasp"] = summary
    if not enabled:
        return summary

    axis_result = scene.get("m39_5_0_visible_mouth_axis_validation")
    if not isinstance(axis_result, Mapping):
        summary.update(status="axis_result_missing", reason="M39.5.x axis result missing")
        return summary
    if str(axis_result.get("classification") or "").upper() != "TILTED_VISIBLE_SIDE":
        return summary

    # Once this class is identified, the historical visible clock-3 candidate
    # must never survive.  Either M39.5.1 supplies the camera-near-rim candidate
    # or this target is rejected/retried.
    scene["robot_candidate"] = None
    summary["executed"] = True

    rid_raw = axis_result.get("selected_ring_instance_id")
    try:
        rid = int(rid_raw)
    except Exception:
        summary.update(status="target_missing", reason="selected tilted-visible ring id missing", terminal_reject=True)
        return summary
    ring = _find_ring(instances, rid)
    if ring is None:
        summary.update(status="target_missing", reason="selected tilted-visible ring instance not found", terminal_reject=True, selected_ring_instance_id=rid)
        return summary

    if axis_result.get("axis_solution_reliable") is not True:
        summary.update(
            status="axis_not_reliable",
            reason="tilted-visible shape is valid but signed 3-D axis did not pass M39.5.1 quality gates",
            terminal_reject=True,
            selected_ring_instance_id=rid,
        )
        return summary

    center_raw = axis_result.get("opening_center_camera_mm")
    axis_in_raw = axis_result.get("axis_into_opening_camera")
    near = axis_result.get("camera_near_rim") if isinstance(axis_result.get("camera_near_rim"), Mapping) else {}
    x_raw = near.get("camera_near_radial_direction_camera")
    if not (
        isinstance(center_raw, Sequence) and len(center_raw) == 3
        and isinstance(axis_in_raw, Sequence) and len(axis_in_raw) == 3
        and isinstance(x_raw, Sequence) and len(x_raw) == 3
    ):
        summary.update(status="frame_missing", reason="opening center / insertion axis / camera-near radial missing", terminal_reject=True, selected_ring_instance_id=rid)
        return summary

    center = np.asarray(center_raw, dtype=np.float64).reshape(3)
    z_axis = _unit(axis_in_raw)
    x_axis = _unit(x_raw)
    # Re-project x into the opening plane and explicitly rebuild an orthonormal
    # right-handed M38.6 visual grasp frame.
    x_axis = _unit(x_axis - z_axis * float(np.dot(x_axis, z_axis)))
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    if not (np.any(x_axis) and np.any(y_axis) and np.any(z_axis)):
        summary.update(status="frame_invalid", reason="camera-near grasp frame is degenerate", terminal_reject=True, selected_ring_instance_id=rid)
        return summary

    R = np.column_stack((x_axis, y_axis, z_axis))
    q = _rotation_to_quaternion_xyzw(R)
    opening_fit: Dict[str, Any] = {
        "reliable": True,
        "status": "m3951_visible_opening_frame_ready",
        "entry_endpoint": "VISIBLE_MOUTH",
        "entry_selection_rule": "visible_mouth_camera_nearest_wall_midpoint",
        "opening_center_camera_mm": _vec(center),
        "opening_frame_camera": {
            "origin_camera_mm": _vec(center),
            "coordinate_contract": "m38_6_visual_grasp",
            "x_closing_axis_camera": _vec(x_axis),
            "y_lateral_axis_camera": _vec(y_axis),
            "z_approach_axis_camera": _vec(z_axis),
            "tcp_forward_insertion_axis_camera": _vec(z_axis),
            "inner_to_outer_closing_axis_camera": _vec(x_axis),
            "rotation_matrix_rows": _rows(R),
            "quaternion_xyzw": _vec(q),
            "inner_finger_side": "negative_x",
            "outer_finger_side": "positive_x",
        },
    }

    # Reuse all established M39.4.2 collision checks, while allowing this stage
    # to own insertion/pregrasp lengths if explicitly configured.
    eval_config = deepcopy(dict(raw_config))
    side_cfg = dict(eval_config.get("m39_4_2_side_entry_validation") or {})
    for key in (
        "grasp_insertion_depth_mm",
        "pregrasp_offset_from_entry_mm",
        "inner_hole_clearance_margin_mm",
        "outer_finger_clearance_margin_mm",
        "target_material_intersection_tolerance_mm",
        "target_axial_tolerance_mm",
        "minimum_environment_clearance_mm",
        "default_component_minimum_box_clearance_mm",
        "component_minimum_box_clearance_mm",
        "include_robot_wrist_in_tool_collision",
        "pregrasp_to_grasp_samples",
        "stop_on_first_hard_reject",
    ):
        if key in cfg:
            side_cfg[key] = deepcopy(cfg[key])
    side_cfg["production_grasp_enabled"] = bool(cfg.get("production_grasp_enabled", True))
    side_cfg["gripper_closing_enabled"] = bool(cfg.get("gripper_closing_enabled", True))
    eval_config["m39_4_2_side_entry_validation"] = side_cfg

    evaluated = build_m3942_side_entry_candidate(
        ring,
        [x for x in instances if str(x.class_name) == "foam_ring"],
        depth_mm,
        intrinsics,
        opening_fit,
        eval_config,
    )
    production_ready = bool(evaluated.get("production_grasp_ready", False))
    summary.update(
        selected_ring_instance_id=rid,
        selected_mouth_instance_id=axis_result.get("selected_mouth_instance_id"),
        axis_tilt_from_box_z_deg=axis_result.get("axis_tilt_from_box_z_deg"),
        axis_source=axis_result.get("signed_axis_source"),
        camera_near_rim=deepcopy(near),
        opening_frame_camera=opening_fit["opening_frame_camera"],
        production_grasp_ready=production_ready,
        terminal_reject=not production_ready,
        status=("tilted_visible_grasp_production_ready" if production_ready else "tilted_visible_grasp_validation_rejected"),
        reason=("camera-nearest visible-mouth rim grasp passed all M39.4.2 collision/clearance checks" if production_ready else "camera-nearest visible-mouth rim grasp failed collision/clearance validation"),
        validation=deepcopy(evaluated),
        rejection_reasons=deepcopy(evaluated.get("rejection_reasons") or []),
    )

    candidate = evaluated.get("robot_candidate") if isinstance(evaluated.get("robot_candidate"), Mapping) else None
    if production_ready and isinstance(candidate, Mapping):
        candidate = deepcopy(dict(candidate))
        candidate["message_type"] = "foam_ring_tilted_visible_grasp_production_candidate"
        candidate["status"] = "tilted_visible_grasp_production_ready"
        candidate["reason"] = "M39.5.1 visible-mouth camera-nearest rim pinch"
        candidate["grasp_branch"] = "m39_5_1_tilted_visible_camera_near_rim_grasp"
        target = dict(candidate.get("target") or {})
        target.update({
            "ring_instance_id": rid,
            "mouth_instance_id": axis_result.get("selected_mouth_instance_id"),
            "axis_tilt_from_box_z_deg": axis_result.get("axis_tilt_from_box_z_deg"),
            "rim_policy": "camera_nearest_visible_opening_wall_midpoint",
        })
        candidate["target"] = target
        candidate["m39_5_1_tilted_visible_validation"] = {
            "axis_source": axis_result.get("signed_axis_source"),
            "axis_selection": deepcopy(axis_result.get("axis_selection") or {}),
            "axis_geometry_quality": deepcopy(axis_result.get("axis_geometry_quality") or {}),
            "flat_reference_shape_test": deepcopy(axis_result.get("flat_reference_shape_test") or {}),
            "camera_near_rim": deepcopy(near),
        }
        # Keep side_entry_validation because the robot validation script and
        # collision contract already understand it; add a semantic alias too.
        candidate["tilted_visible_entry_validation"] = deepcopy(candidate.get("side_entry_validation") or {})
        scene["robot_candidate"] = candidate
        scene["selected_grasp_branch"] = candidate["grasp_branch"]
        summary["robot_candidate"] = candidate
    else:
        scene["robot_candidate"] = None
    scene["m39_5_1_tilted_visible_grasp"] = summary
    return summary
