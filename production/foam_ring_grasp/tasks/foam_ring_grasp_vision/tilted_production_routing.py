"""M39.3.4.1 tilted production routing integration.

M39.3.4 resolves the front-visible ring state and, for TILTED targets, an
analytic circle-plane normal.  M39.3.4.1 is the narrow integration layer that
promotes that resolved tilted surface into the existing M35/M38 rim-pinch
geometry and collision machinery.

Routing contract
----------------
* FLAT      -> keep the proven M39.2.9 floor-constrained production candidate.
* TILTED    -> rebuild the selected target's clock candidate(s) on the
               M39.3.4 analytic plane, including grasp frame and static/3-D
               collision checks, then replace ``scene.robot_candidate``.
* UNCERTAIN -> never silently execute the old flat pose for the selected ring.
               The production candidate is removed when configured to reject.

This module deliberately does not introduce another surface solver.  It only
connects the frozen M39.3.4 surface decision to production geometry.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .geometry import (
    GeometryConfig,
    PlaneModel,
    _clock_candidate,
    _clock_is_preferred,
    _clock_positions,
    _clock_rank_key,
    _distance_map_to_mask_px,
    _prepare_neighbor_point_clouds,
)


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _i(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _norm(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= 1e-9:
        raise ValueError("invalid analytic surface normal")
    return arr / n


def _vector_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    va = _norm(a)
    vb = _norm(b)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _selected_pair(scene: Mapping[str, Any]) -> tuple[Dict[str, Any] | None, int | None, int | None]:
    selected_ring_id = scene.get("selected_ring_instance_id")
    for item in scene.get("instances") or []:
        if not isinstance(item, dict):
            continue
        ring_id = item.get("ring_instance_id")
        if selected_ring_id is not None and ring_id is not None:
            try:
                if int(ring_id) != int(selected_ring_id):
                    continue
            except Exception:
                continue
        if str(item.get("pose_strategy") or "") != "m38_1_front_annulus":
            continue
        mouth_id = item.get("mouth_instance_id")
        return item, (int(ring_id) if ring_id is not None else None), (int(mouth_id) if mouth_id is not None else None)
    return None, None, None


def _route_clock_hours(cfg: Mapping[str, Any], geometry_config: GeometryConfig, previous_hour: Any) -> list[int]:
    configured = cfg.get("preferred_clock_hours")
    hours: list[int] = []
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        for value in configured:
            hour = _i(value, -1)
            hour = 12 if hour == 0 else hour
            if 1 <= hour <= 12 and hour not in hours:
                hours.append(hour)
    if not hours:
        for value in geometry_config.section("clock_search").get("preferred_clock_hours") or [2, 3]:
            hour = _i(value, -1)
            hour = 12 if hour == 0 else hour
            if 1 <= hour <= 12 and hour not in hours:
                hours.append(hour)
    previous = _i(previous_hour, -1)
    previous = 12 if previous == 0 else previous
    if bool(cfg.get("include_previous_selected_clock", True)) and 1 <= previous <= 12 and previous not in hours:
        hours.append(previous)

    fallback = cfg.get("fallback_clock_hours") or []
    if bool(cfg.get("fallback_enabled", True)) and isinstance(fallback, Sequence) and not isinstance(fallback, (str, bytes)):
        for value in fallback:
            hour = _i(value, -1)
            hour = 12 if hour == 0 else hour
            if 1 <= hour <= 12 and hour not in hours:
                hours.append(hour)

    limit = max(1, _i(cfg.get("maximum_clock_candidates"), 4))
    return hours[:limit]


def _patch_robot_candidate(
    previous: Mapping[str, Any],
    best: Mapping[str, Any],
    *,
    tilt_deg: float,
    selected_surface: Mapping[str, Any],
    route_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    document = deepcopy(dict(previous))
    document["pose_source"] = "m39_3_4_1_analytic_conic_tilted_surface"
    document["production_surface_state"] = "TILTED"
    document["production_surface_route"] = deepcopy(dict(route_summary))

    target = document.get("target") if isinstance(document.get("target"), dict) else {}
    target.update({
        "clock_hour": best.get("clock_hour"),
        "clock_angle_deg_cw_from_12": best.get("clock_angle_deg_cw_from_12"),
        "clock_search_batch": "m39_3_4_1_tilted_routing",
        "clock_preferred": bool(best.get("clock_preferred", False)),
        "candidate_score": best.get("score"),
        "tilt_deg": float(tilt_deg),
        "surface_classification": "TILTED",
        "surface_candidate_label": selected_surface.get("candidate_label"),
        "surface_tilt_direction_deg_box": selected_surface.get("tilt_direction_deg_box"),
        "box_wall_status": best.get("box_wall_status"),
        "box_wall_clearance_mm": best.get("box_wall_clearance_mm"),
        "box_wall_nearest_wall": (best.get("box_wall") or {}).get("nearest_wall"),
        "box_wall_worst_stage": (best.get("box_wall") or {}).get("worst_stage"),
        "neighbor_3d_status": (best.get("neighbor_3d") or {}).get("status"),
        "neighbor_3d_clearance_mm": (best.get("neighbor_3d") or {}).get("minimum_clearance_mm"),
        "neighbor_3d_nearest_instance_id": (best.get("neighbor_3d") or {}).get("nearest_instance_id"),
        "neighbor_3d_colliding_instance_ids": (best.get("neighbor_3d") or {}).get("colliding_instance_ids"),
        "neighbor_3d_worst_stage": (best.get("neighbor_3d") or {}).get("worst_stage"),
        "full_gripper_static_status": (best.get("full_gripper_static") or {}).get("status"),
        "full_gripper_static_box_status": (best.get("full_gripper_static") or {}).get("box_status"),
        "full_gripper_static_neighbor_status": (best.get("full_gripper_static") or {}).get("neighbor_status"),
        "full_gripper_static_box_clearance_mm": (best.get("full_gripper_static") or {}).get("box_minimum_clearance_mm"),
        "full_gripper_static_neighbor_clearance_mm": (best.get("full_gripper_static") or {}).get("neighbor_minimum_clearance_mm"),
        "full_gripper_motion_status": (best.get("full_gripper_motion") or {}).get("status"),
        "full_gripper_motion_box_status": (best.get("full_gripper_motion") or {}).get("box_status"),
        "full_gripper_motion_neighbor_status": (best.get("full_gripper_motion") or {}).get("neighbor_status"),
        "full_gripper_motion_worst_stage": (best.get("full_gripper_motion") or {}).get("worst_stage"),
    })
    document["target"] = target
    document["grasp_frame_camera"] = deepcopy(best.get("grasp_frame_camera"))
    document["mounting_interface_frame_camera"] = deepcopy((best.get("full_gripper_static") or {}).get("mounting_interface_frame_camera"))
    document["pregrasp_center_camera_mm"] = deepcopy(best.get("pregrasp_center_camera_mm"))
    document["rim_plane_midpoint_camera_mm"] = deepcopy(best.get("rim_plane_midpoint_camera_mm"))
    document["inner_contact_camera_mm"] = deepcopy(best.get("inner_contact_camera_mm"))
    document["outer_contact_camera_mm"] = deepcopy(best.get("outer_contact_camera_mm"))
    document["pregrasp_motion"] = {
        "scope": "pregrasp_to_grasp_only",
        "post_grasp_lift_checked": False,
        "path_keyframes_camera": deepcopy((best.get("full_gripper_motion") or {}).get("path_keyframes_camera")),
        "stage_summaries": deepcopy((best.get("full_gripper_motion") or {}).get("stage_summaries")),
    }
    document["gripper_command"] = {
        "travel_opening_mm": (best.get("full_gripper_motion") or {}).get("travel_opening_mm"),
        "open_start_offset_mm": (best.get("full_gripper_motion") or {}).get("open_start_offset_mm"),
        "opening_before_approach_mm": best.get("approach_opening_mm"),
        "target_closing_gap_mm": best.get("target_closing_gap_mm"),
        "rim_insert_depth_mm": best.get("rim_insert_depth_mm"),
        "desired_wall_compression_each_side_mm": best.get("desired_wall_compression_each_side_mm"),
        "actual_wall_compression_each_side_mm": best.get("actual_wall_compression_each_side_mm"),
    }
    return document


def apply_m39341_tilted_production_routing(
    scene: Dict[str, Any],
    instances: Sequence[Any],
    depth_geometry: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    raw_config: Mapping[str, Any],
    geometry_config: GeometryConfig,
) -> Dict[str, Any]:
    """Promote a resolved M39.3.4 TILTED surface into the production candidate."""
    started = time.perf_counter()
    cfg_raw = raw_config.get("m39_3_4_1_tilted_production_routing") or {}
    cfg = cfg_raw if isinstance(cfg_raw, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.3.4.1_tilted_production_routing_integration",
        "enabled": enabled,
        "status": "disabled" if not enabled else "pending",
        "route": "NONE",
        "production_routing_enabled": enabled,
        "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
        "classification": None,
        "candidate_replaced": False,
        "terminal_reject": False,
        "reason": None,
        "timing_ms": {},
    }
    if not enabled:
        scene["m39_3_4_1_tilted_production_routing"] = summary
        return summary

    analytic_summary = scene.get("m39_3_4_analytic_conic_surface") or {}
    analytic = analytic_summary.get("selected") if isinstance(analytic_summary, Mapping) else None
    if not isinstance(analytic, Mapping):
        summary.update(status="not_applicable", reason="m39_3_4_selected_result_unavailable")
        scene["m39_3_4_1_tilted_production_routing"] = summary
        return summary
    classification = str(analytic.get("classification") or "UNCERTAIN").upper()
    summary["classification"] = classification
    selected_surface = analytic.get("selected_candidate") if isinstance(analytic.get("selected_candidate"), Mapping) else None

    previous_candidate = scene.get("robot_candidate")
    if not isinstance(previous_candidate, Mapping):
        summary.update(status="not_applicable", reason="no_existing_m38_1_production_candidate")
        scene["m39_3_4_1_tilted_production_routing"] = summary
        return summary
    branch = str(previous_candidate.get("grasp_branch") or scene.get("selected_grasp_branch") or "")
    if branch not in {"m38_1_clear_mouth_front_annulus_rim_pinch", "m36_mouth_visible_rim_pinch"}:
        summary.update(status="not_applicable", reason=f"selected_branch_not_front_visible:{branch}")
        scene["m39_3_4_1_tilted_production_routing"] = summary
        return summary

    # FLAT is deliberately boring: preserve the already validated M39.2.9
    # candidate bit-for-bit, but add an explicit route marker for the robot-side
    # validation script.
    if classification == "FLAT":
        summary.update(
            status="ok",
            route="M39.2.9_FLAT",
            reason="m39_3_4_classified_flat_keep_floor_constrained_candidate",
            source_tilt_deg=(selected_surface or {}).get("tilt_deg"),
        )
        candidate = dict(previous_candidate)
        candidate["production_surface_state"] = "FLAT"
        candidate["production_surface_route"] = deepcopy(summary)
        scene["robot_candidate"] = candidate
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    reject_uncertain = bool(cfg.get("reject_uncertain", True))
    if classification != "TILTED" or not isinstance(selected_surface, Mapping):
        summary.update(
            status="rejected" if reject_uncertain else "bypassed",
            route="REJECT" if reject_uncertain else "M39.2.9_FALLBACK",
            terminal_reject=reject_uncertain,
            reason=(str(analytic.get("reason") or "m39_3_4_surface_uncertain")),
            display_reason_short="圆环倾斜方向不确定，拒绝抓取" if reject_uncertain else "圆环倾斜方向不确定，沿用保守路径",
            operator_action="调整目标姿态或更换可见圆环后重新触发",
        )
        if reject_uncertain:
            scene["robot_candidate"] = None
            scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    source_tilt = _f(selected_surface.get("tilt_deg"), 999.0)
    maximum_tilt = min(
        _f(cfg.get("maximum_routed_tilt_deg"), 35.0),
        _f(geometry_config.section("gripper").get("robot_safe_max_tilt_deg"), 35.0),
    )
    minimum_tilt = _f(cfg.get("minimum_routed_tilt_deg"), 8.0)
    summary.update(
        source_candidate_label=selected_surface.get("candidate_label"),
        source_tilt_deg=float(source_tilt),
        source_tilt_direction_deg_box=selected_surface.get("tilt_direction_deg_box"),
        source_normal_toward_camera=deepcopy(selected_surface.get("normal_toward_camera")),
        allowed_tilt_range_deg=[float(minimum_tilt), float(maximum_tilt)],
    )
    if not (minimum_tilt <= source_tilt <= maximum_tilt):
        summary.update(
            status="rejected",
            route="REJECT",
            terminal_reject=True,
            reason=f"resolved_tilt_outside_production_range:{source_tilt:.3f}deg",
            display_reason_short="圆环倾角超出生产安全范围，拒绝抓取",
            operator_action="选择倾角更小的圆环后重新触发",
        )
        scene["robot_candidate"] = None
        scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    item, ring_id, mouth_id = _selected_pair(scene)
    by_id: Dict[int, Any] = {}
    for instance in instances:
        try:
            by_id[int(instance.instance_id)] = instance
        except Exception:
            continue
    ring = by_id.get(ring_id) if ring_id is not None else None
    mouth = by_id.get(mouth_id) if mouth_id is not None else None
    if item is None or ring is None or mouth is None:
        summary.update(status="rejected", route="REJECT", terminal_reject=True, reason="selected_ring_or_mouth_instance_unavailable", display_reason_short="倾斜目标实例数据不完整，拒绝抓取", operator_action="重新触发视觉")
        scene["robot_candidate"] = None
        scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    try:
        normal = _norm(selected_surface.get("normal_toward_camera"))
        center = np.asarray(selected_surface.get("circle_center_camera_mm"), dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(center)):
            raise ValueError("invalid analytic circle center")
        if float(np.dot(normal, center)) > 0.0:
            normal = -normal
        offset = -float(np.dot(normal, center))
        plane = PlaneModel(
            normal=normal,
            offset=offset,
            centroid=center.copy(),
            inlier_mask=np.ones((1,), dtype=bool),
            inlier_ratio=1.0,
            residual_median_mm=_f(((selected_surface.get("dense_depth") or {}).get("residual_median_mm")), 0.0),
            residual_p95_mm=_f(((selected_surface.get("dense_depth") or {}).get("residual_p90_mm")), 0.0),
        )
    except Exception as exc:
        summary.update(status="rejected", route="REJECT", terminal_reject=True, reason=f"analytic_plane_invalid:{type(exc).__name__}:{exc}", display_reason_short="倾斜平面数据无效，拒绝抓取", operator_action="重新触发视觉")
        scene["robot_candidate"] = None
        scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    classes = geometry_config.section("classes")
    ring_name = str(classes.get("foam_ring") or "foam_ring")
    all_rings = [instance for instance in instances if str(getattr(instance, "class_name", "")) == ring_name]
    other_ring_mask = np.zeros_like(np.asarray(ring.mask, dtype=bool))
    for other in all_rings:
        if int(other.instance_id) != int(ring.instance_id):
            other_ring_mask |= np.asarray(other.mask, dtype=bool)
    distance_map = _distance_map_to_mask_px(other_ring_mask)

    ring_mouth_masks: Dict[int, np.ndarray] = {}
    for pair_item in scene.get("instances") or []:
        if not isinstance(pair_item, Mapping):
            continue
        rid, mid = pair_item.get("ring_instance_id"), pair_item.get("mouth_instance_id")
        if rid is None or mid is None:
            continue
        try:
            mouth_instance = by_id.get(int(mid))
            if mouth_instance is not None:
                ring_mouth_masks[int(rid)] = np.asarray(mouth_instance.mask, dtype=bool)
        except Exception:
            continue

    neighbor_started = time.perf_counter()
    neighbor_clouds, neighbor_summary = _prepare_neighbor_point_clouds(
        all_rings,
        ring,
        ring_mouth_masks,
        depth_geometry,
        intrinsics,
        plane,
        geometry_config,
    )
    summary["timing_ms"]["neighbor_cloud_prepare_ms"] = round((time.perf_counter() - neighbor_started) * 1000.0, 3)

    previous_hour = ((previous_candidate.get("target") or {}).get("clock_hour")) if isinstance(previous_candidate.get("target"), Mapping) else None
    hours = _route_clock_hours(cfg, geometry_config, previous_hour)
    by_hour = {int(row["clock_hour"]): row for row in _clock_positions(12)}
    evaluated: list[Dict[str, Any]] = []
    eval_started = time.perf_counter()
    for hour in hours:
        clock = deepcopy(by_hour[hour])
        clock["search_batch"] = "m39_3_4_1_tilted_routing"
        clock["search_order"] = len(evaluated)
        try:
            candidate = _clock_candidate(
                clock,
                ring,
                mouth,
                other_ring_mask,
                neighbor_clouds,
                depth_geometry,
                intrinsics,
                plane,
                center,
                source_tilt,
                geometry_config,
                evaluation_level="full",
                other_ring_distance_map=distance_map,
            )
        except Exception as exc:
            candidate = {
                **dict(clock),
                "evaluation_stage": "full",
                "full_evaluated": True,
                "valid": False,
                "score": 0.0,
                "warnings": [],
                "rejection_reasons": [f"m39_3_4_1_clock_replan_exception:{type(exc).__name__}:{exc}"],
                "timing_ms": {},
            }
        candidate["production_surface_source"] = "M39.3.4_selected_analytic_candidate"
        candidate["production_surface_tilt_deg"] = float(source_tilt)
        candidate["clock_preferred"] = bool(_clock_is_preferred(candidate, geometry_config))
        evaluated.append(candidate)
    summary["timing_ms"]["clock_replan_ms"] = round((time.perf_counter() - eval_started) * 1000.0, 3)
    summary["evaluated_clock_hours"] = hours
    summary["candidate_count"] = len(evaluated)
    summary["valid_candidate_count"] = len([row for row in evaluated if bool(row.get("valid"))])
    summary["candidate_diagnostics"] = [
        {
            "clock_hour": row.get("clock_hour"),
            "valid": bool(row.get("valid")),
            "score": row.get("score"),
            "rejection_reasons": list(row.get("rejection_reasons") or []),
            "warnings": list(row.get("warnings") or []),
            "box_wall_status": row.get("box_wall_status"),
            "neighbor_3d_status": (row.get("neighbor_3d") or {}).get("status"),
            "full_gripper_static_status": (row.get("full_gripper_static") or {}).get("status"),
        }
        for row in evaluated
    ]
    valid = [row for row in evaluated if bool(row.get("valid"))]
    valid.sort(key=lambda row: _clock_rank_key(row, geometry_config), reverse=True)
    if not valid:
        summary.update(
            status="rejected",
            route="REJECT",
            terminal_reject=True,
            reason="tilted_surface_has_no_collision_valid_clock_candidate",
            display_reason_short="倾斜抓取姿态未通过碰撞/空间检查，拒绝抓取",
            operator_action="更换目标圆环或调整场景后重新触发",
        )
        scene["robot_candidate"] = None
        scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    best = valid[0]
    expected_approach = -normal
    best_frame = best.get("grasp_frame_camera") if isinstance(best.get("grasp_frame_camera"), Mapping) else {}
    output_approach = best_frame.get("z_approach_axis_camera")
    orientation_tolerance_deg = max(0.1, _f(cfg.get("maximum_approach_normal_error_deg"), 2.0))
    try:
        approach_normal_error_deg = _vector_angle_deg(output_approach, expected_approach)
    except Exception as exc:
        approach_normal_error_deg = float("inf")
        summary["orientation_integrity_error"] = f"{type(exc).__name__}:{exc}"
    summary["approach_normal_error_deg"] = float(approach_normal_error_deg)
    summary["maximum_approach_normal_error_deg"] = float(orientation_tolerance_deg)
    if not math.isfinite(approach_normal_error_deg) or approach_normal_error_deg > orientation_tolerance_deg:
        summary.update(
            status="rejected",
            route="REJECT",
            terminal_reject=True,
            reason=f"tilted_grasp_frame_does_not_follow_analytic_normal:{approach_normal_error_deg:.3f}deg",
            display_reason_short="倾斜抓取姿态未保持目标法向，拒绝抓取",
            operator_action="检查抓取坐标系配置后重新触发",
        )
        scene["robot_candidate"] = None
        scene["production_routing_reject"] = deepcopy(summary)
        scene["m39_3_4_1_tilted_production_routing"] = summary
        summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return summary

    summary.update(
        status="ok",
        route="M39.3.4.1_TILTED",
        candidate_replaced=True,
        reason="m39_3_4_tilted_surface_replanned_with_existing_rim_pinch_collision_stack",
        selected_clock_hour=best.get("clock_hour"),
        selected_clock_angle_deg_cw_from_12=best.get("clock_angle_deg_cw_from_12"),
        selected_candidate_score=best.get("score"),
        output_approach_vector_camera=deepcopy(best.get("approach_vector_camera")),
        output_pregrasp_center_camera_mm=deepcopy(best.get("pregrasp_center_camera_mm")),
        output_grasp_origin_camera_mm=deepcopy((best.get("grasp_frame_camera") or {}).get("origin_camera_mm")),
    )

    # Preserve the pre-route production values inside the selected instance for
    # field debugging, then make all downstream consumers see one coherent
    # tilted plane/candidate.
    branch_a = item.get("m38_branch_a") if isinstance(item.get("m38_branch_a"), dict) else {}
    branch_a["m39_3_4_1_production_routing"] = deepcopy(summary)
    item["m38_branch_a"] = branch_a
    item["production_pose_source"] = "m39_3_4_1_analytic_conic_tilted_surface"
    item["production_surface_state"] = "TILTED"
    item["pose"] = {
        **(item.get("pose") if isinstance(item.get("pose"), Mapping) else {}),
        "production_override": "M39.3.4.1",
        "normal_toward_camera": [float(v) for v in normal.tolist()],
        "offset": float(offset),
        "centroid_camera_mm": [float(v) for v in center.tolist()],
    }
    item["ring_center_camera_mm"] = [float(v) for v in center.tolist()]
    item["ring_axis_toward_camera"] = [float(v) for v in normal.tolist()]
    item["approach_vector_camera"] = [-float(v) for v in normal.tolist()]
    item["tilt_deg"] = float(source_tilt)
    item["robot_safe_tilt"] = True
    item["eligible"] = True
    item["robot_eligible"] = True
    grasp = item.get("grasp") if isinstance(item.get("grasp"), dict) else {}
    grasp["m39_3_4_1_clock_candidates"] = evaluated
    grasp["best_clock_candidate"] = best
    item["grasp"] = grasp

    scene["selected_clock_hour"] = best.get("clock_hour")
    scene["selected_clock_angle_deg_cw_from_12"] = best.get("clock_angle_deg_cw_from_12")
    scene["selected_clock_search_batch"] = "m39_3_4_1_tilted_routing"
    scene["selected_clock_preferred"] = bool(best.get("clock_preferred", False))
    scene["robot_candidate"] = _patch_robot_candidate(
        previous_candidate,
        best,
        tilt_deg=source_tilt,
        selected_surface=selected_surface,
        route_summary=summary,
    )
    scene["m39_3_4_1_tilted_production_routing"] = summary
    summary["timing_ms"]["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return summary
