#!/usr/bin/env python3
"""Preflight checks for the cleaned foam_ring_grasp production package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.robot_pose_transform import RobotPoseTransformer

DEFAULT_CONFIG = REPO_ROOT / "production" / "foam_ring_grasp" / "config" / "line.yaml"
STATUS_URL = "http://127.0.0.1:19213/api/foam_ring/status"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _clock_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    flat = _mapping(raw.get("clock_search") or {}, "clock_search")
    tilted = _mapping(
        raw.get("m39_3_4_1_tilted_production_routing") or {},
        "m39_3_4_1_tilted_production_routing",
    )
    flat_pref = [int(v) for v in flat.get("preferred_clock_hours") or []]
    flat_primary = [int(v) for v in flat.get("primary_clock_hours") or []]
    tilted_pref = [int(v) for v in tilted.get("preferred_clock_hours") or []]
    tilted_fallback = [int(v) for v in tilted.get("fallback_clock_hours") or []]
    ok = (
        flat_pref == [3]
        and flat_primary == [3]
        and not bool(flat.get("fallback_to_remaining", True))
        and tilted_pref == [3]
        and not bool(tilted.get("fallback_enabled", True))
        and tilted_fallback == []
        and int(tilted.get("maximum_clock_candidates", 0)) == 1
    )
    if not ok:
        raise RuntimeError(
            "clock contract violated: production must be clock-3 only with all fallbacks disabled"
        )
    return {
        "status": "ok",
        "flat": {"preferred": flat_pref, "primary": flat_primary, "fallback": False},
        "tilted": {"preferred": tilted_pref, "fallback": False, "max_candidates": 1},
    }


def _visible_mouth_scope(raw: Mapping[str, Any]) -> dict[str, Any]:
    branch_a = _mapping(raw.get("m38_branch_a") or {}, "m38_branch_a")
    branch_b = _mapping(raw.get("m38_branch_b") or {}, "m38_branch_b")
    branch_d = _mapping(raw.get("m38_branch_d") or {}, "m38_branch_d")
    branch_c = _mapping(raw.get("m38_branch_c") or {}, "m38_branch_c")
    hybrid = _mapping(raw.get("hybrid_grasp") or {}, "hybrid_grasp")
    if not bool(branch_a.get("enabled", False)):
        raise RuntimeError("visible-mouth Branch A must remain enabled")
    if bool(branch_b.get("enabled", False)) or bool(branch_d.get("enabled", False)):
        raise RuntimeError("pre-M39.4 side/no-mouth branches B/D must remain disabled")
    if not bool(branch_c.get("enabled", False)) or not bool(branch_c.get("fast_terminate", False)):
        raise RuntimeError("pre-M39.4 no-mouth conservative reject must remain enabled")
    if bool(hybrid.get("legacy_m36_enabled", True)) or bool(hybrid.get("side_ring_fallback_enabled", True)):
        raise RuntimeError("legacy M36/M37 fallback must remain disabled in production")
    side_axis = _mapping(raw.get("m39_4_0_side_axis_recovery") or {}, "m39_4_0_side_axis_recovery")
    if not bool(side_axis.get("enabled", False)):
        raise RuntimeError("M39.4.0 side-axis recovery must be enabled")
    if bool(side_axis.get("robot_routing_enabled", True)):
        raise RuntimeError("M39.4.0 must remain diagnostic-only: robot_routing_enabled=false")
    if str(side_axis.get("mode") or "") != "online_diagnostic_only":
        raise RuntimeError("M39.4.0.1 mode must be online_diagnostic_only")
    topology = _mapping(side_axis.get("mouth_topology_gate") or {}, "m39_4_0_side_axis_recovery.mouth_topology_gate")
    if not bool(topology.get("enabled", False)):
        raise RuntimeError("M39.4.0.1 mouth topology gate must be enabled")
    ratio = float(topology.get("side_view_axis_ratio_max", 0.0))
    if not (0.30 <= ratio <= 0.60):
        raise RuntimeError("M39.4.0.1 side-view mouth axis-ratio gate is outside validated range")
    if float(side_axis.get("dual_seed_refine_quick_margin_below", 0.0)) <= 0.0:
        raise RuntimeError("M39.4.0.1 dual-seed rescue threshold must be positive")
    side_opening = _mapping(
        raw.get("m39_4_1_side_opening_reconstruction") or {},
        "m39_4_1_side_opening_reconstruction",
    )
    if not bool(side_opening.get("enabled", False)):
        raise RuntimeError("M39.4.1 side opening reconstruction must be enabled")
    if bool(side_opening.get("robot_routing_enabled", True)):
        raise RuntimeError("M39.4.1 must remain validation-only: robot_routing_enabled=false")
    if str(side_opening.get("mode") or "") != "online_validation_only":
        raise RuntimeError("M39.4.1 mode must be online_validation_only")
    side_entry = _mapping(
        raw.get("m39_4_2_side_entry_validation") or {},
        "m39_4_2_side_entry_validation",
    )
    if not bool(side_entry.get("enabled", False)):
        raise RuntimeError("M39.4.2.2 side grasp routing must be enabled")
    if str(side_entry.get("mode") or "") != "side_grasp_production":
        raise RuntimeError("M39.4.2.2 mode must be side_grasp_production")
    if not bool(side_entry.get("production_grasp_enabled", False)):
        raise RuntimeError("M39.4.2.2 production grasp must be enabled")
    if not bool(side_entry.get("gripper_closing_enabled", False)):
        raise RuntimeError("M39.4.2.2 gripper closing must be enabled")
    thresholds = side_entry.get("component_minimum_box_clearance_mm") or {}
    if float(thresholds.get("mounting_disk", 0.0)) < -6.01:
        raise RuntimeError("M39.4.2.2 mounting-disk model tolerance exceeds 6 mm")
    if float(side_entry.get("grasp_insertion_depth_mm", 0.0)) <= 0.0:
        raise RuntimeError("M39.4.2 grasp insertion depth must be positive")
    pregrasp_offset = float(side_entry.get("pregrasp_offset_from_entry_mm", 0.0))
    if not (10.0 <= pregrasp_offset <= 50.0):
        raise RuntimeError("M39.4.2.2 pregrasp offset must stay in the field-validated short range 10..50 mm")
    if int(side_entry.get("pregrasp_to_grasp_samples", 0)) < 5:
        raise RuntimeError("M39.4.2.2 direct PREGRASP->GRASP sweep needs at least 5 samples")
    m3950 = _mapping(
        raw.get("m39_5_0_visible_mouth_axis_validation") or {},
        "m39_5_0_visible_mouth_axis_validation",
    )
    if not bool(m3950.get("enabled", False)):
        raise RuntimeError("M39.5.x visible-mouth axis source must be enabled")
    if str(m3950.get("mode") or "") != "production_axis_source":
        raise RuntimeError("M39.5.x axis mode must be production_axis_source")
    flat_upright = float(m3950.get("flat_reference_axis_ratio_deficit_upright_max", -1.0))
    flat_tilted = float(m3950.get("flat_reference_axis_ratio_deficit_tilted_min", -1.0))
    if not (0.0 <= flat_upright < flat_tilted):
        raise RuntimeError("M39.5.1 flat-reference shape thresholds must preserve a transition band")
    # M39.5.2 deliberately has a *single* robot READY split: mild visible
    # tilt below side_ready_axis_tilt_min_deg keeps VISIBLE_INITIAL + clock-3,
    # while tilt at/above the threshold uses SIDE_INITIAL + camera-near-rim.
    # There is no UNCERTAIN gap in the READY split.  The only intentional
    # uncertainty/transition band is the independent flat-reference shape
    # test checked immediately above.
    side_ready_tilt = float(m3950.get("side_ready_axis_tilt_min_deg", -1.0))
    if not (10.0 <= side_ready_tilt <= 60.0):
        raise RuntimeError(
            "M39.5.2 side_ready_axis_tilt_min_deg must stay in the safe configurable range 10..60 deg"
        )
    if float(m3950.get("pure_side_mouth_axis_ratio_max", 0.0)) >= 0.50:
        raise RuntimeError("M39.5.1 PURE_SIDE mouth gate must not swallow moderately visible tilted mouths")
    m3951 = _mapping(
        raw.get("m39_5_1_tilted_visible_grasp") or {},
        "m39_5_1_tilted_visible_grasp",
    )
    if not bool(m3951.get("enabled", False)):
        raise RuntimeError("M39.5.1 tilted-visible production grasp must be enabled")
    if not bool(m3951.get("production_grasp_enabled", False)):
        raise RuntimeError("M39.5.1 production grasp must be enabled")
    if not bool(m3951.get("gripper_closing_enabled", False)):
        raise RuntimeError("M39.5.1 gripper closing must be enabled")
    if str(m3951.get("ready_pose") or "") != "SIDE_INITIAL":
        raise RuntimeError("M39.5.1 tilted-visible branch must route through SIDE_INITIAL")
    if str(m3951.get("rim_policy") or "") != "camera_nearest_visible_opening_wall_midpoint":
        raise RuntimeError("M39.5.1 rim policy must be camera-nearest opening wall midpoint")
    return {
        "status": "ok",
        "visible_mouth_branch": "enabled",
        "side_no_mouth": "M39.4.0.1_axis_recovery",
        "pseudo_mouth": "M39.4.0.1_geometry_reroute_to_side_axis",
        "side_opening": "M39.4.1_camera_facing_arc_opening_reconstruction",
        "side_entry": "M39.4.2.2_short_PREGRASP_direct_GRASP_LEFT_LINK7",
        "side_gripper_close": "enabled_at_GRASP",
        "side_production_grasp": "enabled_direct_PREGRASP_to_GRASP_and_GRASP_to_AVOIDANCE",
        "m39_5_0": "flat_reference_shape_plus_signed_axis_source",
        "m39_5_1": "SIDE_INITIAL_camera_nearest_rim_production_with_M39.4.2_collision_gates",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cleaned foam-ring production configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--service", action="store_true", help="also query the running 19213 service")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    raw = load_yaml(config_path)
    transformer = RobotPoseTransformer.from_mapping(raw, config_path)
    report: dict[str, Any] = {
        "status": "ok",
        "config": str(config_path),
        "task": raw.get("task"),
        "schema_version": raw.get("schema_version"),
        "clock_contract": _clock_contract(raw),
        "scope_contract": _visible_mouth_scope(raw),
        "robot_pose_transform": transformer.status(),
    }
    if args.service:
        response = requests.get(STATUS_URL, timeout=5.0)
        response.raise_for_status()
        service = response.json()
        if service.get("status") != "ok":
            raise RuntimeError(f"service status abnormal: {service}")
        report["service"] = service
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
