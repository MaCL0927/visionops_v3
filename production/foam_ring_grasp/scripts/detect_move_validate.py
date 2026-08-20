#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foam-ring production validation for two frozen branches with joint-space
branch-ready routing:
- upright visible-mouth: existing frozen clock-3 visible grasp;
- tilted visible-mouth: M39.5.1 camera-nearest-rim grasp from SIDE_INITIAL;
- pure side-lying: M39.4.2.2 short-PREGRASP direct side rim-pinch grasp.

The vision result owns the branch choice.  The robot is NOT rejected merely
because it is currently parked at the other branch's ready pose.  After vision
routing and operator confirmation, the script automatically switches between
VISIBLE_INITIAL and SIDE_INITIAL as required.  The switch itself uses the frozen
7-DOF joint coordinates and SDK move_j(); move_l() is no longer used for this
large ready-to-ready reorientation.  Automatic switching is allowed only when
the current joint state AND LEFT_LINK7 TCP pose agree with one of the two known
ready configurations; an unknown or inconsistent start remains rejected.

SIDE-family motion (tilted-visible and pure-side) uses its own ready pose and mandatory collision-avoidance
waypoint before PREGRASP. ENTRY is retained only as a visual/geometric reference;
the robot moves directly PREGRASP -> GRASP. After CLOSE it moves directly from
GRASP to the collision-avoidance waypoint before returning to SIDE_INITIAL. Side
motion always retains a second operator Enter; --single-enter applies only to the
mature visible-mouth branch.

IMPORTANT FRAME CONTRACT
------------------------
M39.3.4.1 outputs LEFT_LINK7 poses in `robot_default_base`, which was calibrated
against robot.motion.get_pose(ArmType.Left) with the SDK frame argument omitted.
Therefore all move_l/get_pose calls in this script deliberately use frame=None.
Do NOT change them to frame="base" unless the vision calibration/output frame is
also changed to SDK base_link.

Default mode is DRY RUN. Start with --execute only after dry-run output is sane.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import requests
from ysrobot import RobotClient, ArmType

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
VISION_INFER_URL = "http://127.0.0.1:19213/api/foam_ring/infer_once"
VISION_STATUS_URL = "http://127.0.0.1:19213/api/foam_ring/status"

ROBOT_IP = "192.168.213.111"
ROBOT_PORT = 5010
LOGIN_LEVEL = "L4"
LOGIN_PIN = "admin"

# Left gripper is controlled through the robot backend Modbus register API.
# The register is on the same authenticated RobotClient connection above.
LEFT_GRIPPER_REGISTER = 9661
GRIPPER_OPEN_VALUE = 1
GRIPPER_CLOSE_VALUE = 2
GRIPPER_OPEN_SETTLE_S = 1.0
GRIPPER_CLOSE_SETTLE_S = 1.0

EXPECTED_BASE_FRAME = "robot_default_base"
EXPECTED_FLANGE_FRAME = "left_link7"

# Branch-specific LEFT_LINK7 ready poses in SDK default / robot_default_base.
# These TCP values are retained for frame/arrival verification.  The actual
# VISIBLE_INITIAL <-> SIDE_INITIAL transition is commanded with move_j() using
# the frozen 7-DOF joint coordinates captured at the same two poses.
VISIBLE_INITIAL_POSE_MM = [
    438.543238,
    505.621470,
    882.193036,
    0.504203443,
    0.480148479,
    -0.502595128,
    0.512478745,
]
VISIBLE_INITIAL_JOINTS = [
    -1.384214505,
    1.195394052,
    1.578050856,
    0.950808014,
    -3.172495758,
    -1.835441114,
    -4.496910523,
]

SIDE_INITIAL_POSE_MM = [
    415.540859,
    611.048579,
    774.352657,
    -0.684075344,
    0.729161824,
    0.005389774,
    0.018300499,
]
SIDE_INITIAL_JOINTS = [
    -1.908033563,
    1.307147586,
    1.973150538,
    1.429727333,
    -1.632750113,
    -1.496229154,
    -1.611350731,
]

# Field-provided collision-avoidance waypoint.  Every side-lying production
# cycle must pass through this pose before PREGRASP and again after retreat.
SIDE_COLLISION_AVOIDANCE_POSE_MM = [
    397.3401910460248,
    324.9133103213571,
    764.5796470546751,
    -0.6840744498275672,
    0.729162751670706,
    0.00538940002121367,
    0.018297077267340554
]

# Conservative first-test motion parameters. Adjust only after validation.
TRAVEL_VEL = 10
TRAVEL_ACC = 30
APPROACH_VEL = 5
APPROACH_ACC = 20
RETURN_VEL = 10
RETURN_ACC = 30
# Branch-ready switching is a fixed joint-space transit between the two captured
# 7-DOF configurations. Keep the first field tests deliberately slow.
READY_SWITCH_JOINT_VEL = 10
READY_SWITCH_JOINT_ACC = 20
PLANNER = "pilz"

# Sanity checks. They catch unit/frame/extrinsic explosions; they are not a
# replacement for the robot controller's own reachability/collision checking.
MAX_ABS_POSITION_M = 2.0
MAX_INITIAL_TO_PREGRASP_MM = 1000.0
MAX_PREGRASP_TO_GRASP_MM = 250.0
MIN_PREGRASP_TO_GRASP_MM = 10.0
MAX_START_OFFSET_FROM_INITIAL_MM = 80.0
MAX_START_ROTATION_FROM_INITIAL_DEG = 15.0
# Joint state is now the primary identity of a ready configuration.  Direct/raw
# SDK values are compared intentionally (no 2*pi wrapping) so a different
# multi-turn joint configuration is never treated as the same safe ready state.
MAX_START_JOINT_ERROR_RAD = math.radians(5.0)
# M39.5.2.2: a robot that is already TCP-tight at a frozen READY pose may
# naturally report a slightly different redundant 7-DOF solution after a
# previous motion. Up to 8deg max joint delta is accepted ONLY for this
# TCP-tight case, and is never treated as execution-ready: the script
# immediately normalizes it back to the frozen joint target with move_j.
MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD = math.radians(8.0)
VERIFY_READY_JOINT_TOL_RAD = math.radians(3.0)
VERIFY_TRANSLATION_TOL_MM = 8.0
VERIFY_ROTATION_TOL_DEG = 5.0

VISION_TIMEOUT_S = 30.0
VISION_WAIT_TIMEOUT_MS = 25000
LOG_DIR = Path("./data/foam_ring_robot_validation_logs")
EXPECTED_TILTED_ROUTE = "M39.3.4.1_TILTED"
EXPECTED_FLAT_ROUTE = "M39.2.9_FLAT"
DEFAULT_MIN_TILTED_ROUTE_DEG = 8.0
DEFAULT_MAX_TILTED_ROUTE_DEG = 35.0
MAX_PREGRASP_GRASP_ROTATION_DELTA_DEG = 1.0
SIDE_ENTRY_VEL = 3
SIDE_ENTRY_ACC = 15
SIDE_GRASP_VEL = 2
SIDE_GRASP_ACC = 10
SIDE_TRANSIT_VEL = 8
SIDE_TRANSIT_ACC = 25
MAX_SIDE_PREGRASP_TO_ENTRY_MM = 160.0
MAX_SIDE_ENTRY_TO_GRASP_MM = 40.0


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _quat_norm(q: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in q))


def _normalize_quat(q: Sequence[float]) -> List[float]:
    n = _quat_norm(q)
    if n <= 1e-12:
        raise ValueError("zero quaternion")
    return [float(v) / n for v in q]


def _quat_angle_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = _normalize_quat(q1)
    b = _normalize_quat(q2)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _distance_mm(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt(sum((float(p1[i]) - float(p2[i])) ** 2 for i in range(3)))


def _pose_mm_to_sdk(pose_mm: Sequence[float]) -> List[float]:
    if len(pose_mm) != 7:
        raise ValueError("pose must contain 7 values")
    return [
        float(pose_mm[0]) / 1000.0,
        float(pose_mm[1]) / 1000.0,
        float(pose_mm[2]) / 1000.0,
        float(pose_mm[3]),
        float(pose_mm[4]),
        float(pose_mm[5]),
        float(pose_mm[6]),
    ]


def _sdk_pose_to_mm(pose: Any) -> List[float]:
    return [
        float(pose.x) * 1000.0,
        float(pose.y) * 1000.0,
        float(pose.z) * 1000.0,
        float(pose.qx),
        float(pose.qy),
        float(pose.qz),
        float(pose.qw),
    ]


def _format_pose(label: str, pose_mm: Sequence[float]) -> str:
    return (
        f"{label:<22} ["
        f"{pose_mm[0]:.3f}, {pose_mm[1]:.3f}, {pose_mm[2]:.3f}, "
        f"{pose_mm[3]:.6f}, {pose_mm[4]:.6f}, {pose_mm[5]:.6f}, {pose_mm[6]:.6f}]"
    )


def _pose_from_document(doc: Mapping[str, Any], *, expected_frame: str) -> List[float]:
    frame_id = str(doc.get("frame_id") or "")
    if frame_id != expected_frame:
        raise RuntimeError(
            f"pose frame mismatch: {frame_id!r}, expected {expected_frame!r}"
        )
    pos = doc.get("position_mm")
    quat = doc.get("quaternion_xyzw")
    if not (isinstance(pos, list) and len(pos) == 3):
        raise RuntimeError("pose.position_mm missing/invalid")
    if not (isinstance(quat, list) and len(quat) == 4):
        raise RuntimeError("pose.quaternion_xyzw missing/invalid")
    pose = [float(v) for v in pos + quat]
    if not _finite(pose):
        raise RuntimeError("pose contains NaN/Inf")
    qn = _quat_norm(pose[3:7])
    if not (0.98 <= qn <= 1.02):
        raise RuntimeError(f"quaternion norm abnormal: {qn:.6f}")
    if any(abs(float(v)) > MAX_ABS_POSITION_M * 1000.0 for v in pose[:3]):
        raise RuntimeError(f"position exceeds sanity bound: {pose[:3]} mm")
    return pose


def _vector_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    va = [float(v) for v in a]
    vb = [float(v) for v in b]
    na = math.sqrt(sum(v * v for v in va))
    nb = math.sqrt(sum(v * v for v in vb))
    if na <= 1e-12 or nb <= 1e-12:
        raise ValueError("zero vector")
    dot = sum(x * y for x, y in zip(va, vb)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def make_vision_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    return session


def check_vision_service(session: requests.Session) -> None:
    r = session.get(VISION_STATUS_URL, timeout=5.0)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"vision service status abnormal: {data}")
    print("✓ Vision service reachable")


def trigger_vision(session: requests.Session, save_debug: bool) -> Dict[str, Any]:
    request_id = "foam-ring-robot-" + datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    payload = {
        "request_id": request_id,
        "wait": True,
        "timeout_ms": VISION_WAIT_TIMEOUT_MS,
        "save_debug": bool(save_debug),
    }
    print(f"\n[Vision] trigger request_id={request_id}")
    r = session.post(VISION_INFER_URL, json=payload, timeout=VISION_TIMEOUT_S)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"vision returned non-JSON HTTP {r.status_code}: {r.text[:500]}")
    if r.status_code != 200:
        raise RuntimeError(f"vision HTTP {r.status_code}: {json.dumps(data, ensure_ascii=False)}")
    if data.get("status") != "ok":
        raise RuntimeError(f"vision result status={data.get('status')!r}: {data.get('error')}")
    return data


def validate_m39341_production_route(
    data: Mapping[str, Any],
    *,
    minimum_tilted_deg: float,
    maximum_tilted_deg: float,
) -> Dict[str, Any]:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    routing = summary.get("m39_3_4_1_tilted_production_routing")
    if not isinstance(routing, Mapping):
        raise RuntimeError("M39.3.4.1 production routing summary missing; refuse robot motion")

    classification = str(routing.get("classification") or "").upper()
    route = str(routing.get("route") or "")
    status = str(routing.get("status") or "")
    source_tilt = routing.get("source_tilt_deg")
    candidate = data.get("candidate") if isinstance(data.get("candidate"), Mapping) else {}
    candidate_state = str(candidate.get("production_surface_state") or "").upper()
    candidate_route = candidate.get("production_surface_route") if isinstance(candidate.get("production_surface_route"), Mapping) else {}

    if bool(routing.get("terminal_reject", False)) or route == "REJECT":
        raise RuntimeError(
            "M39.3.4.1 production route rejected target: "
            f"{routing.get('display_reason_short') or routing.get('reason')}"
        )
    if status != "ok":
        raise RuntimeError(f"M39.3.4.1 routing status={status!r}, expected 'ok'")

    if classification == "TILTED":
        if route != EXPECTED_TILTED_ROUTE:
            raise RuntimeError(
                f"TILTED target did not use tilted production route: route={route!r}"
            )
        if candidate_state != "TILTED":
            raise RuntimeError(
                f"TILTED route returned candidate state={candidate_state!r}; refuse motion"
            )
        if source_tilt is None:
            raise RuntimeError("TILTED route missing source_tilt_deg")
        tilt = float(source_tilt)
        if not (minimum_tilted_deg <= tilt <= maximum_tilted_deg):
            raise RuntimeError(
                f"TILTED route angle {tilt:.2f}deg outside script safety range "
                f"{minimum_tilted_deg:.2f}..{maximum_tilted_deg:.2f}deg"
            )
        if str(candidate_route.get("route") or route) != EXPECTED_TILTED_ROUTE:
            raise RuntimeError("candidate.production_surface_route is inconsistent with scene routing")
    elif classification == "FLAT":
        if route != EXPECTED_FLAT_ROUTE:
            raise RuntimeError(f"FLAT target route mismatch: {route!r}")
        if candidate_state not in {"", "FLAT"}:
            raise RuntimeError(f"FLAT route returned candidate state={candidate_state!r}")
    else:
        raise RuntimeError(
            f"surface classification={classification!r} is not executable; expected FLAT/TILTED"
        )

    frame = candidate.get("grasp_frame_camera") if isinstance(candidate.get("grasp_frame_camera"), Mapping) else {}
    approach = frame.get("z_approach_axis_camera")
    if not (isinstance(approach, list) and len(approach) == 3 and _finite(approach)):
        raise RuntimeError("candidate grasp frame missing valid z_approach_axis_camera")

    approach_error_deg = 0.0
    if classification == "TILTED":
        source_normal = routing.get("source_normal_toward_camera")
        if not (isinstance(source_normal, list) and len(source_normal) == 3 and _finite(source_normal)):
            raise RuntimeError("TILTED route missing valid source_normal_toward_camera")
        expected_approach = [-float(v) for v in source_normal]
        approach_error_deg = _vector_angle_deg(approach, expected_approach)
        allowed_error_deg = float(routing.get("maximum_approach_normal_error_deg") or 2.0)
        if approach_error_deg > allowed_error_deg + 1e-6:
            raise RuntimeError(
                f"TILTED candidate approach differs from -analytic_normal by {approach_error_deg:.3f}deg "
                f"> {allowed_error_deg:.3f}deg; refuse motion"
            )
        routed_output = routing.get("output_approach_vector_camera")
        if isinstance(routed_output, list) and len(routed_output) == 3 and _finite(routed_output):
            routed_error = _vector_angle_deg(approach, routed_output)
            if routed_error > allowed_error_deg + 1e-6:
                raise RuntimeError(
                    f"candidate approach and routing output differ by {routed_error:.3f}deg; refuse motion"
                )

    return {
        "classification": classification,
        "route": route,
        "status": status,
        "source_tilt_deg": float(source_tilt) if source_tilt is not None else 0.0,
        "source_tilt_direction_deg_box": routing.get("source_tilt_direction_deg_box"),
        "source_candidate_label": routing.get("source_candidate_label"),
        "selected_clock_hour": routing.get("selected_clock_hour") or summary.get("selected_clock_hour"),
        "candidate_state": candidate_state,
        "visual_approach_camera": [float(v) for v in approach],
        "approach_normal_error_deg": float(approach_error_deg),
        "routing": dict(routing),
    }


def extract_left_link7_targets(data: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    if data.get("target_found") is not True:
        summary = data.get("scene_summary") or {}
        raise RuntimeError(
            "vision did not return a safe grasp candidate; "
            f"terminal_reject={data.get('terminal_reject')}, "
            f"reason={data.get('terminal_reject_display') or data.get('terminal_reject_message')}, "
            f"eligible_count={summary.get('eligible_count')}"
        )
    if data.get("terminal_reject") is True:
        raise RuntimeError(
            "vision terminal reject: "
            f"{data.get('terminal_reject_display') or data.get('terminal_reject_message')}"
        )
    if data.get("robot_pose_transform_ready") is not True:
        raise RuntimeError("robot_pose_transform_ready is not true; refuse robot motion")

    rt = data.get("robot_pose_transform")
    if not isinstance(rt, Mapping):
        raise RuntimeError("robot_pose_transform missing")
    if str(rt.get("status") or "") != "ok":
        raise RuntimeError(f"robot_pose_transform.status={rt.get('status')!r}, expected 'ok'")
    if str(rt.get("arm") or "").lower() != "left":
        raise RuntimeError(f"robot_pose_transform.arm={rt.get('arm')!r}, expected left")
    if str(rt.get("base_frame_id") or "") != EXPECTED_BASE_FRAME:
        raise RuntimeError(
            f"robot_pose_transform.base_frame_id={rt.get('base_frame_id')!r}, "
            f"expected {EXPECTED_BASE_FRAME!r}"
        )

    flange = rt.get("flange")
    if not isinstance(flange, Mapping) or flange.get("available") is not True:
        raise RuntimeError("robot_pose_transform.flange is unavailable")
    if str(flange.get("frame_id") or "") != EXPECTED_FLANGE_FRAME:
        raise RuntimeError(
            f"flange.frame_id={flange.get('frame_id')!r}, expected {EXPECTED_FLANGE_FRAME!r}"
        )

    pre = flange.get("pregrasp")
    grasp = flange.get("grasp")
    if not isinstance(pre, Mapping) or not isinstance(grasp, Mapping):
        raise RuntimeError("flange pregrasp/grasp missing")
    pre_doc = pre.get("pose_base")
    grasp_doc = grasp.get("pose_base")
    if not isinstance(pre_doc, Mapping) or not isinstance(grasp_doc, Mapping):
        raise RuntimeError("flange.pregrasp/grasp.pose_base missing")

    pre_pose = _pose_from_document(pre_doc, expected_frame=EXPECTED_BASE_FRAME)
    grasp_pose = _pose_from_document(grasp_doc, expected_frame=EXPECTED_BASE_FRAME)

    approach_mm = _distance_mm(pre_pose, grasp_pose)
    orientation_delta_deg = _quat_angle_deg(pre_pose[3:7], grasp_pose[3:7])
    if orientation_delta_deg > MAX_PREGRASP_GRASP_ROTATION_DELTA_DEG:
        raise RuntimeError(
            f"pregrasp/grasp orientation changed by {orientation_delta_deg:.3f}deg; "
            "M39.3.4.1 requires the same routed grasp orientation along the approach"
        )
    if not (MIN_PREGRASP_TO_GRASP_MM <= approach_mm <= MAX_PREGRASP_TO_GRASP_MM):
        raise RuntimeError(
            f"pregrasp->grasp distance abnormal: {approach_mm:.1f} mm "
            f"(allowed {MIN_PREGRASP_TO_GRASP_MM:.1f}..{MAX_PREGRASP_TO_GRASP_MM:.1f})"
        )
    home_to_pre = _distance_mm(VISIBLE_INITIAL_POSE_MM, pre_pose)
    if home_to_pre > MAX_INITIAL_TO_PREGRASP_MM:
        raise RuntimeError(
            f"initial->pregrasp distance abnormal: {home_to_pre:.1f} mm > "
            f"{MAX_INITIAL_TO_PREGRASP_MM:.1f} mm"
        )
    return pre_pose, grasp_pose



def extract_m3942_left_link7_targets(data: Mapping[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    m3942 = summary.get("m39_4_2_side_entry_validation") if isinstance(summary, Mapping) else None
    if not isinstance(m3942, Mapping) or not bool(m3942.get("executed", False)):
        raise RuntimeError("M39.4.2 side entry summary missing")
    if not bool(m3942.get("production_grasp_ready", False)):
        raise RuntimeError(
            "M39.4.2.2 side grasp is not production-ready: "
            f"{m3942.get('display_reason_short') or m3942.get('reason')}"
        )
    if str(data.get("selected_grasp_branch") or "") != "m39_4_2_2_side_grasp_production":
        raise RuntimeError(
            f"unexpected side branch: {data.get('selected_grasp_branch')!r}"
        )
    if data.get("target_found") is not True or data.get("terminal_reject") is True:
        raise RuntimeError("M39.4.2 ready summary did not produce a non-rejected candidate")
    candidate = data.get("candidate") if isinstance(data.get("candidate"), Mapping) else {}
    if not bool(candidate.get("production_grasp_ready", False)):
        raise RuntimeError("candidate.production_grasp_ready is not true")
    command = candidate.get("gripper_command") if isinstance(candidate.get("gripper_command"), Mapping) else {}
    if command.get("close_allowed") is not True:
        raise RuntimeError("M39.4.2.2 candidate does not explicitly allow gripper CLOSE")
    validation = candidate.get("side_entry_validation") if isinstance(candidate.get("side_entry_validation"), Mapping) else {}
    path = validation.get("path_collision") if isinstance(validation.get("path_collision"), Mapping) else {}
    if str(path.get("status") or "") != "clear":
        raise RuntimeError(f"M39.4.2 path_collision.status={path.get('status')!r}")
    inner = validation.get("inner_finger_hole_envelope") if isinstance(validation.get("inner_finger_hole_envelope"), Mapping) else {}
    outer = validation.get("outer_finger_clearance") if isinstance(validation.get("outer_finger_clearance"), Mapping) else {}
    if inner.get("pass") is not True:
        raise RuntimeError("M39.4.2 inner finger hole envelope did not pass")
    if outer.get("pass") is not True:
        raise RuntimeError("M39.4.2 outer finger clearance did not pass")
    closed_env = validation.get("closed_grasp_environment") if isinstance(validation.get("closed_grasp_environment"), Mapping) else {}
    if str(closed_env.get("status") or "") != "clear":
        raise RuntimeError(f"M39.4.2.2 closed grasp environment status={closed_env.get('status')!r}")
    if data.get("robot_pose_transform_ready") is not True:
        raise RuntimeError("robot_pose_transform_ready is not true")
    rt = data.get("robot_pose_transform") if isinstance(data.get("robot_pose_transform"), Mapping) else {}
    if str(rt.get("status") or "") != "ok":
        raise RuntimeError(f"robot_pose_transform.status={rt.get('status')!r}")
    flange = rt.get("flange") if isinstance(rt.get("flange"), Mapping) else {}
    if flange.get("available") is not True or str(flange.get("frame_id") or "") != EXPECTED_FLANGE_FRAME:
        raise RuntimeError("M39.4.2 LEFT_LINK7 flange transform unavailable")
    docs = {}
    for name in ("pregrasp", "entry", "grasp"):
        row = flange.get(name) if isinstance(flange.get(name), Mapping) else {}
        doc = row.get("pose_base") if isinstance(row.get("pose_base"), Mapping) else None
        if not isinstance(doc, Mapping):
            raise RuntimeError(f"flange.{name}.pose_base missing")
        docs[name] = _pose_from_document(doc, expected_frame=EXPECTED_BASE_FRAME)
    pre_pose, entry_pose, grasp_pose = docs["pregrasp"], docs["entry"], docs["grasp"]
    for name, pose in (("entry", entry_pose), ("grasp", grasp_pose)):
        dr = _quat_angle_deg(pre_pose[3:7], pose[3:7])
        if dr > MAX_PREGRASP_GRASP_ROTATION_DELTA_DEG:
            raise RuntimeError(f"M39.4.2 {name} orientation changed by {dr:.3f}deg")
    pre_entry = _distance_mm(pre_pose, entry_pose)
    entry_grasp = _distance_mm(entry_pose, grasp_pose)
    if not (10.0 <= pre_entry <= MAX_SIDE_PREGRASP_TO_ENTRY_MM):
        raise RuntimeError(f"side PREGRASP->ENTRY distance abnormal: {pre_entry:.1f} mm")
    if not (1.0 <= entry_grasp <= MAX_SIDE_ENTRY_TO_GRASP_MM):
        raise RuntimeError(f"side ENTRY->GRASP distance abnormal: {entry_grasp:.1f} mm")
    pre_grasp = _distance_mm(pre_pose, grasp_pose)
    expected_direct = pre_entry + entry_grasp
    if abs(pre_grasp - expected_direct) > 1.0:
        raise RuntimeError(
            f"side PREGRASP->GRASP is not a coaxial direct segment: "
            f"direct={pre_grasp:.1f} mm, via-entry={expected_direct:.1f} mm"
        )
    return pre_pose, entry_pose, grasp_pose


def extract_m3951_left_link7_targets(data: Mapping[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    m3951 = summary.get("m39_5_1_tilted_visible_grasp") if isinstance(summary, Mapping) else None
    if not isinstance(m3951, Mapping) or not bool(m3951.get("executed", False)):
        raise RuntimeError("M39.5.1 tilted-visible production summary missing")
    if not bool(m3951.get("production_grasp_ready", False)):
        raise RuntimeError(
            "M39.5.1 tilted-visible grasp is not production-ready: "
            f"{m3951.get('reason')} / {m3951.get('rejection_reasons')}"
        )
    expected_branch = "m39_5_1_tilted_visible_camera_near_rim_grasp"
    if str(data.get("selected_grasp_branch") or "") != expected_branch:
        raise RuntimeError(f"unexpected M39.5.1 branch: {data.get('selected_grasp_branch')!r}")
    if data.get("target_found") is not True or data.get("terminal_reject") is True:
        raise RuntimeError("M39.5.1 ready summary did not produce a non-rejected candidate")
    candidate = data.get("candidate") if isinstance(data.get("candidate"), Mapping) else {}
    if not bool(candidate.get("production_grasp_ready", False)):
        raise RuntimeError("M39.5.1 candidate.production_grasp_ready is not true")
    command = candidate.get("gripper_command") if isinstance(candidate.get("gripper_command"), Mapping) else {}
    if command.get("close_allowed") is not True:
        raise RuntimeError("M39.5.1 candidate does not explicitly allow gripper CLOSE")
    validation = candidate.get("tilted_visible_entry_validation") if isinstance(candidate.get("tilted_visible_entry_validation"), Mapping) else {}
    if not validation:
        validation = candidate.get("side_entry_validation") if isinstance(candidate.get("side_entry_validation"), Mapping) else {}
    path = validation.get("path_collision") if isinstance(validation.get("path_collision"), Mapping) else {}
    if str(path.get("status") or "") != "clear":
        raise RuntimeError(f"M39.5.1 path_collision.status={path.get('status')!r}")
    inner = validation.get("inner_finger_hole_envelope") if isinstance(validation.get("inner_finger_hole_envelope"), Mapping) else {}
    outer = validation.get("outer_finger_clearance") if isinstance(validation.get("outer_finger_clearance"), Mapping) else {}
    if inner.get("pass") is not True:
        raise RuntimeError("M39.5.1 inner finger hole envelope did not pass")
    if outer.get("pass") is not True:
        raise RuntimeError("M39.5.1 outer finger clearance did not pass")
    closed_env = validation.get("closed_grasp_environment") if isinstance(validation.get("closed_grasp_environment"), Mapping) else {}
    if str(closed_env.get("status") or "") != "clear":
        raise RuntimeError(f"M39.5.1 closed grasp environment status={closed_env.get('status')!r}")
    if data.get("robot_pose_transform_ready") is not True:
        raise RuntimeError("M39.5.1 robot_pose_transform_ready is not true")
    rt = data.get("robot_pose_transform") if isinstance(data.get("robot_pose_transform"), Mapping) else {}
    if str(rt.get("status") or "") != "ok":
        raise RuntimeError(f"robot_pose_transform.status={rt.get('status')!r}")
    flange = rt.get("flange") if isinstance(rt.get("flange"), Mapping) else {}
    if flange.get("available") is not True or str(flange.get("frame_id") or "") != EXPECTED_FLANGE_FRAME:
        raise RuntimeError("M39.5.1 LEFT_LINK7 flange transform unavailable")
    docs = {}
    for name in ("pregrasp", "entry", "grasp"):
        row = flange.get(name) if isinstance(flange.get(name), Mapping) else {}
        doc = row.get("pose_base") if isinstance(row.get("pose_base"), Mapping) else None
        if not isinstance(doc, Mapping):
            raise RuntimeError(f"M39.5.1 flange.{name}.pose_base missing")
        docs[name] = _pose_from_document(doc, expected_frame=EXPECTED_BASE_FRAME)
    pre_pose, entry_pose, grasp_pose = docs["pregrasp"], docs["entry"], docs["grasp"]
    for name, pose in (("entry", entry_pose), ("grasp", grasp_pose)):
        dr = _quat_angle_deg(pre_pose[3:7], pose[3:7])
        if dr > MAX_PREGRASP_GRASP_ROTATION_DELTA_DEG:
            raise RuntimeError(f"M39.5.1 {name} orientation changed by {dr:.3f}deg")
    pre_entry = _distance_mm(pre_pose, entry_pose)
    entry_grasp = _distance_mm(entry_pose, grasp_pose)
    pre_grasp = _distance_mm(pre_pose, grasp_pose)
    if not (10.0 <= pre_entry <= MAX_SIDE_PREGRASP_TO_ENTRY_MM):
        raise RuntimeError(f"M39.5.1 PREGRASP->ENTRY distance abnormal: {pre_entry:.1f} mm")
    if not (1.0 <= entry_grasp <= MAX_SIDE_ENTRY_TO_GRASP_MM):
        raise RuntimeError(f"M39.5.1 ENTRY->GRASP distance abnormal: {entry_grasp:.1f} mm")
    if abs(pre_grasp - (pre_entry + entry_grasp)) > 1.0:
        raise RuntimeError(
            f"M39.5.1 PREGRASP->GRASP is not coaxial: direct={pre_grasp:.1f} mm, "
            f"via-entry={pre_entry + entry_grasp:.1f} mm"
        )
    return pre_pose, entry_pose, grasp_pose

def save_result(data: Mapping[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    request_id = str(data.get("request_id") or int(time.time() * 1000))
    path = LOG_DIR / f"{request_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def connect_robot() -> RobotClient:
    robot = RobotClient(host=ROBOT_IP, port=ROBOT_PORT, timeout_ms=5000)
    print(f"[Robot] login {ROBOT_IP}:{ROBOT_PORT} ...")
    ret = robot.login(LOGIN_LEVEL, LOGIN_PIN)
    if not ret.success:
        raise RuntimeError(f"robot login failed: {ret.message}")
    ret = robot.connect()
    if not ret.success:
        raise RuntimeError(f"robot connect failed: {ret.message}")
    print("✓ Robot connected")
    return robot


def current_pose_mm(robot: RobotClient) -> List[float]:
    # Deliberately omit frame: this is the calibrated robot_default_base/default SDK frame.
    return _sdk_pose_to_mm(robot.motion.get_pose(ArmType.Left))


def current_joints(robot: RobotClient) -> List[float]:
    js = robot.motion.get_joint_state(ArmType.Left)
    joints = [float(v) for v in (js.positions or [])]
    if len(joints) != 7:
        raise RuntimeError(f"expected 7 LEFT-arm joints, got {len(joints)}: {joints}")
    if not _finite(joints):
        raise RuntimeError(f"LEFT-arm joint state contains NaN/Inf: {joints}")
    return joints


def _joint_abs_errors(actual: Sequence[float], target: Sequence[float]) -> List[float]:
    if len(actual) != 7 or len(target) != 7:
        raise ValueError("joint vectors must contain 7 values")
    # Deliberately do NOT wrap by 2*pi.  The captured raw multi-turn joint
    # configuration is part of the ready-pose safety contract.
    return [abs(float(a) - float(t)) for a, t in zip(actual, target)]


def _max_joint_error_rad(actual: Sequence[float], target: Sequence[float]) -> float:
    return max(_joint_abs_errors(actual, target))


def _format_joints(label: str, joints: Sequence[float]) -> str:
    values = ", ".join(f"{float(v):.6f}" for v in joints)
    return f"{label:<22} [{values}]"


def _joint_ready_match(actual: Sequence[float], target: Sequence[float]) -> bool:
    return _max_joint_error_rad(actual, target) <= MAX_START_JOINT_ERROR_RAD


def set_left_gripper(robot: RobotClient, *, opened: bool, label: str) -> None:
    """Control the LEFT gripper via Modbus register 9661.

    Site contract:
      9661 = 1 -> open
      9661 = 2 -> close

    We intentionally use robot.device.write_modbus() rather than the SDK's
    robot.finger API because the production gripper is wired to this register.
    """
    value = GRIPPER_OPEN_VALUE if opened else GRIPPER_CLOSE_VALUE
    action = "OPEN" if opened else "CLOSE"
    print(f"\n[Gripper] {label}: {action} (register={LEFT_GRIPPER_REGISTER}, value={value})")
    result = robot.device.write_modbus(LEFT_GRIPPER_REGISTER, value)
    if not result.success:
        raise RuntimeError(
            f"gripper {action} failed: register={LEFT_GRIPPER_REGISTER}, "
            f"value={value}, message={result.message}"
        )

    settle_s = GRIPPER_OPEN_SETTLE_S if opened else GRIPPER_CLOSE_SETTLE_S
    if settle_s > 0:
        time.sleep(settle_s)
    print(f"✓ LEFT gripper {action} command accepted")


def verify_reached(robot: RobotClient, target_mm: Sequence[float], label: str) -> None:
    actual = current_pose_mm(robot)
    dt = _distance_mm(actual, target_mm)
    dr = _quat_angle_deg(actual[3:7], target_mm[3:7])
    print(_format_pose(f"actual {label}", actual))
    print(f"  reach error: translation={dt:.3f} mm, rotation={dr:.3f} deg")
    if dt > VERIFY_TRANSLATION_TOL_MM or dr > VERIFY_ROTATION_TOL_DEG:
        raise RuntimeError(
            f"{label} reach verification failed: {dt:.3f} mm / {dr:.3f} deg"
        )


def move_l_mm(
    robot: RobotClient,
    pose_mm: Sequence[float],
    *,
    label: str,
    vel: int,
    acc: int,
) -> None:
    print(f"\n[Robot] move -> {label}")
    print(_format_pose(label, pose_mm))
    target = _pose_mm_to_sdk(pose_mm)
    result = robot.motion.move_l(
        arm=ArmType.Left,
        pose=target,
        vel=vel,
        acc=acc,
        wait=True,
        planner=PLANNER,
        # IMPORTANT: frame intentionally omitted / None.
    )
    if not result.success:
        raise RuntimeError(f"move {label} failed: {result.message}")
    verify_reached(robot, pose_mm, label)
    print(f"✓ {label} reached")


def verify_ready_reached(
    robot: RobotClient,
    *,
    target_pose_mm: Sequence[float],
    target_joints: Sequence[float],
    label: str,
) -> None:
    actual_joints = current_joints(robot)
    errors = _joint_abs_errors(actual_joints, target_joints)
    max_err = max(errors)
    print(_format_joints(f"actual {label} J", actual_joints))
    print(
        f"  joint reach error: max={max_err:.6f} rad / "
        f"{math.degrees(max_err):.3f} deg; per-joint="
        + "[" + ", ".join(f"{math.degrees(e):.2f}" for e in errors) + "] deg"
    )
    if max_err > VERIFY_READY_JOINT_TOL_RAD:
        raise RuntimeError(
            f"{label} joint verification failed: max joint error "
            f"{max_err:.6f} rad / {math.degrees(max_err):.3f} deg"
        )

    # Secondary independent check in the calibrated/default TCP frame.
    actual_pose = current_pose_mm(robot)
    dt = _distance_mm(actual_pose, target_pose_mm)
    dr = _quat_angle_deg(actual_pose[3:7], target_pose_mm[3:7])
    print(_format_pose(f"actual {label} TCP", actual_pose))
    print(f"  TCP reach error: translation={dt:.3f} mm, rotation={dr:.3f} deg")
    if dt > VERIFY_TRANSLATION_TOL_MM or dr > VERIFY_ROTATION_TOL_DEG:
        raise RuntimeError(
            f"{label} TCP secondary verification failed: {dt:.3f} mm / {dr:.3f} deg"
        )


def move_j_ready(
    robot: RobotClient,
    *,
    target_pose_mm: Sequence[float],
    target_joints: Sequence[float],
    label: str,
    vel: int,
    acc: int,
) -> None:
    if len(target_joints) != 7 or not _finite(target_joints):
        raise RuntimeError(f"{label} target joints invalid: {target_joints}")
    print(f"\n[Robot] move_j -> {label}")
    print(_format_joints(f"{label} J", target_joints))
    print(_format_pose(f"{label} TCP(ref)", target_pose_mm))
    result = robot.motion.move_j(
        arm=ArmType.Left,
        joints=[float(v) for v in target_joints],
        vel=vel,
        acc=acc,
        wait=True,
        planner=PLANNER,
    )
    if not result.success:
        raise RuntimeError(f"move_j {label} failed: {result.message}")
    verify_ready_reached(
        robot,
        target_pose_mm=target_pose_mm,
        target_joints=target_joints,
        label=label,
    )
    print(f"✓ {label} reached by move_j")


def print_vision_summary(
    data: Mapping[str, Any],
    pre_pose: Sequence[float],
    grasp_pose: Sequence[float],
    route_info: Mapping[str, Any],
) -> None:
    summary = data.get("scene_summary") or {}
    plane = summary.get("selected_annulus_plane_selection") or {}
    selected_hyp = plane.get("selected_hypothesis") or plane.get("selected") or "n/a"
    rescue = plane.get("floor_parallel_rescue_applied")
    print("\n" + "=" * 78)
    print("VISION RESULT")
    print("=" * 78)
    print(f"request_id               : {data.get('request_id')}")
    print(f"selected_grasp_branch    : {data.get('selected_grasp_branch')}")
    print(f"selected_clock_hour       : {summary.get('selected_clock_hour')}")
    print(f"selected_plane_hypothesis: {selected_hyp}")
    print(f"floor_parallel_rescue     : {rescue}")
    print(f"surface_classification    : {route_info.get('classification')}")
    print(f"production_surface_route  : {route_info.get('route')}")
    print(f"surface_tilt_deg          : {float(route_info.get('source_tilt_deg') or 0.0):.3f}")
    print(f"tilt_direction_deg_box    : {route_info.get('source_tilt_direction_deg_box')}")
    print(f"analytic_candidate        : {route_info.get('source_candidate_label')}")
    print(f"visual approach camera    : {route_info.get('visual_approach_camera')}")
    print(f"approach vs -normal error : {float(route_info.get('approach_normal_error_deg') or 0.0):.3f} deg")
    print(f"eligible_count            : {summary.get('eligible_count')}")
    print(_format_pose("LEFT_LINK7 pregrasp", pre_pose))
    print(_format_pose("LEFT_LINK7 grasp", grasp_pose))
    print(f"pregrasp -> grasp         : {_distance_mm(pre_pose, grasp_pose):.3f} mm")
    print(f"pre/grasp rotation delta  : {_quat_angle_deg(pre_pose[3:7], grasp_pose[3:7]):.3f} deg")
    print("=" * 78)





def print_m3942_side_entry_summary(data: Mapping[str, Any], poses: Tuple[Sequence[float], Sequence[float], Sequence[float]] | None = None) -> bool:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    m3942 = summary.get("m39_4_2_side_entry_validation") if isinstance(summary, Mapping) else None
    if not isinstance(m3942, Mapping) or not bool(m3942.get("executed", False)):
        return False
    row = m3942.get("selected") if isinstance(m3942.get("selected"), Mapping) else m3942.get("diagnostic")
    if not isinstance(row, Mapping):
        row = {}
    inner = row.get("inner_finger_hole_envelope") if isinstance(row.get("inner_finger_hole_envelope"), Mapping) else {}
    outer = row.get("outer_finger_clearance") if isinstance(row.get("outer_finger_clearance"), Mapping) else {}
    path = row.get("path_collision") if isinstance(row.get("path_collision"), Mapping) else {}
    print("\n" + "=" * 78)
    print("M39.4.2.2 SIDE PRODUCTION GRASP")
    print("=" * 78)
    print(f"request_id               : {data.get('request_id')}")
    print(f"status                   : {m3942.get('status')}")
    print(f"selected_grasp_branch    : {data.get('selected_grasp_branch')}")
    print(f"entry endpoint           : {row.get('entry_endpoint')} / {row.get('entry_selection_rule')}")
    print(f"opening center camera    : {row.get('opening_center_camera_mm')}")
    print(f"rim midpoint ENTRY       : {row.get('entry_center_camera_mm')}")
    print(f"GRASP preview            : {row.get('grasp_center_camera_mm')}")
    print(f"PREGRASP                 : {row.get('pregrasp_center_camera_mm')}")
    print(f"approach opening         : {row.get('approach_opening_mm')} mm")
    print(f"inner hole clearance     : {inner.get('minimum_clearance_mm')} mm / pass={inner.get('pass')}")
    print(f"outer finger clearance   : {outer.get('minimum_clearance_mm')} mm / pass={outer.get('pass')}")
    print(f"open insertion path      : {path.get('status')}")
    first = path.get("first_reject") if isinstance(path.get("first_reject"), Mapping) else None
    if first:
        print(f"first collision/reject   : {first.get('stage')} sample={first.get('sample_index')}")
    print(f"rejection_reasons        : {row.get('rejection_reasons')}")
    if poses is not None:
        pre_pose, entry_pose, grasp_pose = poses
        print(_format_pose("LEFT_LINK7 PREGRASP", pre_pose))
        print(_format_pose("LEFT_LINK7 ENTRY", entry_pose))
        print(_format_pose("LEFT_LINK7 GRASP", grasp_pose))
        print(f"PREGRASP -> ENTRY(ref)   : {_distance_mm(pre_pose, entry_pose):.3f} mm")
        print(f"ENTRY(ref) -> GRASP      : {_distance_mm(entry_pose, grasp_pose):.3f} mm")
        print(f"PREGRASP -> GRASP direct : {_distance_mm(pre_pose, grasp_pose):.3f} mm")
    print("robot action             : PREGRASP -> GRASP direct; CLOSE; GRASP -> AVOIDANCE direct")
    print("ENTRY robot stop         : DISABLED (diagnostic reference only)")
    print("GRASP/CLOSE              : ENABLED; CLOSE only after SIDE-GRASP is reached")
    print("=" * 78)
    return True

def print_m3941_side_opening_summary(data: Mapping[str, Any]) -> bool:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    m3941 = summary.get("m39_4_1_side_opening_reconstruction") if isinstance(summary, Mapping) else None
    if not isinstance(m3941, Mapping) or not bool(m3941.get("executed", False)):
        return False
    fit = m3941.get("selected") if isinstance(m3941.get("selected"), Mapping) else m3941.get("diagnostic")
    if not isinstance(fit, Mapping):
        fit = {}
    arc = fit.get("camera_facing_outer_arc") if isinstance(fit.get("camera_facing_outer_arc"), Mapping) else {}
    support = fit.get("opening_support") if isinstance(fit.get("opening_support"), Mapping) else {}
    frame = fit.get("side_grasp_frame_camera") if isinstance(fit.get("side_grasp_frame_camera"), Mapping) else {}
    print("\n" + "=" * 78)
    print("M39.4.1 CAMERA-FACING ARC + SIDE OPENING FRAME [VALIDATION ONLY]")
    print("=" * 78)
    print(f"request_id               : {data.get('request_id')}")
    print(f"status                   : {m3941.get('status')}")
    print(f"ring_instance_id         : {fit.get('ring_instance_id')}")
    print(f"entry_endpoint           : {fit.get('entry_endpoint')}")
    print(f"axis_image_angle_deg     : {fit.get('axis_image_angle_deg_0_180')}")
    print(f"arc residual med/p90     : {arc.get('residual_median_mm')} / {arc.get('residual_p90_mm')} mm")
    print(f"arc inlier/raw ratio     : {arc.get('arc_inlier_ratio')} / {arc.get('raw_arc_inlier_ratio')}")
    print(f"arc span                 : {arc.get('arc_span_deg_p5_p95')} deg")
    print(f"opening support status   : {support.get('status')}")
    print(f"opening drop ratio       : {support.get('drop_ratio')}")
    print(f"opening center camera    : {fit.get('opening_center_camera_mm')}")
    print(f"opening shift vs nominal : {fit.get('opening_shift_vs_m39401_nominal_endpoint_mm')} mm")
    print(f"preview grasp center     : {fit.get('preview_grasp_center_camera_mm')}")
    print(f"frame +X closing         : {frame.get('x_closing_axis_camera')}")
    print(f"frame +Y lateral         : {frame.get('y_lateral_axis_camera')}")
    print(f"frame +Z insertion       : {frame.get('z_approach_axis_camera')}")
    print(f"frame quaternion xyzw    : {frame.get('quaternion_xyzw')}")
    print(f"rejection_reasons        : {fit.get('rejection_reasons')}")
    print(f"warnings                 : {fit.get('warnings')}")
    print("robot motion             : DISABLED BY M39.4.1 CONTRACT")
    print("=" * 78)
    return True

def print_m3940_side_axis_summary(data: Mapping[str, Any]) -> bool:
    summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
    m3940 = summary.get("m39_4_0_side_axis_recovery") if isinstance(summary, Mapping) else None
    if not isinstance(m3940, Mapping) or not bool(m3940.get("executed", False)):
        return False
    selected = m3940.get("selected") if isinstance(m3940.get("selected"), Mapping) else None
    fits = m3940.get("fits") if isinstance(m3940.get("fits"), list) else []
    diagnostic = selected if selected is not None else (fits[0] if fits and isinstance(fits[0], Mapping) else {})
    print("\n" + "=" * 78)
    print("M39.4.0.1 SIDE TOPOLOGY + AXIS RECOVERY [VALIDATION ONLY]")
    print("=" * 78)
    print(f"request_id               : {data.get('request_id')}")
    topology = summary.get("m39_4_0_1_mouth_topology_arbitration") if isinstance(summary, Mapping) else {}
    if isinstance(topology, Mapping):
        print(f"mouth topology status    : {topology.get('status')}")
        print(f"pseudo mouth ring ids    : {topology.get('pseudo_mouth_ring_ids')}")
    print(f"status                   : {m3940.get('status')}")
    print(f"selected_grasp_branch    : {data.get('selected_grasp_branch')}")
    print(f"ring_instance_id         : {diagnostic.get('ring_instance_id')}")
    print(f"axis_reliable            : {diagnostic.get('axis_reliable', diagnostic.get('reliable'))}")
    print(f"center_reliable          : {diagnostic.get('center_reliable')}")
    print(f"candidate_source         : {diagnostic.get('side_candidate_source')}")
    print(f"pseudo_mouth_axis_ratio  : {diagnostic.get('pseudo_mouth_axis_ratio')}")
    print(f"axis_image_angle_deg     : {diagnostic.get('axis_image_angle_deg_0_180')}")
    print(f"axis_camera_undirected   : {diagnostic.get('axis_camera_undirected')}")
    print(f"quick axis margin        : {diagnostic.get('axis_score_margin')}")
    print(f"dual-seed rescue         : {diagnostic.get('dual_seed_refine_applied')}")
    print(f"full geometry margin     : {diagnostic.get('full_geometry_score_margin')}")
    refinement = diagnostic.get("selected_axis_refinement") if isinstance(diagnostic.get("selected_axis_refinement"), Mapping) else {}
    print(f"radial residual med/p90  : {refinement.get('radial_residual_median_mm')} / {refinement.get('radial_residual_p90_mm')} mm")
    print(f"center_height_error_mm   : {diagnostic.get('center_height_error_mm')}")
    print(f"endpoint A uv            : {diagnostic.get('endpoint_a_uv')}")
    print(f"endpoint B uv            : {diagnostic.get('endpoint_b_uv')}")
    print(f"entry_endpoint           : {diagnostic.get('entry_endpoint')}")
    print(f"entry_selection_rule     : {diagnostic.get('entry_selection_rule')}")
    print(f"entry_wall_clearance_mm  : {diagnostic.get('entry_outward_wall_clearance_mm')}")
    print(f"rejection_reasons        : {diagnostic.get('rejection_reasons')}")
    print("robot motion             : DISABLED BY M39.4.0.1 CONTRACT")
    print("=" * 78)
    return True


def print_debug_artifacts(data: Mapping[str, Any]) -> None:
    files = data.get("files") if isinstance(data.get("files"), Mapping) else {}
    if not files:
        return
    root = Path("/opt/visionops_v3")
    wanted = [
        ("RGB", "rgb"),
        ("OVERLAY", "overlay"),
        ("DEPTH", "depth_colormap"),
        ("RESULT JSON", "geometry_result"),
        ("ANALYTIC", "m39_3_4_analytic_conic_surface"),
        ("ROUTING", "m39_3_4_1_tilted_production_routing"),
        ("MOUTH TOPO", "m39_4_0_1_mouth_topology_arbitration"),
        ("SIDE AXIS", "m39_4_0_side_axis_recovery"),
        ("SIDE OPEN", "m39_4_1_side_opening_reconstruction"),
        ("SIDE ENTRY", "m39_4_2_side_entry_validation"),
    ]
    rows = []
    for label, key in wanted:
        value = files.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        rows.append((label, path))
    if not rows:
        return
    print("\nDEBUG ARTIFACTS")
    print("-" * 78)
    for label, path in rows:
        print(f"{label:<12}: {path}")
    print("-" * 78)

def _pose_is_near_ready(actual_mm: Sequence[float], ready_pose_mm: Sequence[float]) -> bool:
    return (
        _distance_mm(actual_mm, ready_pose_mm) <= MAX_START_OFFSET_FROM_INITIAL_MM
        and _quat_angle_deg(actual_mm[3:7], ready_pose_mm[3:7]) <= MAX_START_ROTATION_FROM_INITIAL_DEG
    )


def _ready_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "label": "VISIBLE initial",
            "pose_mm": VISIBLE_INITIAL_POSE_MM,
            "joints": VISIBLE_INITIAL_JOINTS,
        },
        {
            "label": "SIDE initial",
            "pose_mm": SIDE_INITIAL_POSE_MM,
            "joints": SIDE_INITIAL_JOINTS,
        },
    ]


def plan_branch_ready_transition(
    robot: RobotClient,
    *,
    target_pose_mm: Sequence[float],
    target_joints: Sequence[float],
    target_label: str,
) -> Dict[str, Any]:
    """Plan a deterministic joint-space branch-ready transition.

    Ready-state identity is joint-primary and TCP-secondary.  This deliberately
    prevents a TCP pose with an unexpected/mirrored/multi-turn joint solution
    from being accepted as a known safe starting configuration.
    """
    actual_pose = current_pose_mm(robot)
    actual_joints = current_joints(robot)
    known_ready = _ready_definitions()

    print(_format_pose("current LEFT_LINK7", actual_pose))
    print(_format_joints("current LEFT joints", actual_joints))
    print(_format_pose(target_label, target_pose_mm))
    print(_format_joints(f"{target_label} J", target_joints))

    target_d = _distance_mm(actual_pose, target_pose_mm)
    target_r = _quat_angle_deg(actual_pose[3:7], target_pose_mm[3:7])
    target_j = _max_joint_error_rad(actual_joints, target_joints)
    print(
        f"target ready offset       : TCP={target_d:.3f} mm / {target_r:.3f} deg; "
        f"joint_max={target_j:.6f} rad / {math.degrees(target_j):.3f} deg"
    )

    matches: List[Dict[str, Any]] = []
    diagnostics: List[str] = []
    for ready in known_ready:
        pose = ready["pose_mm"]
        joints = ready["joints"]
        dt = _distance_mm(actual_pose, pose)
        dr = _quat_angle_deg(actual_pose[3:7], pose[3:7])
        dj = _max_joint_error_rad(actual_joints, joints)
        joint_ok = dj <= MAX_START_JOINT_ERROR_RAD
        tcp_ok = dt <= MAX_START_OFFSET_FROM_INITIAL_MM and dr <= MAX_START_ROTATION_FROM_INITIAL_DEG
        tcp_tight = dt <= VERIFY_TRANSLATION_TOL_MM and dr <= VERIFY_ROTATION_TOL_DEG
        joint_normalizable = dj <= MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD
        diagnostics.append(
            f"{ready['label']}: joint_max={math.degrees(dj):.2f}deg, "
            f"TCP={dt:.1f}mm/{dr:.1f}deg"
        )
        if joint_ok and tcp_ok:
            matches.append(ready)
        elif tcp_tight and joint_normalizable:
            # M39.5.2.2: this is a known READY TCP with a slightly different
            # redundant joint solution. Accept it only as a source for
            # normalization. Later, because it is outside the 3deg tight joint
            # gate, move_j(target frozen joints) is mandatory before grasping.
            print(
                f"ready normalization        : TCP is tightly at {ready['label']}, "
                f"joint_max={math.degrees(dj):.2f}deg; accept for move_j normalization "
                f"(limit={math.degrees(MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD):.2f}deg)"
            )
            matches.append(ready)
        elif tcp_ok and not joint_ok:
            raise RuntimeError(
                f"TCP is near {ready['label']} but the 7-DOF joint configuration is outside the "
                f"safe ready identity/normalization envelope (max joint error={math.degrees(dj):.2f}deg; "
                f"coarse={math.degrees(MAX_START_JOINT_ERROR_RAD):.2f}deg, "
                f"TCP-tight normalize={math.degrees(MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD):.2f}deg). "
                "Refuse automatic routing."
            )
        elif joint_ok and not tcp_ok:
            raise RuntimeError(
                f"joints are near {ready['label']} but TCP verification is inconsistent "
                f"({dt:.1f}mm/{dr:.1f}deg). Refuse automatic routing; check frame/state."
            )

    if len(matches) != 1:
        raise RuntimeError(
            "robot is not in exactly one validated ready configuration; automatic branch routing "
            "is disabled from an unknown/ambiguous start. " + "; ".join(diagnostics)
        )

    source = matches[0]
    if str(source["label"]) == target_label:
        target_joint_tight = target_j <= VERIFY_READY_JOINT_TOL_RAD
        target_tcp_tight = (
            target_d <= VERIFY_TRANSLATION_TOL_MM
            and target_r <= VERIFY_ROTATION_TOL_DEG
        )
        if target_joint_tight and target_tcp_tight:
            print(f"ready routing             : ALREADY at {target_label}; no move_j switch needed")
            return {
                "action": "already_ready",
                "source_label": target_label,
                "target_label": target_label,
                "target_pose_mm": list(target_pose_mm),
                "target_joints": list(target_joints),
            }
        print(
            f"ready routing             : NORMALIZE {target_label} with move_j "
            f"(inside coarse ready gate but outside tight execution tolerance)"
        )
    else:
        print(
            f"ready routing             : AUTO MOVEJ {source['label']} -> {target_label} "
            f"(joint-space frozen target)"
        )
    return {
        "action": "switch_ready",
        "source_label": source["label"],
        "target_label": target_label,
        "target_pose_mm": list(target_pose_mm),
        "target_joints": list(target_joints),
    }


def execute_branch_ready_transition(robot: RobotClient, plan: Mapping[str, Any]) -> None:
    """Execute the planned ready-pose switch with SDK move_j after confirmation."""
    target_label = str(plan.get("target_label") or "branch initial")
    target_pose = plan.get("target_pose_mm")
    target_joints = plan.get("target_joints")
    if not (isinstance(target_pose, list) and len(target_pose) == 7):
        raise RuntimeError("branch ready transition target TCP pose is invalid")
    if not (isinstance(target_joints, list) and len(target_joints) == 7):
        raise RuntimeError("branch ready transition target joints are invalid")

    # Re-read BOTH joint and TCP state after operator confirmation.
    fresh_plan = plan_branch_ready_transition(
        robot,
        target_pose_mm=target_pose,
        target_joints=target_joints,
        target_label=target_label,
    )
    fresh_action = str(fresh_plan.get("action") or "")
    if fresh_action == "already_ready":
        # Even when no move is needed, perform a tighter joint + TCP verification
        # before entering a production grasp branch.
        verify_ready_reached(
            robot,
            target_pose_mm=target_pose,
            target_joints=target_joints,
            label=target_label,
        )
        return
    if fresh_action != "switch_ready":
        raise RuntimeError(f"unexpected branch ready transition action: {fresh_action!r}")

    print(
        f"\n[Branch Ready] vision-selected branch requires {target_label}; "
        f"joint-space switch from {fresh_plan.get('source_label')}"
    )
    set_left_gripper(robot, opened=True, label="AUTO READY MOVEJ SWITCH")
    move_j_ready(
        robot,
        target_pose_mm=target_pose,
        target_joints=target_joints,
        label=f"{target_label.upper()}-AUTO",
        vel=READY_SWITCH_JOINT_VEL,
        acc=READY_SWITCH_JOINT_ACC,
    )


def print_known_start_poses(robot: RobotClient) -> None:
    actual_pose = current_pose_mm(robot)
    actual_joints = current_joints(robot)
    print(_format_pose("current LEFT_LINK7", actual_pose))
    print(_format_joints("current LEFT joints", actual_joints))
    for ready in _ready_definitions():
        label = str(ready["label"])
        pose = ready["pose_mm"]
        joints = ready["joints"]
        dj = _max_joint_error_rad(actual_joints, joints)
        print(_format_pose(label, pose))
        print(_format_joints(f"{label} J", joints))
        print(
            f"  offset -> {label}: joint_max={math.degrees(dj):.2f} deg; "
            f"TCP={_distance_mm(actual_pose, pose):.1f} mm / "
            f"{_quat_angle_deg(actual_pose[3:7], pose[3:7]):.1f} deg"
        )
    print("Vision selects the branch; ready-to-ready switching uses frozen 7-DOF move_j targets.")


def wait_for_confirmation(prompt: str = "Press Enter to confirm robot execution (or 'c' to cancel)") -> bool:
    """Wait for user confirmation. Returns True if confirmed, False if cancelled."""
    while True:
        cmd = input(f"\n{prompt} > ").strip().lower()
        if cmd in {"", "y", "yes", "confirm"}:
            return True
        if cmd in {"c", "cancel", "n", "no"}:
            return False
        print("Please press Enter to confirm, or type 'c' to cancel.")


def normalize_ready_after_cycle(
    robot: RobotClient,
    *,
    target_pose_mm: Sequence[float],
    target_joints: Sequence[float],
    label: str,
) -> None:
    """Leave every completed cycle in the frozen 7-DOF READY identity.

    A Cartesian MoveL back to the same TCP can end in a slightly different
    redundant joint solution.  That makes the next trigger fail the READY
    identity gate even though the TCP looks correct.  At the high/safe READY
    pose, normalize only when needed and verify both joints and TCP.
    """
    actual_pose = current_pose_mm(robot)
    actual_joints = current_joints(robot)
    dt = _distance_mm(actual_pose, target_pose_mm)
    dr = _quat_angle_deg(actual_pose[3:7], target_pose_mm[3:7])
    dj = _max_joint_error_rad(actual_joints, target_joints)
    print(
        f"[Ready Normalize] {label}: TCP={dt:.3f}mm/{dr:.3f}deg, "
        f"joint_max={math.degrees(dj):.3f}deg"
    )
    if dt > VERIFY_TRANSLATION_TOL_MM or dr > VERIFY_ROTATION_TOL_DEG:
        raise RuntimeError(
            f"cannot normalize {label}: return TCP is not tightly at frozen READY "
            f"({dt:.3f}mm/{dr:.3f}deg)"
        )
    if dj <= VERIFY_READY_JOINT_TOL_RAD:
        print(f"✓ {label} already has frozen/tight joint identity")
        return
    if dj > MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD:
        raise RuntimeError(
            f"cannot normalize {label}: TCP is tight but max joint error "
            f"{math.degrees(dj):.3f}deg exceeds normalization envelope "
            f"{math.degrees(MAX_TCP_TIGHT_NORMALIZE_JOINT_ERROR_RAD):.3f}deg"
        )
    move_j_ready(
        robot,
        target_pose_mm=target_pose_mm,
        target_joints=target_joints,
        label=f"{label.upper()}-CYCLE-NORMALIZE",
        vel=READY_SWITCH_JOINT_VEL,
        acc=READY_SWITCH_JOINT_ACC,
    )


def execute_cycle(robot: RobotClient, pre_pose: Sequence[float], grasp_pose: Sequence[float]) -> None:
    # Production validation sequence:
    #   INITIAL/READY --OPEN--> PREGRASP -> GRASP --CLOSE--> PREGRASP -> INITIAL/READY --OPEN
    # The gripper stays OPEN for the entire approach and CLOSED for the entire retreat.

    # 0) At the ready/initial point, force the gripper OPEN before any motion.
    # If this command fails, no robot motion is allowed.
    set_left_gripper(robot, opened=True, label="READY/INITIAL before approach")

    # 1) Initial -> pregrasp, gripper remains OPEN.
    move_l_mm(robot, pre_pose, label="PREGRASP", vel=TRAVEL_VEL, acc=TRAVEL_ACC)

    # 2) Pregrasp -> grasp, still OPEN and deliberately slower.
    try:
        move_l_mm(robot, grasp_pose, label="GRASP", vel=APPROACH_VEL, acc=APPROACH_ACC)
    except Exception:
        print("\n! GRASP motion failed while gripper should still be OPEN.")
        print("! Trying conservative retreat to PREGRASP, then INITIAL; gripper stays OPEN.")
        try:
            move_l_mm(robot, pre_pose, label="PREGRASP-RECOVERY", vel=APPROACH_VEL, acc=APPROACH_ACC)
            move_l_mm(robot, VISIBLE_INITIAL_POSE_MM, label="INITIAL-RECOVERY", vel=RETURN_VEL, acc=RETURN_ACC)
            # Reassert the required ready-state command after recovery.
            set_left_gripper(robot, opened=True, label="READY/INITIAL after recovery")
        except Exception as recovery_error:
            print(f"!! automatic recovery also failed: {recovery_error}")
            print("!! Stop and inspect the robot manually.")
        raise

    # 3) At the final grasp pose, CLOSE the gripper before any retreat motion.
    # If CLOSE fails, do not execute the return path because grasp state is unknown.
    set_left_gripper(robot, opened=False, label="GRASP pose")

    # 4) Keep the gripper CLOSED while retreating: grasp -> pregrasp -> initial.
    # The extra pregrasp retreat avoids a direct diagonal grasp->initial path.
    try:
        move_l_mm(robot, pre_pose, label="PREGRASP-RETURN", vel=APPROACH_VEL, acc=APPROACH_ACC)
        move_l_mm(robot, VISIBLE_INITIAL_POSE_MM, label="INITIAL", vel=RETURN_VEL, acc=RETURN_ACC)
    except Exception:
        print("\n! Return motion failed AFTER gripper CLOSE.")
        print("! Gripper will remain CLOSED; it will NOT be opened away from the ready/initial point.")
        raise

    # 5) Only after the robot is verified back at INITIAL/READY, OPEN the gripper.
    set_left_gripper(robot, opened=True, label="READY/INITIAL cycle complete")

    # M39.5.3: MoveL can return the same TCP using a slightly different
    # redundant 7-DOF solution. Normalize at the safe ready pose so the next
    # trigger starts from the exact frozen VISIBLE joint identity.
    normalize_ready_after_cycle(
        robot,
        target_pose_mm=VISIBLE_INITIAL_POSE_MM,
        target_joints=VISIBLE_INITIAL_JOINTS,
        label="VISIBLE initial",
    )


def execute_side_grasp_cycle(
    robot: RobotClient,
    pre_pose: Sequence[float],
    entry_pose: Sequence[float],
    grasp_pose: Sequence[float],
) -> None:
    """M39.4.2.2 direct side grasp with a short PREGRASP.

    ENTRY is intentionally diagnostic-only.  The executable motion contract is:
      SIDE_INITIAL[OPEN] -> AVOIDANCE[OPEN] -> PREGRASP[OPEN] -> GRASP[OPEN]
      -> CLOSE -> AVOIDANCE[CLOSED] -> SIDE_INITIAL[CLOSED] -> OPEN.

    `entry_pose` is kept in the function signature so the script can continue to
    validate/print the reconstructed opening-plane pose, but no robot move is
    issued to it.
    """
    del entry_pose  # geometric reference only; never a robot stop in M39.4.2.2
    set_left_gripper(robot, opened=True, label="M39.4.2.2 SIDE INITIAL")
    closed = False
    reached_grasp = False
    try:
        move_l_mm(robot, SIDE_COLLISION_AVOIDANCE_POSE_MM, label="SIDE-AVOIDANCE", vel=SIDE_TRANSIT_VEL, acc=SIDE_TRANSIT_ACC)
        move_l_mm(robot, pre_pose, label="SIDE-PREGRASP", vel=SIDE_TRANSIT_VEL, acc=SIDE_TRANSIT_ACC)
        move_l_mm(robot, grasp_pose, label="SIDE-GRASP", vel=SIDE_GRASP_VEL, acc=SIDE_GRASP_ACC)
        reached_grasp = True
        set_left_gripper(robot, opened=False, label="SIDE-GRASP pose")
        closed = True

        # Field requirement: once the part is secured, do not retrace the
        # insertion axis.  Leave the box directly via the fixed avoidance pose.
        move_l_mm(robot, SIDE_COLLISION_AVOIDANCE_POSE_MM, label="SIDE-AVOIDANCE-RETURN", vel=SIDE_TRANSIT_VEL, acc=SIDE_TRANSIT_ACC)
        move_l_mm(robot, SIDE_INITIAL_POSE_MM, label="SIDE-INITIAL", vel=RETURN_VEL, acc=RETURN_ACC)
        set_left_gripper(robot, opened=True, label="SIDE INITIAL cycle complete")
        normalize_ready_after_cycle(
            robot,
            target_pose_mm=SIDE_INITIAL_POSE_MM,
            target_joints=SIDE_INITIAL_JOINTS,
            label="SIDE initial",
        )
    except Exception:
        if closed:
            print("\n! M39.4.2.2 failure AFTER gripper CLOSE.")
            print("! Gripper remains CLOSED. No automatic reverse-to-PREGRASP is attempted.")
            print("! Best effort is direct SIDE-AVOIDANCE -> SIDE_INITIAL only.")
        else:
            print("\n! M39.4.2.2 approach failure while gripper is OPEN.")
            print("! Attempting conservative PREGRASP -> AVOIDANCE -> SIDE INITIAL recovery.")
        try:
            if closed and reached_grasp:
                move_l_mm(robot, SIDE_COLLISION_AVOIDANCE_POSE_MM, label="SIDE-AVOIDANCE-RECOVERY", vel=SIDE_TRANSIT_VEL, acc=SIDE_TRANSIT_ACC)
                move_l_mm(robot, SIDE_INITIAL_POSE_MM, label="SIDE-INITIAL-RECOVERY", vel=RETURN_VEL, acc=RETURN_ACC)
            else:
                move_l_mm(robot, pre_pose, label="SIDE-PREGRASP-RECOVERY", vel=SIDE_ENTRY_VEL, acc=SIDE_ENTRY_ACC)
                move_l_mm(robot, SIDE_COLLISION_AVOIDANCE_POSE_MM, label="SIDE-AVOIDANCE-RECOVERY", vel=SIDE_TRANSIT_VEL, acc=SIDE_TRANSIT_ACC)
                move_l_mm(robot, SIDE_INITIAL_POSE_MM, label="SIDE-INITIAL-RECOVERY", vel=RETURN_VEL, acc=RETURN_ACC)
                set_left_gripper(robot, opened=True, label="SIDE INITIAL after recovery")
        except Exception as recovery_error:
            print(f"!! automatic side recovery failed: {recovery_error}")
            print("!! Stop and inspect the robot manually.")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="M39.5.4 dense-scene-exposed-target LEFT robot motion validation")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually move the robot; without this flag the script only triggers/parses/saves vision results",
    )
    parser.add_argument(
        "--no-save-debug",
        action="store_true",
        help="do not ask the vision service to save debug artifacts",
    )
    parser.add_argument(
        "--single-enter",
        action="store_true",
        help="with --execute, execute immediately after a safe vision result; skips the second Enter confirmation",
    )
    parser.add_argument(
        "--min-tilted-deg",
        type=float,
        default=DEFAULT_MIN_TILTED_ROUTE_DEG,
        help="minimum angle accepted for the explicit TILTED production route",
    )
    parser.add_argument(
        "--max-tilted-deg",
        type=float,
        default=DEFAULT_MAX_TILTED_ROUTE_DEG,
        help="maximum angle accepted for the explicit TILTED production route",
    )
    args = parser.parse_args()
    if args.single_enter and not args.execute:
        parser.error("--single-enter requires --execute")
    if args.min_tilted_deg < 0.0 or args.max_tilted_deg <= args.min_tilted_deg:
        parser.error("invalid tilted angle range")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print("=" * 78)
    print(f"M39.5.2.2 HYBRID READY + JOINT NORMALIZATION + CAMERA-NEAR PRODUCTION [{mode}]")
    print("Enter : trigger vision")
    print("After vision: " + ("safe result executes immediately (--single-enter)" if args.single_enter else "Enter to execute robot motion, 'c' to cancel"))
    print("q     : quit")
    print("Frame : robot_default_base / SDK default frame=None")
    print("Ready switch: SDK move_j with frozen raw 7-DOF joint targets; TCP used only for verification")
    print(f"Gripper: Modbus register {LEFT_GRIPPER_REGISTER}, 1=OPEN, 2=CLOSE")
    print("=" * 78)

    vision_session = make_vision_session()
    check_vision_service(vision_session)
    robot = connect_robot() if args.execute else None
    try:
        if args.execute:
            assert robot is not None
            print_known_start_poses(robot)
        else:
            print("DRY-RUN: robot is not connected and no motion is possible. Use --execute after checking the output.")

        while True:
            try:
                # Step 1: Wait for first Enter to trigger vision
                cmd = input("\nPress Enter to trigger vision (q=quit) > ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    break
                if cmd:
                    print("Only Enter triggers vision; q quits.")
                    continue

                # Step 2: Trigger vision
                print("\n>>> TRIGGERING VISION...")
                data = trigger_vision(vision_session, save_debug=not args.no_save_debug)
                log_path = save_result(data)
                scene_summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), Mapping) else {}
                m3950 = scene_summary.get("m39_5_0_visible_mouth_axis_validation") if isinstance(scene_summary, Mapping) else None
                if isinstance(m3950, Mapping):
                    m3950_class = str(m3950.get("classification") or "UNCERTAIN").upper()
                    m3950_ready = str(m3950.get("recommended_ready_pose") or "NONE")
                    m3950_tilt = m3950.get("axis_tilt_from_box_z_deg")
                    tilt_text = "n/a" if m3950_tilt is None else f"{float(m3950_tilt):.2f} deg"
                    print(
                        f"[M39.5.0] {m3950_class} | axis={tilt_text} | "
                        f"recommended_ready={m3950_ready} | reason={m3950.get('reason')}"
                    )
                    # M39.5.1 owns TILTED_VISIBLE_SIDE and will be handled
                    # immediately below through its camera-near-rim production
                    # summary.  Never execute the historical tilted clock-3 path.
                    m3950_policy = str(m3950.get("production_grasp_policy") or "")
                    if m3950_policy == "VISIBLE_CLOCK3_MILD_TILT":
                        print(
                            f"[M39.5.2] mild tilt <30deg -> VISIBLE_INITIAL + preferred clock-3; "
                            f"measured_axis={tilt_text}"
                        )
                    if m3950_class == "TILTED_VISIBLE_SIDE":
                        near = m3950.get("camera_near_rim") if isinstance(m3950.get("camera_near_rim"), Mapping) else {}
                        print(
                            f"[M39.5.1] tilted visible mouth -> SIDE_INITIAL; "
                            f"axis_reliable={m3950.get('axis_solution_reliable')} | "
                            f"camera_near_rim={near.get('camera_near_rim_midpoint_camera_mm')}"
                        )
                    if m3950_class == "UNCERTAIN":
                        # M39.5.3: M39.5 axis uncertainty is not a blind hard veto
                        # for an otherwise authoritative, already collision-checked
                        # FLAT clock-3 production result.  This case occurs when
                        # conic/ellipse pose inference is weak at some workspace
                        # positions while M38.1 + M39.3.1 still provide a validated
                        # floor-constrained flat grasp.  Let the normal production
                        # route validator below decide.  Any true terminal reject,
                        # missing candidate, or non-FLAT route will still be refused.
                        summary_route = scene_summary.get("m39_3_4_1_tilted_production_routing") if isinstance(scene_summary, Mapping) else None
                        protected_flat = bool(
                            data.get("target_found") is True
                            and data.get("terminal_reject") is not True
                            and isinstance(summary_route, Mapping)
                            and str(summary_route.get("classification") or "").upper() == "FLAT"
                            and str(summary_route.get("route") or "") == EXPECTED_FLAT_ROUTE
                            and not bool(summary_route.get("terminal_reject", False))
                        )
                        if protected_flat:
                            print(
                                "[M39.5.3] axis classification UNCERTAIN, but an authoritative "
                                "collision-checked FLAT clock-3 route exists; continue with "
                                "VISIBLE_INITIAL production validation."
                            )
                        else:
                            print(f"vision JSON saved         : {log_path}")
                            print_debug_artifacts(data)
                            print("M39.5 axis/READY classification is UNCERTAIN and no protected FLAT route exists; robot will NOT move.")
                            continue
                m3951 = scene_summary.get("m39_5_1_tilted_visible_grasp") if isinstance(scene_summary, Mapping) else None
                if isinstance(m3951, Mapping) and bool(m3951.get("executed", False)):
                    if bool(m3951.get("production_grasp_ready", False)):
                        tilted_side_poses = extract_m3951_left_link7_targets(data)
                        print("\n" + "=" * 78)
                        print("M39.5.1 TILTED-VISIBLE CAMERA-NEAR-RIM PRODUCTION GRASP")
                        print("=" * 78)
                        print(f"axis tilt               : {m3951.get('axis_tilt_from_box_z_deg')} deg")
                        near = m3951.get("camera_near_rim") if isinstance(m3951.get("camera_near_rim"), Mapping) else {}
                        print(f"camera-near rim camera  : {near.get('camera_near_rim_midpoint_camera_mm')}")
                        print(f"axis source              : {m3951.get('axis_source')}")
                        print(f"rejection_reasons        : {m3951.get('rejection_reasons') or []}")
                        print(f"LEFT_LINK7 PREGRASP      : {tilted_side_poses[0]}")
                        print(f"LEFT_LINK7 ENTRY(ref)    : {tilted_side_poses[1]}")
                        print(f"LEFT_LINK7 GRASP         : {tilted_side_poses[2]}")
                        print("robot action             : SIDE_INITIAL -> AVOIDANCE -> PREGRASP -> GRASP; CLOSE; AVOIDANCE -> SIDE_INITIAL")
                        print("=" * 78)
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        if not args.execute:
                            print("DRY-RUN complete: M39.5.1 tilted-visible grasp was not executed.")
                            continue
                        assert robot is not None
                        ready_plan = plan_branch_ready_transition(
                            robot,
                            target_pose_mm=SIDE_INITIAL_POSE_MM,
                            target_joints=SIDE_INITIAL_JOINTS,
                            target_label="SIDE initial",
                        )
                        print("\n" + "-" * 60)
                        print("M39.5.1 TILTED-VISIBLE NEXT STEPS:")
                        print("  → If needed, AUTO-MOVEJ to SIDE initial using frozen 7-DOF joints")
                        print("  → OPEN left gripper")
                        print("  → Move to SIDE-AVOIDANCE waypoint")
                        print("  → Move to camera-near-rim PREGRASP")
                        print("  → Direct coaxial PREGRASP → GRASP along recovered 3-D cylinder axis")
                        print("  → CLOSE at GRASP")
                        print("  → Direct GRASP → SIDE-AVOIDANCE → SIDE-INITIAL")
                        print("  → OPEN at SIDE-INITIAL")
                        print("-" * 60)
                        confirmed = wait_for_confirmation("Press Enter for M39.5.1 TILTED-VISIBLE motion, or 'c' to cancel")
                        if not confirmed:
                            print(">> M39.5.1 tilted-visible execution cancelled by operator.")
                            continue
                        execute_branch_ready_transition(robot, ready_plan)
                        execute_side_grasp_cycle(robot, *tilted_side_poses)
                        print("\n✓ M39.5.1 TILTED-VISIBLE GRASP CYCLE COMPLETE")
                        continue
                    print("\nM39.5.1 TILTED-VISIBLE target rejected before robot motion")
                    print(f"status                   : {m3951.get('status')}")
                    print(f"reason                   : {m3951.get('reason')}")
                    print(f"rejection_reasons        : {m3951.get('rejection_reasons') or []}")
                    print(f"vision JSON saved         : {log_path}")
                    print_debug_artifacts(data)
                    continue

                m3942 = scene_summary.get("m39_4_2_side_entry_validation") if isinstance(scene_summary, Mapping) else None
                if isinstance(m3942, Mapping) and bool(m3942.get("executed", False)):
                    if bool(m3942.get("production_grasp_ready", False)):
                        side_poses = extract_m3942_left_link7_targets(data)
                        print_m3942_side_entry_summary(data, side_poses)
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        if not args.execute:
                            print("DRY-RUN complete: M39.4.2.2 side grasp was not executed.")
                            continue
                        assert robot is not None
                        ready_plan = plan_branch_ready_transition(
                            robot,
                            target_pose_mm=SIDE_INITIAL_POSE_MM,
                            target_joints=SIDE_INITIAL_JOINTS,
                            target_label="SIDE initial",
                        )
                        print("\n" + "-" * 60)
                        print("M39.4.2.2 SIDE GRASP NEXT STEPS:")
                        print("  → If needed, AUTO-MOVEJ to SIDE initial using frozen 7-DOF joints (gripper OPEN)")
                        print("  → OPEN left gripper")
                        print("  → Move to SIDE-AVOIDANCE waypoint")
                        print("  → Move to SIDE-PREGRASP")
                        print("  → Direct coaxial move SIDE-PREGRASP → SIDE-GRASP (ENTRY is diagnostic only)")
                        print("  → CLOSE left gripper at SIDE-GRASP")
                        print("  → After CLOSE: direct SIDE-GRASP → SIDE-AVOIDANCE → SIDE-INITIAL")
                        print("  → OPEN left gripper at SIDE-INITIAL")
                        print("-" * 60)
                        # Keep a second Enter for the first production side-grasp tests.
                        confirmed = wait_for_confirmation("Press Enter for SIDE GRASP motion, or 'c' to cancel")
                        if not confirmed:
                            print(">> M39.4.2.2 side-grasp execution cancelled by operator.")
                            continue
                        execute_branch_ready_transition(robot, ready_plan)
                        pre_side, entry_side, grasp_side = side_poses
                        execute_side_grasp_cycle(robot, pre_side, entry_side, grasp_side)
                        print("\n✓ M39.4.2.2 SIDE GRASP CYCLE COMPLETE")
                        continue
                    print_m3942_side_entry_summary(data)
                    print(f"vision JSON saved         : {log_path}")
                    print_debug_artifacts(data)
                    print("M39.4.2.2 rejected this side-grasp candidate; robot will NOT move.")
                    continue

                if data.get("target_found") is not True:
                    if print_m3941_side_opening_summary(data):
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        print("M39.4.1 opening reconstruction did not reach M39.4.2.2 side-grasp routing; robot will NOT move.")
                        continue
                    if print_m3940_side_axis_summary(data):
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        print("M39.4.0.1 side axis is diagnostic-only until M39.4.1/.2 pass.")
                        continue
                route_info = validate_m39341_production_route(
                    data,
                    minimum_tilted_deg=float(args.min_tilted_deg),
                    maximum_tilted_deg=float(args.max_tilted_deg),
                )
                pre_pose, grasp_pose = extract_left_link7_targets(data)
                print_vision_summary(data, pre_pose, grasp_pose, route_info)
                print(f"vision JSON saved         : {log_path}")
                print_debug_artifacts(data)

                # Step 3: Wait for second confirmation before executing robot motion
                if not args.execute:
                    print("DRY-RUN complete: no robot motion performed.")
                    continue

                # Plan the vision-selected branch ready pose.  Being parked at the
                # other known branch ready pose is no longer a rejection condition.
                assert robot is not None
                ready_plan = plan_branch_ready_transition(
                    robot,
                    target_pose_mm=VISIBLE_INITIAL_POSE_MM,
                    target_joints=VISIBLE_INITIAL_JOINTS,
                    target_label="VISIBLE initial",
                )

                # Show what will happen
                print("\n" + "-" * 60)
                print("NEXT STEPS (if confirmed):")
                print("  → If needed, AUTO-MOVEJ to VISIBLE initial using frozen 7-DOF joints (gripper OPEN)")
                print("  → OPEN left gripper")
                print("  → Move to PREGRASP pose")
                print("  → Move to GRASP pose")
                print("  → CLOSE left gripper")
                print("  → Return to PREGRASP pose")
                print("  → Return to INITIAL pose")
                print("  → OPEN left gripper")
                print("-" * 60)

                # Wait for confirmation unless explicitly using the field-test single-enter mode.
                if args.single_enter:
                    confirmed = True
                    print("\n--single-enter active: safe routed result passed all script checks; executing now.")
                else:
                    confirmed = wait_for_confirmation("Press Enter to execute, or 'c' to cancel")
                if not confirmed:
                    print(">> Robot execution cancelled by operator.")
                    print("   Robot remains at current position.")
                    continue

                # Step 4: Route to the vision-selected ready pose, then execute.
                execute_branch_ready_transition(robot, ready_plan)
                print("\n>>> EXECUTING ROBOT MOTION...")
                print(
                    "Motion sequence: "
                    "INITIAL[OPEN] -> PREGRASP[OPEN] -> GRASP -> CLOSE -> "
                    "PREGRASP[CLOSED] -> INITIAL[CLOSED] -> OPEN"
                )
                execute_cycle(robot, pre_pose, grasp_pose)
                print("\n✓ FULL CYCLE COMPLETE")

            except KeyboardInterrupt:
                print("\n\nKeyboardInterrupt: issuing LEFT-arm fast stop...")
                try:
                    if robot is not None:
                        robot.motion.fast_stop(ArmType.Left)
                    print("✓ fast_stop sent")
                except Exception as stop_error:
                    print(f"! fast_stop failed: {stop_error}")
                print("Robot will NOT auto-return after an operator interrupt.")
                break
            except Exception as error:
                print(f"\n✗ CYCLE ABORTED: {error}")
                print("No next motion step will be executed. Inspect the cause before retrying.")

    finally:
        try:
            if robot is not None:
                robot.disconnect()
        except Exception:
            pass
        print("Robot SDK disconnected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())