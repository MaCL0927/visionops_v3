#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foam-ring production validation (M39.3.4.1 visible-mouth baseline):
Enter -> vision trigger -> M39.3.4.1 surface-route verification -> [optional confirm]
-> open left gripper -> LEFT_LINK7 pregrasp -> LEFT_LINK7 grasp -> close left gripper
-> retract -> initial/ready pose -> open left gripper.

The script refuses motion when a TILTED/UNCERTAIN result silently falls back to
the old floor-parallel route. Use --single-enter only after repeated dry-run and
confirmed tilted-pose validation.

IMPORTANT FRAME CONTRACT
------------------------
M39.3.4.1 outputs LEFT_LINK7 poses in `robot_default_base`, which was calibrated
against robot.motion.get_pose(ArmType.Left) with the SDK frame argument omitted.
Therefore all move_l/get_pose calls in this script deliberately use frame=None.
Do NOT change them to frame="base" unless the vision calibration/output frame is
also changed to SDK base_link.

Default mode is DRY RUN. Start with --execute only after dry-run output is sane.
"""

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

# User-provided initial LEFT_LINK7 pose in SDK default / robot_default_base.
INITIAL_POSE_MM = [
    438.5426645162859,
    505.62191497575617,
    882.1935751476652,
    0.5042044184726129,
    0.48014826760863427,
    -0.5025941271032957,
    0.5124789643550238
]

# Conservative first-test motion parameters. Adjust only after validation.
TRAVEL_VEL = 10
TRAVEL_ACC = 30
APPROACH_VEL = 5
APPROACH_ACC = 20
RETURN_VEL = 10
RETURN_ACC = 30
PLANNER = "pilz"

# Sanity checks. They catch unit/frame/extrinsic explosions; they are not a
# replacement for the robot controller's own reachability/collision checking.
MAX_ABS_POSITION_M = 2.0
MAX_INITIAL_TO_PREGRASP_MM = 1000.0
MAX_PREGRASP_TO_GRASP_MM = 250.0
MIN_PREGRASP_TO_GRASP_MM = 10.0
MAX_START_OFFSET_FROM_INITIAL_MM = 80.0
MAX_START_ROTATION_FROM_INITIAL_DEG = 15.0
VERIFY_TRANSLATION_TOL_MM = 8.0
VERIFY_ROTATION_TOL_DEG = 5.0

VISION_TIMEOUT_S = 30.0
VISION_WAIT_TIMEOUT_MS = 25000
LOG_DIR = Path("foam_ring_robot_validation_logs")
EXPECTED_TILTED_ROUTE = "M39.3.4.1_TILTED"
EXPECTED_FLAT_ROUTE = "M39.2.9_FLAT"
DEFAULT_MIN_TILTED_ROUTE_DEG = 8.0
DEFAULT_MAX_TILTED_ROUTE_DEG = 35.0
MAX_PREGRASP_GRASP_ROTATION_DELTA_DEG = 1.0


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
    home_to_pre = _distance_mm(INITIAL_POSE_MM, pre_pose)
    if home_to_pre > MAX_INITIAL_TO_PREGRASP_MM:
        raise RuntimeError(
            f"initial->pregrasp distance abnormal: {home_to_pre:.1f} mm > "
            f"{MAX_INITIAL_TO_PREGRASP_MM:.1f} mm"
        )
    return pre_pose, grasp_pose


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

def ensure_start_near_initial(robot: RobotClient) -> None:
    actual = current_pose_mm(robot)
    d = _distance_mm(actual, INITIAL_POSE_MM)
    r = _quat_angle_deg(actual[3:7], INITIAL_POSE_MM[3:7])
    print(_format_pose("current LEFT_LINK7", actual))
    print(_format_pose("configured initial", INITIAL_POSE_MM))
    print(f"start offset              : {d:.3f} mm / {r:.3f} deg")
    if d > MAX_START_OFFSET_FROM_INITIAL_MM or r > MAX_START_ROTATION_FROM_INITIAL_DEG:
        raise RuntimeError(
            f"robot start pose differs from configured initial pose by {d:.1f} mm / {r:.1f} deg; "
            f"limits are {MAX_START_OFFSET_FROM_INITIAL_MM:.1f} mm / "
            f"{MAX_START_ROTATION_FROM_INITIAL_DEG:.1f} deg. Move it near the initial pose first."
        )


def wait_for_confirmation(prompt: str = "Press Enter to confirm robot execution (or 'c' to cancel)") -> bool:
    """Wait for user confirmation. Returns True if confirmed, False if cancelled."""
    while True:
        cmd = input(f"\n{prompt} > ").strip().lower()
        if cmd in {"", "y", "yes", "confirm"}:
            return True
        if cmd in {"c", "cancel", "n", "no"}:
            return False
        print("Please press Enter to confirm, or type 'c' to cancel.")


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
            move_l_mm(robot, INITIAL_POSE_MM, label="INITIAL-RECOVERY", vel=RETURN_VEL, acc=RETURN_ACC)
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
        move_l_mm(robot, INITIAL_POSE_MM, label="INITIAL", vel=RETURN_VEL, acc=RETURN_ACC)
    except Exception:
        print("\n! Return motion failed AFTER gripper CLOSE.")
        print("! Gripper will remain CLOSED; it will NOT be opened away from the ready/initial point.")
        raise

    # 5) Only after the robot is verified back at INITIAL/READY, OPEN the gripper.
    set_left_gripper(robot, opened=True, label="READY/INITIAL cycle complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="M39.4.1 side-opening frame validation + visible-mouth LEFT robot motion validation")
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
    print(f"M39.4.1 SIDE OPENING FRAME + VISIBLE-MOUTH PRODUCTION VALIDATION [{mode}]")
    print("Enter : trigger vision")
    print("After vision: " + ("safe result executes immediately (--single-enter)" if args.single_enter else "Enter to execute robot motion, 'c' to cancel"))
    print("q     : quit")
    print("Frame : robot_default_base / SDK default frame=None")
    print(f"Gripper: Modbus register {LEFT_GRIPPER_REGISTER}, 1=OPEN, 2=CLOSE")
    print("=" * 78)

    vision_session = make_vision_session()
    check_vision_service(vision_session)
    robot = connect_robot() if args.execute else None
    try:
        if args.execute:
            assert robot is not None
            ensure_start_near_initial(robot)
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
                if data.get("target_found") is not True:
                    if print_m3941_side_opening_summary(data):
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        print("M39.4.1 is validation-only: this side-lying target will NOT move the robot.")
                        continue
                    if print_m3940_side_axis_summary(data):
                        print(f"vision JSON saved         : {log_path}")
                        print_debug_artifacts(data)
                        print("M39.4.0.1 is diagnostic-only: this side-lying target will NOT move the robot.")
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

                # Check start state before waiting for confirmation (so user sees if it's valid)
                assert robot is not None
                ensure_start_near_initial(robot)

                # Show what will happen
                print("\n" + "-" * 60)
                print("NEXT STEPS (if confirmed):")
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

                # Step 4: Execute robot motion
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