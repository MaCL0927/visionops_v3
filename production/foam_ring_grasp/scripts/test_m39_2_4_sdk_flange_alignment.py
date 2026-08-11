#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M39.2.4 SDK flange / hand_tcp fixed-transform calibration helper.

This is a TEST-ONLY utility. It does not modify production line.yaml and does
not change the online RobotPoseTransformer. Its purpose is to estimate the
fixed rigid transform

    T_hand_tcp_sdk_flange
      = inv(T_base_hand_tcp_visual) @ T_base_sdk_flange_actual

where:
  * T_base_hand_tcp_visual comes from the online M38.6/M39.2 visual result;
  * T_base_sdk_flange_actual is the LEFT flange pose manually copied from
    robot.motion.get_pose(ArmType.Left), expressed in robot_default_base.

For calibration collection, use only one visual clock direction (default 1) so
clock-dependent grasp-frame effects are not mixed into the fixed tool transform.
After 8-12 samples, the script writes a fitted transform JSON. A later run may
load that JSON with --trial-transform to report the corrected SDK-flange pose
and validation residual without changing production code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_URL = "http://127.0.0.1:19213/api/foam_ring/infer_once"
DEFAULT_OUT_DIR = "m39_2_4_sdk_flange_alignment"
DEFAULT_EXPECTED_BASE = "robot_default_base"
DEFAULT_EXPECTED_METHOD = "PARK"
DEFAULT_EXPECTED_QUALITY = "PASS"
DEFAULT_EXPECTED_SAMPLES = 24
DEFAULT_CLOCK_HOUR = 1


def post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc}") from exc
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Response is not valid JSON: {raw[:500]!r}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Response JSON is not an object")
    return obj


def find_robot_transform(data: Mapping[str, Any]) -> Dict[str, Any]:
    rt = data.get("robot_pose_transform")
    if isinstance(rt, dict):
        return rt
    for key in ("result", "data", "output"):
        child = data.get(key)
        if isinstance(child, dict):
            rt = child.get("robot_pose_transform")
            if isinstance(rt, dict):
                return rt
    raise RuntimeError("robot_pose_transform not found in infer_once response")


def selected_clock_hour(data: Mapping[str, Any]) -> Optional[int]:
    scene = data.get("scene_summary")
    value = None
    if isinstance(scene, Mapping):
        value = scene.get("selected_clock_hour")
    if value is None:
        value = data.get("selected_clock_hour")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def validate_online_contract(
    rt: Mapping[str, Any],
    *,
    expected_base: str,
    expected_method: str,
    expected_quality: str,
    expected_samples: int,
) -> None:
    errors: List[str] = []
    if str(rt.get("arm") or "").lower() != "left":
        errors.append(f"arm={rt.get('arm')!r}, expected 'left'")
    if str(rt.get("base_frame_id") or "") != expected_base:
        errors.append(
            f"base_frame_id={rt.get('base_frame_id')!r}, expected {expected_base!r}"
        )
    cal = rt.get("calibration")
    if not isinstance(cal, Mapping):
        errors.append("robot_pose_transform.calibration missing")
    else:
        if str(cal.get("selected_method") or "") != expected_method:
            errors.append(
                f"selected_method={cal.get('selected_method')!r}, expected {expected_method!r}"
            )
        if str(cal.get("quality_status") or "") != expected_quality:
            errors.append(
                f"quality_status={cal.get('quality_status')!r}, expected {expected_quality!r}"
            )
        try:
            actual_n = int(cal.get("sample_count_used") or 0)
        except Exception:
            actual_n = -1
        if actual_n != int(expected_samples):
            errors.append(f"sample_count_used={actual_n}, expected {expected_samples}")
    if errors:
        raise RuntimeError(
            "在线服务不是当前 M39.2.4 允许的 24组PARK/robot_default_base 配置，拒绝采样:\n  - "
            + "\n  - ".join(errors)
        )


def pose7_from_pose_dict(pose: Mapping[str, Any]) -> List[float]:
    pos = pose.get("position_mm")
    quat = pose.get("quaternion_xyzw")
    if not (isinstance(pos, list) and len(pos) == 3):
        raise RuntimeError("pose.position_mm is missing or invalid")
    if not (isinstance(quat, list) and len(quat) == 4):
        raise RuntimeError("pose.quaternion_xyzw is missing or invalid")
    return [float(v) for v in list(pos) + list(quat)]


def extract_hand_tcp_grasp(rt: Mapping[str, Any]) -> List[float]:
    hand = rt.get("hand_tcp")
    if not isinstance(hand, Mapping) or hand.get("available") is not True:
        raise RuntimeError("online hand_tcp pose is unavailable")
    pose = hand.get("grasp_pose_base")
    if not isinstance(pose, Mapping):
        raise RuntimeError("hand_tcp.grasp_pose_base missing")
    return pose7_from_pose_dict(pose)


def extract_legacy_flange_grasp(rt: Mapping[str, Any]) -> Optional[List[float]]:
    flange = rt.get("flange")
    if not isinstance(flange, Mapping) or flange.get("available") is not True:
        return None
    grasp = flange.get("grasp")
    if not isinstance(grasp, Mapping):
        return None
    pose = grasp.get("pose_base")
    if not isinstance(pose, Mapping):
        return None
    return pose7_from_pose_dict(pose)


def normalize_quat_xyzw(q: Sequence[float]) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        raise ValueError("zero quaternion")
    return arr / n


def quat_xyzw_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return normalize_quat_xyzw([qx, qy, qz, qw])


def pose7_to_T(p: Sequence[float]) -> np.ndarray:
    if len(p) != 7:
        raise ValueError("pose must contain 7 values")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R(p[3:7])
    T[:3, 3] = np.asarray(p[:3], dtype=np.float64)
    return T


def T_to_pose7(T: np.ndarray) -> List[float]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    q = R_to_quat_xyzw(T[:3, :3])
    return [float(v) for v in T[:3, 3]] + [float(v) for v in q]


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def rotation_distance_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = np.asarray(Ra).T @ np.asarray(Rb)
    c = float((np.trace(dR) - 1.0) * 0.5)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def pose_error(pred: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(pred[:3], dtype=np.float64)
    a = np.asarray(actual[:3], dtype=np.float64)
    d = a - p
    q1 = normalize_quat_xyzw(pred[3:7])
    q2 = normalize_quat_xyzw(actual[3:7])
    dot = abs(float(np.dot(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return {
        "dx_mm": float(d[0]),
        "dy_mm": float(d[1]),
        "dz_mm": float(d[2]),
        "position_error_mm": float(np.linalg.norm(d)),
        "orientation_error_deg": math.degrees(2.0 * math.acos(dot)),
    }


def parse_actual_pose(text: str, unit: str) -> List[float]:
    clean = text.replace(",", " ").replace("[", " ").replace("]", " ")
    parts = [x for x in clean.split() if x]
    if len(parts) != 7:
        raise ValueError("Need exactly 7 numbers: X Y Z QX QY QZ QW")
    vals = [float(x) for x in parts]
    if unit == "m":
        vals[0] *= 1000.0
        vals[1] *= 1000.0
        vals[2] *= 1000.0
    vals[3:7] = [float(v) for v in normalize_quat_xyzw(vals[3:7])]
    return vals


def fmt_pose(p: Sequence[float]) -> str:
    return (
        f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, "
        f"{p[3]:.6f}, {p[4]:.6f}, {p[5]:.6f}, {p[6]:.6f}]"
    )


def fmt_matrix(T: np.ndarray) -> str:
    return "\n".join(
        "[" + "  ".join(f"{float(v): .8f}" for v in row) + "]"
        for row in np.asarray(T, dtype=np.float64)
    )


def save_raw(out_dir: Path, request_id: str, data: Mapping[str, Any]) -> Path:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in request_id)
    if not safe:
        safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = raw_dir / f"{safe}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    fields = [
        "timestamp", "request_id", "clock_hour", "grasp_branch",
        "hand_x_mm", "hand_y_mm", "hand_z_mm", "hand_qx", "hand_qy", "hand_qz", "hand_qw",
        "actual_x_mm", "actual_y_mm", "actual_z_mm", "actual_qx", "actual_qy", "actual_qz", "actual_qw",
        "tool_tx_mm", "tool_ty_mm", "tool_tz_mm", "tool_qx", "tool_qy", "tool_qz", "tool_qw",
        "legacy_position_error_mm", "legacy_orientation_error_deg",
        "trial_position_error_mm", "trial_orientation_error_deg",
        "raw_json",
    ]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def load_records(path: Path, clock_hour: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if clock_hour is not None and int(row.get("clock_hour", -999)) != clock_hour:
            continue
        matrix = row.get("T_hand_tcp_sdk_flange_mm")
        try:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
                continue
        except Exception:
            continue
        rows.append(row)
    return rows


def average_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("no rotations")
    A = np.zeros((4, 4), dtype=np.float64)
    for R in rotations:
        q = R_to_quat_xyzw(R)
        # q and -q produce the same outer product, so no sign alignment is needed.
        A += np.outer(q, q)
    values, vectors = np.linalg.eigh(A)
    q = vectors[:, int(np.argmax(values))]
    q = normalize_quat_xyzw(q)
    return quat_xyzw_to_R(q)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fit_transform(records: Sequence[Mapping[str, Any]], clock_hour: int) -> Dict[str, Any]:
    transforms = [
        np.asarray(r["T_hand_tcp_sdk_flange_mm"], dtype=np.float64).reshape(4, 4)
        for r in records
    ]
    if not transforms:
        raise ValueError("no valid transforms")
    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    # Median translation is intentionally robust to a few millimetres of manual alignment error.
    t_fit = np.median(translations, axis=0)
    R_fit = average_rotation([T[:3, :3] for T in transforms])
    T_fit = np.eye(4, dtype=np.float64)
    T_fit[:3, :3] = R_fit
    T_fit[:3, 3] = t_fit

    t_dev = [float(np.linalg.norm(T[:3, 3] - t_fit)) for T in transforms]
    r_dev = [rotation_distance_deg(R_fit, T[:3, :3]) for T in transforms]
    q_fit = R_to_quat_xyzw(R_fit)
    return {
        "schema_version": "1.0",
        "stage": "M39.2.4_sdk_flange_alignment_fit",
        "status": "TEST_ONLY_NOT_FOR_PRODUCTION",
        "clock_hour": int(clock_hour),
        "sample_count": len(transforms),
        "parent_frame": "hand_tcp_link",
        "child_frame": "sdk_left_flange",
        "from_frame": "sdk_left_flange",
        "to_frame": "hand_tcp_link",
        "transform_direction": "T_hand_tcp_sdk_flange maps SDK flange coordinates into hand_tcp coordinates; compose as T_base_sdk_flange = T_base_hand_tcp @ T_hand_tcp_sdk_flange",
        "T_hand_tcp_sdk_flange_mm": T_fit.tolist(),
        "translation_mm": t_fit.tolist(),
        "quaternion_xyzw": q_fit.tolist(),
        "translation_component_mean_mm": np.mean(translations, axis=0).tolist(),
        "translation_component_std_mm": np.std(translations, axis=0, ddof=0).tolist(),
        "fit_residuals": {
            "translation_mean_mm": float(np.mean(t_dev)),
            "translation_p95_mm": percentile(t_dev, 95),
            "translation_max_mm": float(np.max(t_dev)),
            "rotation_mean_deg": float(np.mean(r_dev)),
            "rotation_p95_deg": percentile(r_dev, 95),
            "rotation_max_deg": float(np.max(r_dev)),
        },
        "sample_request_ids": [str(r.get("request_id", "")) for r in records],
        "notes": [
            "This fit is generated from manual SDK-flange alignment samples and is test-only.",
            "Do not copy it into production hand_tcp_to_flange until an independent validation set passes.",
            "Calibration samples should use one clock direction only; validate clock 2/3 separately later.",
        ],
    }


def write_fit(records_path: Path, fit_path: Path, clock_hour: int, min_samples: int) -> Optional[Dict[str, Any]]:
    records = load_records(records_path, clock_hour=clock_hour)
    if not records:
        print("[FIT] no valid samples yet")
        return None
    fit = fit_transform(records, clock_hour)
    fit["recommended_minimum_samples"] = int(min_samples)
    fit["sample_count_ready"] = bool(len(records) >= int(min_samples))
    fit_path.write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r = fit["fit_residuals"]
    print("\n--- M39.2.4 running fixed-transform fit ---")
    print(f"samples       : {fit['sample_count']} / recommended >= {min_samples}")
    print(f"clock         : {clock_hour}")
    print(
        "translation : "
        f"[{fit['translation_mm'][0]:.2f}, {fit['translation_mm'][1]:.2f}, {fit['translation_mm'][2]:.2f}] mm"
    )
    q = fit["quaternion_xyzw"]
    print(f"quaternion    : [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(
        f"fit spread    : T mean/p95/max={r['translation_mean_mm']:.2f}/"
        f"{r['translation_p95_mm']:.2f}/{r['translation_max_mm']:.2f} mm, "
        f"R mean/p95/max={r['rotation_mean_deg']:.2f}/"
        f"{r['rotation_p95_deg']:.2f}/{r['rotation_max_deg']:.2f} deg"
    )
    print("T_hand_tcp_sdk_flange_mm =")
    print(fmt_matrix(np.asarray(fit["T_hand_tcp_sdk_flange_mm"], dtype=np.float64)))
    print(f"fit JSON      : {fit_path}")
    if len(records) < int(min_samples):
        print("[FIT] sample count is still insufficient for an independent trial transform.")
    else:
        print("[FIT] enough samples for REVIEW; next use a separate validation set with --trial-transform.")
    return fit


def load_trial_transform(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    matrix = obj.get("T_hand_tcp_sdk_flange_mm") if isinstance(obj, Mapping) else None
    T = np.asarray(matrix, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise RuntimeError(f"invalid trial transform in {p}")
    return T


def run_one(args: argparse.Namespace, out_dir: Path, records_path: Path, csv_path: Path, fit_path: Path) -> bool:
    data = post_json(
        args.url,
        {
            "wait": True,
            "timeout_ms": int(args.infer_timeout_ms),
            "save_debug": bool(args.save_debug),
        },
        timeout_s=float(args.http_timeout_s),
    )
    request_id = str(data.get("request_id") or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    raw_path = save_raw(out_dir, request_id, data)

    if data.get("status") != "ok":
        raise RuntimeError(f"infer_once status={data.get('status')!r}; raw={raw_path}")
    if data.get("target_found") is not True:
        print(f"[SKIP] target_found != true; raw={raw_path}")
        return False
    if data.get("robot_pose_transform_ready") is not True:
        raise RuntimeError(f"robot_pose_transform_ready != true; raw={raw_path}")

    rt = find_robot_transform(data)
    validate_online_contract(
        rt,
        expected_base=args.expected_base_frame,
        expected_method=args.expected_method,
        expected_quality=args.expected_quality,
        expected_samples=args.expected_samples,
    )
    clock = selected_clock_hour(data)
    branch = data.get("selected_grasp_branch") or rt.get("grasp_branch")
    hand_pose = extract_hand_tcp_grasp(rt)
    legacy_flange = extract_legacy_flange_grasp(rt)

    print("\n" + "=" * 92)
    print(f"request_id       : {request_id}")
    print(f"clock            : {clock} (required calibration clock={args.clock_hour})")
    print(f"branch           : {branch}")
    print(f"base             : {rt.get('base_frame_id')}")
    cal = rt.get("calibration") or {}
    print(
        "handeye          : "
        f"{cal.get('selected_method')} / {cal.get('quality_status')} / {cal.get('sample_count_used')} samples"
    )
    print("HAND_TCP visual   : " + fmt_pose(hand_pose))
    if legacy_flange is not None:
        print("LEGACY flange est.: " + fmt_pose(legacy_flange))
        print("  NOTE: legacy flange uses current line.yaml hand_tcp_to_flange and is NOT the M39.2.4 fit target.")
    print(f"raw JSON         : {raw_path}")

    if clock != int(args.clock_hour):
        print(
            f"[SKIP] selected clock={clock}, but fixed-transform calibration is locked to clock={args.clock_hour}. "
            "Do not enter robot pose; trigger again."
        )
        return False

    T_base_hand = pose7_to_T(hand_pose)
    trial_T = load_trial_transform(args.trial_transform)
    trial_pose: Optional[List[float]] = None
    if trial_T is not None:
        trial_pose = T_to_pose7(T_base_hand @ trial_T)
        print("TRIAL SDK flange : " + fmt_pose(trial_pose))
        print("  This is test-only and is NOT written into production online output.")

    if args.no_input:
        return True

    print("\n将机器人手动移动到你认为正确的抓取姿态后，读取:")
    print("  robot.motion.get_pose(ArmType.Left)")
    print("它应表示 SDK 左臂末端法兰 pose，参考坐标为 robot_default_base。")
    print(f"XYZ 输入单位: {args.planner_unit}; 四元数顺序 QX QY QZ QW")
    print("输入 7 个数；直接回车=跳过；q=退出")
    text = input("SDK flange pose > ").strip()
    if not text:
        return True
    if text.lower() in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    actual_pose = parse_actual_pose(text, args.planner_unit)
    T_base_sdk = pose7_to_T(actual_pose)
    T_hand_sdk = invert_T(T_base_hand) @ T_base_sdk
    tool_pose = T_to_pose7(T_hand_sdk)

    legacy_err: Optional[Dict[str, float]] = None
    if legacy_flange is not None:
        legacy_err = pose_error(legacy_flange, actual_pose)
    trial_err: Optional[Dict[str, float]] = None
    if trial_pose is not None:
        trial_err = pose_error(trial_pose, actual_pose)

    print("\n--- 单样本反推固定工具变换 ---")
    print("actual SDK flange : " + fmt_pose(actual_pose))
    print(
        "tool translation  : "
        f"[{T_hand_sdk[0,3]:+.3f}, {T_hand_sdk[1,3]:+.3f}, {T_hand_sdk[2,3]:+.3f}] mm"
    )
    q = tool_pose[3:7]
    print(f"tool quaternion   : [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
    print("T_hand_tcp_sdk_flange_mm =")
    print(fmt_matrix(T_hand_sdk))
    if legacy_err is not None:
        print(
            f"legacy error      : {legacy_err['position_error_mm']:.2f} mm / "
            f"{legacy_err['orientation_error_deg']:.2f} deg"
        )
    if trial_err is not None:
        print(
            f"trial error       : {trial_err['position_error_mm']:.2f} mm / "
            f"{trial_err['orientation_error_deg']:.2f} deg"
        )

    record: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.2.4_sdk_flange_alignment_sample",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id,
        "clock_hour": int(clock),
        "grasp_branch": branch,
        "base_frame_id": str(rt.get("base_frame_id") or ""),
        "handeye": {
            "selected_method": cal.get("selected_method"),
            "quality_status": cal.get("quality_status"),
            "sample_count_used": cal.get("sample_count_used"),
        },
        "T_base_hand_tcp_visual_mm": T_base_hand.tolist(),
        "hand_tcp_visual_pose": hand_pose,
        "T_base_sdk_flange_actual_mm": T_base_sdk.tolist(),
        "sdk_flange_actual_pose": actual_pose,
        "T_hand_tcp_sdk_flange_mm": T_hand_sdk.tolist(),
        "tool_transform_pose": tool_pose,
        "legacy_flange_pose": legacy_flange,
        "legacy_error": legacy_err,
        "trial_transform_file": str(Path(args.trial_transform).expanduser().resolve()) if args.trial_transform else None,
        "trial_sdk_flange_pose": trial_pose,
        "trial_error": trial_err,
        "raw_json": str(raw_path),
    }
    append_jsonl(records_path, record)

    csv_row: Dict[str, Any] = {
        "timestamp": record["timestamp"],
        "request_id": request_id,
        "clock_hour": clock,
        "grasp_branch": branch,
        "raw_json": str(raw_path),
    }
    for prefix, pose in (("hand", hand_pose), ("actual", actual_pose)):
        for key, val in zip(("x_mm", "y_mm", "z_mm", "qx", "qy", "qz", "qw"), pose):
            csv_row[f"{prefix}_{key}"] = val
    for key, val in zip(("tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"), tool_pose):
        csv_row[f"tool_{key}"] = val
    if legacy_err is not None:
        csv_row["legacy_position_error_mm"] = legacy_err["position_error_mm"]
        csv_row["legacy_orientation_error_deg"] = legacy_err["orientation_error_deg"]
    if trial_err is not None:
        csv_row["trial_position_error_mm"] = trial_err["position_error_mm"]
        csv_row["trial_orientation_error_deg"] = trial_err["orientation_error_deg"]
    append_csv(csv_path, csv_row)
    print(f"[OK] sample appended: {records_path}")
    write_fit(records_path, fit_path, int(args.clock_hour), int(args.min_fit_samples))
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M39.2.4 test-only hand_tcp -> SDK left flange fixed-transform calibration"
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--clock-hour", type=int, default=DEFAULT_CLOCK_HOUR,
                   help="accept only this M38.6 clock direction for calibration (default: 1)")
    p.add_argument("--planner-unit", choices=("m", "mm"), default="m")
    p.add_argument("--expected-base-frame", default=DEFAULT_EXPECTED_BASE)
    p.add_argument("--expected-method", default=DEFAULT_EXPECTED_METHOD)
    p.add_argument("--expected-quality", default=DEFAULT_EXPECTED_QUALITY)
    p.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES)
    p.add_argument("--min-fit-samples", type=int, default=8)
    p.add_argument("--trial-transform", default=None,
                   help="optional prior fit JSON; prints corrected SDK flange prediction for independent validation")
    p.add_argument("--infer-timeout-ms", type=int, default=30000)
    p.add_argument("--http-timeout-s", type=float, default=40.0)
    p.add_argument("--save-debug", action="store_true", default=True)
    p.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-input", action="store_true")
    p.add_argument("--fit-only", action="store_true",
                   help="do not trigger inference; recompute fit from existing samples.jsonl")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "m39_2_4_samples.jsonl"
    csv_path = out_dir / "m39_2_4_samples.csv"
    fit_path = out_dir / "m39_2_4_sdk_flange_fit.json"

    print("M39.2.4 SDK flange alignment TEST (LEFT arm)")
    print(f"infer URL       : {args.url}")
    print(f"calibration lock: clock={args.clock_hour}")
    print(
        "online contract : "
        f"base={args.expected_base_frame}, method={args.expected_method}, "
        f"quality={args.expected_quality}, samples={args.expected_samples}"
    )
    print(f"planner unit    : {args.planner_unit}")
    print(f"output dir      : {out_dir}")
    print("IMPORTANT: production hand_tcp_to_flange is NOT modified by this script.")

    if args.fit_only:
        write_fit(records_path, fit_path, int(args.clock_hour), int(args.min_fit_samples))
        return 0

    try:
        while True:
            if not args.once:
                cmd = input("\nEnter=trigger; f=show current fit; q=quit > ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    break
                if cmd == "f":
                    write_fit(records_path, fit_path, int(args.clock_hour), int(args.min_fit_samples))
                    continue
            try:
                run_one(args, out_dir, records_path, csv_path, fit_path)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
            if args.once:
                break
    except KeyboardInterrupt:
        pass

    print("\nDone.")
    if records_path.exists():
        print(f"samples JSONL: {records_path}")
        print(f"samples CSV  : {csv_path}")
        write_fit(records_path, fit_path, int(args.clock_hour), int(args.min_fit_samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
