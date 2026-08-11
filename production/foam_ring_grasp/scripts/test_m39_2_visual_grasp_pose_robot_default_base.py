#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M39.2/M39.2.1 visual grasp pose manual verification helper.

Purpose
-------
On RK3576, trigger one foam-ring inference, directly print the visual-recommended
LEFT-arm grasp pose in the robot planning base frame, and optionally record the
pose manually read from the robot planner for later multi-sample comparison.

Only Python standard library is required.

Default comparison target is left_link7 (flange), because this is the pose used
in the current manual comparison. The script ALWAYS prints both hand_tcp_link
and left_link7 so frame confusion is visible immediately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_URL = "http://127.0.0.1:19213/api/foam_ring/infer_once"
DEFAULT_OUT_DIR = "m39_2_pose_samples"

POSE_HEADERS = ["x_mm", "y_mm", "z_mm", "qx", "qy", "qz", "qw"]


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


def find_robot_transform(data: Dict[str, Any]) -> Dict[str, Any]:
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


def pose7_from_pose_dict(pose: Dict[str, Any]) -> List[float]:
    pos = pose.get("position_mm")
    quat = pose.get("quaternion_xyzw")
    if not (isinstance(pos, list) and len(pos) == 3):
        raise RuntimeError("pose.position_mm is missing or invalid")
    if not (isinstance(quat, list) and len(quat) == 4):
        raise RuntimeError("pose.quaternion_xyzw is missing or invalid")
    return [float(v) for v in pos + quat]


def extract_poses(rt: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}

    hand = rt.get("hand_tcp")
    if isinstance(hand, dict) and hand.get("available") is True:
        g = hand.get("grasp_pose_base")
        p = hand.get("pregrasp_pose_base")
        if isinstance(g, dict):
            out.setdefault("hand_tcp", {})["grasp"] = pose7_from_pose_dict(g)
        if isinstance(p, dict):
            out.setdefault("hand_tcp", {})["pregrasp"] = pose7_from_pose_dict(p)

    flange = rt.get("flange")
    if isinstance(flange, dict) and flange.get("available") is True:
        g = flange.get("grasp")
        p = flange.get("pregrasp")
        if isinstance(g, dict) and isinstance(g.get("pose_base"), dict):
            out.setdefault("flange", {})["grasp"] = pose7_from_pose_dict(g["pose_base"])
        if isinstance(p, dict) and isinstance(p.get("pose_base"), dict):
            out.setdefault("flange", {})["pregrasp"] = pose7_from_pose_dict(p["pose_base"])

    return out


def fmt_pose(p: Sequence[float]) -> str:
    return (
        f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, "
        f"{p[3]:.6f}, {p[4]:.6f}, {p[5]:.6f}, {p[6]:.6f}]"
    )


def normalize_quat(q: Sequence[float]) -> Tuple[float, float, float, float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n <= 1e-12:
        raise ValueError("zero quaternion")
    return tuple(float(v) / n for v in q)  # type: ignore[return-value]


def pose_error(visual: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    dx = actual[0] - visual[0]
    dy = actual[1] - visual[1]
    dz = actual[2] - visual[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    qv = normalize_quat(visual[3:7])
    qa = normalize_quat(actual[3:7])
    # q and -q represent the same rotation.
    dot = abs(sum(a * b for a, b in zip(qv, qa)))
    dot = max(-1.0, min(1.0, dot))
    angle_deg = math.degrees(2.0 * math.acos(dot))

    return {
        "dx_mm": dx,
        "dy_mm": dy,
        "dz_mm": dz,
        "position_error_mm": dist,
        "orientation_error_deg": angle_deg,
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
    return vals


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    fields = [
        "timestamp",
        "request_id",
        "selected_grasp_branch",
        "compare_frame",
        "visual_x_mm", "visual_y_mm", "visual_z_mm",
        "visual_qx", "visual_qy", "visual_qz", "visual_qw",
        "actual_x_mm", "actual_y_mm", "actual_z_mm",
        "actual_qx", "actual_qy", "actual_qz", "actual_qw",
        "dx_mm", "dy_mm", "dz_mm",
        "position_error_mm", "orientation_error_deg",
        "raw_json",
    ]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def save_raw(out_dir: Path, request_id: str, data: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in request_id)
    if not safe:
        safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"{safe}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_result_summary(data: Dict[str, Any], rt: Dict[str, Any], poses: Dict[str, Dict[str, List[float]]], compare_frame: str) -> None:
    request_id = str(data.get("request_id", ""))
    branch = data.get("selected_grasp_branch") or rt.get("grasp_branch")
    print("\n" + "=" * 88)
    print(f"request_id : {request_id}")
    print(f"status     : {data.get('status')}")
    print(f"target     : {data.get('target_found')}")
    print(f"arm        : {rt.get('arm')}")
    print(f"base frame : {rt.get('base_frame_id')}")
    scene_summary = data.get("scene_summary") if isinstance(data.get("scene_summary"), dict) else {}
    selected_clock = scene_summary.get("selected_clock_hour", data.get("selected_clock_hour"))
    selected_preferred = scene_summary.get("selected_clock_preferred", data.get("selected_clock_preferred"))
    preferred_hours = scene_summary.get("preferred_clock_hours", data.get("preferred_clock_hours"))
    print(f"branch     : {branch}")
    print(f"clock      : {selected_clock}  preferred={selected_preferred}  preferred_hours={preferred_hours}")
    print("=" * 88)

    if "hand_tcp" in poses:
        print("HAND_TCP grasp     [X_mm, Y_mm, Z_mm, QX, QY, QZ, QW]")
        print("  " + fmt_pose(poses["hand_tcp"]["grasp"]))
        if "pregrasp" in poses["hand_tcp"]:
            print("HAND_TCP pregrasp  [X_mm, Y_mm, Z_mm, QX, QY, QZ, QW]")
            print("  " + fmt_pose(poses["hand_tcp"]["pregrasp"]))
        print()

    if "flange" in poses:
        print("LEFT_LINK7 grasp    [X_mm, Y_mm, Z_mm, QX, QY, QZ, QW]")
        print("  " + fmt_pose(poses["flange"]["grasp"]))
        if "pregrasp" in poses["flange"]:
            print("LEFT_LINK7 pregrasp [X_mm, Y_mm, Z_mm, QX, QY, QZ, QW]")
            print("  " + fmt_pose(poses["flange"]["pregrasp"]))
        print()

    label = "left_link7 / flange" if compare_frame == "flange" else "hand_tcp_link"
    if compare_frame in poses and "grasp" in poses[compare_frame]:
        print(f">>> 本轮建议人工对照的 {label} 抓取目标：")
        print("    " + fmt_pose(poses[compare_frame]["grasp"]))
        print()


def run_one(args: argparse.Namespace, out_dir: Path, csv_path: Path) -> bool:
    payload = {
        "wait": True,
        "timeout_ms": int(args.infer_timeout_ms),
        "save_debug": bool(args.save_debug),
    }
    data = post_json(args.url, payload, timeout_s=args.http_timeout_s)

    request_id = str(data.get("request_id", ""))
    raw_path = save_raw(out_dir, request_id, data)

    if data.get("status") != "ok":
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"[ERROR] infer_once status != ok; raw response saved: {raw_path}")
        return False
    if data.get("target_found") is not True:
        print(f"[WARN] target_found != true; raw response saved: {raw_path}")
        return False
    if data.get("robot_pose_transform_ready") is not True:
        print(f"[ERROR] robot_pose_transform_ready != true; raw response saved: {raw_path}")
        return False

    rt = find_robot_transform(data)
    if rt.get("arm") != "left":
        raise RuntimeError(f"Refusing non-left result: arm={rt.get('arm')!r}")
    actual_base_frame = str(rt.get("base_frame_id") or "")
    if actual_base_frame != args.expected_base_frame:
        raise RuntimeError(
            f"Unexpected base frame: {actual_base_frame!r}; "
            f"expected {args.expected_base_frame!r}"
        )

    poses = extract_poses(rt)
    print_result_summary(data, rt, poses, args.compare_frame)
    print(f"raw JSON   : {raw_path}")
    print(f"sample CSV : {csv_path}")

    target = poses.get(args.compare_frame, {}).get("grasp")
    if target is None:
        print(f"[ERROR] requested compare frame {args.compare_frame!r} is unavailable")
        return False

    if args.no_input:
        return True

    print("\n机器人保持/移动到你认为正确的抓取位置后，可录入机器人实际位姿。")
    if args.compare_frame == "flange":
        print(
            "当前比较对象: visual left_link7/flange ↔ "
            "robot.motion.get_pose(ArmType.Left) 返回的末端法兰 pose"
        )
    else:
        print(
            "当前比较对象: visual hand_tcp_link ↔ 手工提供的 hand_tcp pose "
            "(仅在机器人端明确输出同一个 TCP 时使用)"
        )
    print(f"规划器 XYZ 输入单位当前设为: {args.planner_unit}")
    print("输入 7 个数: X Y Z QX QY QZ QW")
    print("直接回车 = 本轮暂不录入；q = 退出整个脚本")
    text = input("planner pose > ").strip()
    if not text:
        return True
    if text.lower() in {"q", "quit", "exit"}:
        raise KeyboardInterrupt

    try:
        actual = parse_actual_pose(text, args.planner_unit)
    except Exception as exc:
        print(f"[WARN] 输入无效，本轮只保存视觉结果: {exc}")
        return True

    err = pose_error(target, actual)
    print("\n--- 本轮对比 ---")
    print(f"visual : {fmt_pose(target)}")
    print(f"actual : {fmt_pose(actual)}")
    print(
        f"delta  : dX={err['dx_mm']:+.3f} mm, "
        f"dY={err['dy_mm']:+.3f} mm, dZ={err['dz_mm']:+.3f} mm"
    )
    print(f"position error    : {err['position_error_mm']:.3f} mm")
    print(f"orientation error : {err['orientation_error_deg']:.3f} deg")

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id,
        "selected_grasp_branch": data.get("selected_grasp_branch") or rt.get("grasp_branch"),
        "compare_frame": "left_link7" if args.compare_frame == "flange" else "hand_tcp_link",
        "raw_json": str(raw_path),
    }
    for key, value in zip(POSE_HEADERS, target):
        row[f"visual_{key}"] = value
    for key, value in zip(POSE_HEADERS, actual):
        row[f"actual_{key}"] = value
    row.update(err)
    append_csv(csv_path, row)
    print(f"[OK] paired sample appended to {csv_path}")
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trigger M39.2 foam-ring inference and print/save LEFT-arm 7D grasp poses."
    )
    p.add_argument("--url", default=DEFAULT_URL, help=f"infer_once URL (default: {DEFAULT_URL})")
    p.add_argument(
        "--expected-base-frame",
        default="robot_default_base",
        help=(
            "expected robot planning/reference base frame in infer output "
            "(default: robot_default_base)"
        ),
    )
    p.add_argument(
        "--compare-frame",
        choices=("flange", "hand_tcp"),
        default="flange",
        help=(
            "frame used for manual planner comparison; both are always printed. "
            "Default flange means visual left_link7 is compared with "
            "robot.motion.get_pose(ArmType.Left)."
        ),
    )
    p.add_argument(
        "--planner-unit",
        choices=("m", "mm"),
        default="m",
        help="unit of XYZ manually copied from planner (default: m; quaternion is unitless)",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--infer-timeout-ms", type=int, default=30000, help="service infer timeout_ms")
    p.add_argument("--http-timeout-s", type=float, default=40.0, help="HTTP socket timeout seconds")
    p.add_argument("--save-debug", action="store_true", default=True, help="ask service to save debug artifacts")
    p.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    p.add_argument("--once", action="store_true", help="run only one trigger, otherwise interactive repeated testing")
    p.add_argument("--no-input", action="store_true", help="only print/save visual pose; do not ask for planner pose")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = out_dir / "m39_2_pose_comparison.csv"

    print("M39.2 LEFT visual grasp pose manual verification")
    print(f"infer URL     : {args.url}")
    print(f"expected base : {args.expected_base_frame}")
    print(f"compare frame : {'left_link7 / flange' if args.compare_frame == 'flange' else 'hand_tcp_link'}")
    print(f"planner unit  : {args.planner_unit}")
    print(f"output dir    : {out_dir}")
    print("NOTE: visual XYZ printed by this script is always mm; quaternion order is QX QY QZ QW (xyzw).")

    try:
        while True:
            if not args.once:
                cmd = input("\nEnter = trigger one visual test; q = quit > ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    break
            try:
                run_one(args, out_dir, csv_path)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
            if args.once:
                break
    except KeyboardInterrupt:
        pass

    print("\nDone.")
    if csv_path.exists():
        print(f"comparison CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
