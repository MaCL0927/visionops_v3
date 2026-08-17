#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive 8-point 3-D box calibration for foam_ring_grasp.

Python 3.8 compatible.

Workflow
--------
1. Acquire one synchronized RGB-D frame from the Orbbec shared-memory bridge,
   or load an existing foam-ring debug capture directory.
2. Click 8 image points in this exact order:
     opening: TL, TR, BR, BL
     bottom : TL, TR, BR, BL
3. The four bottom clicks seed local depth patches used to robustly fit the box
   bottom plane. The opening edge is depth-discontinuous, so opening clicks are
   treated as camera rays and intersected with a plane parallel to the fitted
   bottom plane, offset by the known box depth (default: existing box_model).
4. Width/height/depth remain the measured physical inner dimensions from the
   existing production box model unless explicitly overridden.
5. A production-compatible box_model JSON and diagnostic overlay are written.
   Existing config/box_model.json is NOT overwritten unless --install is used.

Run from /opt/visionops_v3, for example:

  python3 production/foam_ring_grasp/scripts/fit_box_model_8point.py

or with a previously saved exact RGB-D debug capture:

  python3 production/foam_ring_grasp/scripts/fit_box_model_8point.py \
      --capture-dir data/foam_ring_online_geometry/<timestamp>
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


POINT_LABELS = [
    "OPEN_TL", "OPEN_TR", "OPEN_BR", "OPEN_BL",
    "BOTTOM_TL", "BOTTOM_TR", "BOTTOM_BR", "BOTTOM_BL",
]
OPEN_COLORS = [(0, 255, 255)] * 4
BOTTOM_COLORS = [(255, 180, 0)] * 4


def _project_root() -> Path:
    here = Path(__file__).resolve()
    # Installed location: <root>/production/foam_ring_grasp/scripts/<this.py>
    if len(here.parents) >= 4 and here.parent.name == "scripts":
        return here.parents[3]
    return Path.cwd().resolve()


def _default_config_dir() -> Path:
    root = _project_root()
    p = root / "production" / "foam_ring_grasp" / "config"
    if p.exists():
        return p
    # Standalone development copy fallback.
    return Path.cwd().resolve()


def _unit(v: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 1e-9:
        raise RuntimeError("%s is zero" % name)
    return a / n


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("JSON root must be object: %s" % path)
    return data


def _load_reference_box(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError("reference box model not found: %s" % path)
    d = _load_json(path)
    if str(d.get("model_type") or "") != "calibrated_3d_cuboid":
        raise RuntimeError("reference box model is not calibrated_3d_cuboid")
    return d


def _intrinsics_from_box(d: Mapping[str, Any]) -> Dict[str, float]:
    intr = d.get("intrinsics") or {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intr:
            raise RuntimeError("reference box model intrinsics missing %s" % key)
    return {key: float(intr[key]) for key in ("fx", "fy", "cx", "cy")}


def _resolution_from_box(d: Mapping[str, Any]) -> Tuple[int, int]:
    r = d.get("camera_resolution") or {}
    w, h = int(r.get("width", 0)), int(r.get("height", 0))
    if w <= 0 or h <= 0:
        raise RuntimeError("reference box model camera_resolution invalid")
    return w, h


def _size_from_box(d: Mapping[str, Any]) -> Tuple[float, float, float]:
    s = d.get("inner_size_mm") or {}
    vals = (float(s.get("width", 0)), float(s.get("height", 0)), float(s.get("depth", 0)))
    if min(vals) <= 0:
        raise RuntimeError("reference box model inner_size_mm invalid")
    return vals


def _margins_from_box(d: Mapping[str, Any]) -> Dict[str, float]:
    src = d.get("safety_margin_mm") or {}
    return {
        "left": float(src.get("left", 8.0)),
        "right": float(src.get("right", 8.0)),
        "top": float(src.get("top", 8.0)),
        "bottom": float(src.get("bottom", 10.0)),
        "back": float(src.get("back", 8.0)),
    }


def _find_latest_capture(root: Path) -> Path:
    candidates = []
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and (p / "exact_rgb.png").exists() and (p / "exact_depth.png").exists():
                candidates.append(p)
    if not candidates:
        raise RuntimeError("no debug capture with exact_rgb.png/exact_depth.png under %s" % root)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_capture_dir(path: Path, intrinsics: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    rgb_path = path / "exact_rgb.png"
    depth_path = path / "exact_depth.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise RuntimeError("cannot read %s" % rgb_path)
    if depth is None or depth.dtype != np.uint16:
        raise RuntimeError("cannot read uint16 depth %s" % depth_path)
    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError("RGB/depth shape mismatch: %r vs %r" % (rgb.shape, depth.shape))
    effective = dict(intrinsics)
    result_path = path / "online_geometry_result.json"
    if result_path.exists():
        try:
            d = _load_json(result_path)
            cand = d.get("intrinsics")
            if isinstance(cand, dict) and all(k in cand for k in ("fx", "fy", "cx", "cy")):
                effective = {k: float(cand[k]) for k in ("fx", "fy", "cx", "cy")}
        except Exception:
            pass
    meta = {"mode": "capture_dir", "capture_dir": str(path), "rgb": str(rgb_path), "depth": str(depth_path)}
    return rgb, depth, effective, meta


def _load_live_frame(line_yaml: Path, reference_box: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    # Import only for live acquisition. This keeps offline capture mode simple.
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.rgbd_cache import (  # type: ignore
            RgbdFrameCache,
            load_rgbd_cache_settings,
        )
    except Exception as error:
        raise RuntimeError("cannot import production RGB-D cache: %s" % error)

    settings = load_rgbd_cache_settings(line_yaml)
    cache = RgbdFrameCache(
        rgb_name=settings.rgb_name,
        depth_name=settings.depth_name,
        max_frames=settings.cache_frames,
        max_age_ms=settings.max_age_ms,
        poll_interval_ms=settings.poll_interval_ms,
        cache_rgb=True,
    )
    cache.start()
    try:
        if not cache.wait_until_ready(timeout=3.0):
            raise RuntimeError("RGB-D shared-memory cache is not ready: %s" % cache.status())
        time.sleep(0.08)
        frame = cache.latest()
        if frame is None or frame.rgb is None:
            raise RuntimeError("latest RGB-D frame unavailable")
        rgb_full = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        depth_full = np.asarray(frame.depth_mm).copy()
        ref_intr = _intrinsics_from_box(reference_box)
        out_w, out_h = _resolution_from_box(reference_box)
        roi_x1 = int(round(float(frame.cx) - float(ref_intr["cx"])))
        roi_y1 = int(round(float(frame.cy) - float(ref_intr["cy"])))
        roi_x2, roi_y2 = roi_x1 + out_w, roi_y1 + out_h
        if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.width or roi_y2 > frame.height:
            raise RuntimeError(
                "derived production ROI is outside live frame: roi=%r frame=%dx%d" %
                ((roi_x1, roi_y1, roi_x2, roi_y2), frame.width, frame.height)
            )
        rgb = rgb_full[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        depth = depth_full[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        intr = {
            "fx": float(frame.fx), "fy": float(frame.fy),
            "cx": float(frame.cx) - roi_x1, "cy": float(frame.cy) - roi_y1,
        }
        meta = {
            "mode": "live_shared_memory",
            "timestamp_epoch_ms": int(frame.timestamp_epoch_ms),
            "full_resolution": [int(frame.width), int(frame.height)],
            "geometry_roi_xyxy": [roi_x1, roi_y1, roi_x2, roi_y2],
            "line_yaml": str(line_yaml),
        }
        return rgb, depth, intr, meta
    finally:
        cache.stop()


def _draw_clicks(image: np.ndarray, points: Sequence[Tuple[int, int]]) -> np.ndarray:
    vis = image.copy()
    for i, (u, v) in enumerate(points):
        color = OPEN_COLORS[i] if i < 4 else BOTTOM_COLORS[i - 4]
        cv2.circle(vis, (int(u), int(v)), 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(u), int(v)), 10, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, POINT_LABELS[i], (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if len(points) < 8:
        msg = "CLICK %d/8: %s" % (len(points) + 1, POINT_LABELS[len(points)])
    else:
        msg = "8/8 READY - ENTER fit | U undo | R reset | Q quit"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, msg, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def collect_points(image: np.ndarray) -> List[Tuple[int, int]]:
    window = "M39 box 8-point calibration"
    points: List[Tuple[int, int]] = []

    def callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
            points.append((int(x), int(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1100, image.shape[1]), min(850, image.shape[0]))
    cv2.setMouseCallback(window, callback)
    try:
        while True:
            cv2.imshow(window, _draw_clicks(image, points))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("operator cancelled")
            if key in (ord("u"), 8, 127) and points:
                points.pop()
            elif key == ord("r"):
                points[:] = []
            elif key in (10, 13) and len(points) == 8:
                return list(points)
    finally:
        cv2.destroyWindow(window)


def _pixel_to_ray(u: float, v: float, intr: Mapping[str, float]) -> np.ndarray:
    ray = np.asarray([
        (float(u) - float(intr["cx"])) / float(intr["fx"]),
        (float(v) - float(intr["cy"])) / float(intr["fy"]),
        1.0,
    ], dtype=np.float64)
    return ray


def _backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, intr: Mapping[str, float]) -> np.ndarray:
    x = (u.astype(np.float64) - float(intr["cx"])) * z / float(intr["fx"])
    y = (v.astype(np.float64) - float(intr["cy"])) * z / float(intr["fy"])
    return np.column_stack((x, y, z))


def _patch_points(depth: np.ndarray, uv: Tuple[int, int], intr: Mapping[str, float], radius: int, depth_window_mm: float) -> np.ndarray:
    u0, v0 = uv
    h, w = depth.shape[:2]
    x1, x2 = max(0, u0 - radius), min(w, u0 + radius + 1)
    y1, y2 = max(0, v0 - radius), min(h, v0 + radius + 1)
    patch = depth[y1:y2, x1:x2].astype(np.float64)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    valid = (patch > 100.0) & (patch < 4000.0)
    if np.count_nonzero(valid) < 6:
        raise RuntimeError("too few valid depth pixels near bottom click %r" % (uv,))
    med = float(np.median(patch[valid]))
    valid &= np.abs(patch - med) <= float(depth_window_mm)
    if np.count_nonzero(valid) < 6:
        raise RuntimeError("too few depth inliers near bottom click %r" % (uv,))
    return _backproject(xx[valid], yy[valid], patch[valid], intr)


def _fit_plane(points: np.ndarray, iterations: int = 4) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(max(1, iterations)):
        cur = pts[keep]
        if len(cur) < 12:
            raise RuntimeError("too few points for floor plane")
        center = np.mean(cur, axis=0)
        _, _, vh = np.linalg.svd(cur - center, full_matrices=False)
        n = _unit(vh[-1], "floor normal")
        c = float(np.dot(n, center))
        residual = np.abs(pts @ n - c)
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        sigma = max(0.5, 1.4826 * mad)
        threshold = min(8.0, max(2.0, med + 3.0 * sigma))
        new_keep = residual <= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    cur = pts[keep]
    center = np.mean(cur, axis=0)
    _, _, vh = np.linalg.svd(cur - center, full_matrices=False)
    n = _unit(vh[-1], "floor normal")
    if float(np.dot(n, center)) > 0.0:
        n = -n  # point from floor toward camera origin
    c = float(np.dot(n, center))
    residual = np.abs(cur @ n - c)
    diag = {
        "point_count": int(len(pts)),
        "inlier_count": int(len(cur)),
        "inlier_ratio": float(len(cur)) / float(len(pts)),
        "residual_median_mm": float(np.median(residual)),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
    }
    return n, c, keep, diag


def _intersect_ray_plane(uv: Tuple[int, int], intr: Mapping[str, float], n: np.ndarray, c: float) -> np.ndarray:
    ray = _pixel_to_ray(float(uv[0]), float(uv[1]), intr)
    denom = float(np.dot(n, ray))
    if abs(denom) < 1e-8:
        raise RuntimeError("opening click ray nearly parallel to opening plane: %r" % (uv,))
    t = float(c) / denom
    if t <= 0.0:
        raise RuntimeError("opening click intersects plane behind camera: %r" % (uv,))
    return ray * t


def _median_patch_point(depth: np.ndarray, uv: Tuple[int, int], intr: Mapping[str, float], radius: int, depth_window_mm: float) -> np.ndarray:
    pts = _patch_points(depth, uv, intr, radius, depth_window_mm)
    return np.median(pts, axis=0)


def _project(points: np.ndarray, intr: Mapping[str, float]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uv = np.empty((len(p), 2), dtype=np.float64)
    uv[:, 0] = float(intr["fx"]) * p[:, 0] / p[:, 2] + float(intr["cx"])
    uv[:, 1] = float(intr["fy"]) * p[:, 1] / p[:, 2] + float(intr["cy"])
    return uv


def fit_box(
    depth: np.ndarray,
    points_uv: Sequence[Tuple[int, int]],
    intr: Mapping[str, float],
    width_mm: float,
    height_mm: float,
    depth_mm: float,
    patch_radius: int,
    patch_depth_window_mm: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if len(points_uv) != 8:
        raise RuntimeError("exactly 8 points required")
    opening_uv = list(points_uv[:4])
    bottom_uv = list(points_uv[4:])

    patch_clouds = [
        _patch_points(depth, uv, intr, patch_radius, patch_depth_window_mm)
        for uv in bottom_uv
    ]
    floor_points = np.concatenate(patch_clouds, axis=0)
    n_to_camera, floor_c, _keep, plane_diag = _fit_plane(floor_points)
    z_inside = -n_to_camera
    opening_c = float(floor_c + depth_mm)  # offset toward camera by known depth
    opening_raw = np.asarray([
        _intersect_ray_plane(uv, intr, n_to_camera, opening_c) for uv in opening_uv
    ], dtype=np.float64)

    x_seed = (opening_raw[1] - opening_raw[0]) + (opening_raw[2] - opening_raw[3])
    y_seed = (opening_raw[3] - opening_raw[0]) + (opening_raw[2] - opening_raw[1])
    x_proj = x_seed - z_inside * float(np.dot(x_seed, z_inside))
    x_axis = _unit(x_proj, "x_right")
    y_axis = _unit(np.cross(z_inside, x_axis), "y_down")
    if float(np.dot(y_axis, y_seed)) < 0.0:
        raise RuntimeError(
            "clicked corner order is inconsistent with TL,TR,BR,BL; y_down points opposite clicked top->bottom"
        )

    ideal_xy = np.asarray([
        [0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]
    ], dtype=np.float64)
    origin_candidates = []
    for q, xy in zip(opening_raw, ideal_xy):
        origin_candidates.append(q - x_axis * xy[0] - y_axis * xy[1])
    origin = np.mean(np.asarray(origin_candidates), axis=0)
    # Reproject origin onto the opening plane exactly.
    origin = origin + n_to_camera * (opening_c - float(np.dot(n_to_camera, origin)))

    rotation = np.column_stack((x_axis, y_axis, z_inside))
    # Re-orthonormalize without changing handedness.
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if float(np.linalg.det(rotation)) < 0.0:
        rotation[:, 1] *= -1.0
    x_axis, y_axis, z_inside = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    if float(np.dot(z_inside, -n_to_camera)) < 0.0:
        raise RuntimeError("internal orientation error: z_inside does not point from opening toward box bottom")

    corners_box = np.asarray([
        [0, 0, 0], [width_mm, 0, 0], [width_mm, height_mm, 0], [0, height_mm, 0],
        [0, 0, depth_mm], [width_mm, 0, depth_mm], [width_mm, height_mm, depth_mm], [0, height_mm, depth_mm],
    ], dtype=np.float64)
    corners_camera = corners_box @ rotation.T + origin
    opening_ideal = corners_camera[:4]
    bottom_ideal = corners_camera[4:]

    opening_residual = np.linalg.norm(opening_raw - opening_ideal, axis=1)
    bottom_measured = np.asarray([
        _median_patch_point(depth, uv, intr, patch_radius, patch_depth_window_mm)
        for uv in bottom_uv
    ], dtype=np.float64)
    bottom_residual = np.linalg.norm(bottom_measured - bottom_ideal, axis=1)

    measured_top = 0.5 * (np.linalg.norm(opening_raw[1] - opening_raw[0]) + np.linalg.norm(opening_raw[2] - opening_raw[3]))
    measured_left = 0.5 * (np.linalg.norm(opening_raw[3] - opening_raw[0]) + np.linalg.norm(opening_raw[2] - opening_raw[1]))

    opening_center = np.mean(opening_ideal, axis=0)
    bottom_center = np.mean(bottom_ideal, axis=0)
    camera_z = np.asarray([0.0, 0.0, 1.0])
    z_angle = math.degrees(math.acos(float(np.clip(np.dot(z_inside, camera_z), -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0])))

    diag = {
        "floor_plane": plane_diag,
        "opening_click_ray_intersections_camera_mm": opening_raw.tolist(),
        "bottom_click_depth_points_camera_mm": bottom_measured.tolist(),
        "opening_corner_fit_error_mm": opening_residual.tolist(),
        "opening_corner_fit_error_median_mm": float(np.median(opening_residual)),
        "opening_corner_fit_error_max_mm": float(np.max(opening_residual)),
        "bottom_corner_model_error_mm": bottom_residual.tolist(),
        "bottom_corner_model_error_median_mm": float(np.median(bottom_residual)),
        "bottom_corner_model_error_max_mm": float(np.max(bottom_residual)),
        "manual_top_edge_length_projected_mm": float(measured_top),
        "manual_left_edge_length_projected_mm": float(measured_left),
        "opening_center_camera_mm": opening_center.tolist(),
        "bottom_center_camera_mm": bottom_center.tolist(),
        "box_z_vs_camera_z_deg": float(z_angle),
        "box_x_yaw_in_image_deg": float(yaw),
        "corners_camera_mm": corners_camera.tolist(),
    }
    model_core = {
        "origin_camera_mm": origin.tolist(),
        "rotation_camera_from_box_rows": rotation.tolist(),
        "axes_camera": {
            "x_right": x_axis.tolist(), "y_down": y_axis.tolist(), "z_inside": z_inside.tolist()
        },
        "corners_camera_mm": corners_camera.tolist(),
    }
    return model_core, diag


def _overlay_fit(rgb: np.ndarray, clicks: Sequence[Tuple[int, int]], corners_camera: np.ndarray, intr: Mapping[str, float], diag: Mapping[str, Any]) -> np.ndarray:
    vis = _draw_clicks(rgb, clicks)
    uv = _project(corners_camera, intr)
    front = [0, 1, 2, 3]
    rear = [4, 5, 6, 7]
    for inds, color in ((front, (0, 255, 0)), (rear, (255, 0, 255))):
        pts = np.round(uv[inds]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
    for i in range(4):
        a = tuple(np.round(uv[i]).astype(int))
        b = tuple(np.round(uv[i + 4]).astype(int))
        cv2.line(vis, a, b, (0, 180, 255), 2, cv2.LINE_AA)
    lines = [
        "FIT: green=opening magenta=bottom orange=depth edges",
        "floor residual med/p95 %.2f/%.2f mm" % (
            float((diag.get("floor_plane") or {}).get("residual_median_mm", 0.0)),
            float((diag.get("floor_plane") or {}).get("residual_p95_mm", 0.0)),
        ),
        "opening corner err med/max %.2f/%.2f mm" % (
            float(diag.get("opening_corner_fit_error_median_mm", 0.0)),
            float(diag.get("opening_corner_fit_error_max_mm", 0.0)),
        ),
        "bottom corner err med/max %.2f/%.2f mm" % (
            float(diag.get("bottom_corner_model_error_median_mm", 0.0)),
            float(diag.get("bottom_corner_model_error_max_mm", 0.0)),
        ),
    ]
    y = 48
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return vis


def main() -> int:
    config_dir = _default_config_dir()
    parser = argparse.ArgumentParser(description="8-click 3-D box calibration for foam_ring_grasp")
    parser.add_argument("--reference-box", default=str(config_dir / "box_model.json"), help="existing box model used for intrinsics, size, margins")
    parser.add_argument("--line-yaml", default=str(config_dir / "line.yaml"), help="production line.yaml for live shared-memory names")
    parser.add_argument("--capture-dir", default="", help="offline capture dir containing exact_rgb.png and exact_depth.png; if omitted, capture latest live RGB-D")
    parser.add_argument("--latest-debug", action="store_true", help="use latest data/foam_ring_online_geometry debug capture instead of live shared memory")
    parser.add_argument("--output", default=str(config_dir / "box_model_8point_candidate.json"))
    parser.add_argument("--width-mm", type=float, default=None)
    parser.add_argument("--height-mm", type=float, default=None)
    parser.add_argument("--depth-mm", type=float, default=None)
    parser.add_argument("--patch-radius-px", type=int, default=7)
    parser.add_argument("--patch-depth-window-mm", type=float, default=20.0)
    parser.add_argument("--install", action="store_true", help="after fitting, back up and replace production config/box_model.json")
    parser.add_argument("--points-json", default="", help="non-interactive JSON list of 8 [u,v] points for testing/replay")
    args = parser.parse_args()

    ref_path = Path(args.reference_box).expanduser().resolve()
    ref = _load_reference_box(ref_path)
    intr_ref = _intrinsics_from_box(ref)
    ref_w, ref_h = _resolution_from_box(ref)
    width0, height0, depth0 = _size_from_box(ref)
    width_mm = float(args.width_mm if args.width_mm is not None else width0)
    height_mm = float(args.height_mm if args.height_mm is not None else height0)
    depth_mm = float(args.depth_mm if args.depth_mm is not None else depth0)

    if args.capture_dir:
        rgb, depth, intr, source_meta = _load_capture_dir(Path(args.capture_dir).expanduser().resolve(), intr_ref)
    elif args.latest_debug:
        root = _project_root() / "data" / "foam_ring_online_geometry"
        capture = _find_latest_capture(root)
        print("Using latest debug capture: %s" % capture)
        rgb, depth, intr, source_meta = _load_capture_dir(capture, intr_ref)
    else:
        rgb, depth, intr, source_meta = _load_live_frame(Path(args.line_yaml).expanduser().resolve(), ref)

    if rgb.shape[1] != ref_w or rgb.shape[0] != ref_h:
        raise RuntimeError(
            "calibration image resolution %dx%d does not match production box model %dx%d" %
            (rgb.shape[1], rgb.shape[0], ref_w, ref_h)
        )
    if depth.shape[:2] != rgb.shape[:2]:
        raise RuntimeError("depth resolution mismatch")

    if args.points_json:
        raw_points = json.loads(Path(args.points_json).read_text(encoding="utf-8"))
        if isinstance(raw_points, dict):
            raw_points = raw_points.get("points_uv")
        if not isinstance(raw_points, list) or len(raw_points) != 8:
            raise RuntimeError("--points-json must contain exactly 8 [u,v] points or an object with points_uv")
        points = [(int(round(p[0])), int(round(p[1]))) for p in raw_points]
    else:
        print("\nClick exactly in this order:")
        print("  1-4 opening: TL -> TR -> BR -> BL")
        print("  5-8 bottom : TL -> TR -> BR -> BL")
        print("Keys: U=undo, R=reset, ENTER=fit after 8 points, Q=cancel\n")
        points = collect_points(rgb)

    core, diag = fit_box(
        depth, points, intr, width_mm, height_mm, depth_mm,
        max(2, int(args.patch_radius_px)), float(args.patch_depth_window_mm),
    )
    rotation = np.asarray(core["rotation_camera_from_box_rows"], dtype=np.float64)
    corners_camera = np.asarray(core["corners_camera_mm"], dtype=np.float64)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path = output_path.with_name(output_path.stem + "_overlay.jpg")
    points_path = output_path.with_name(output_path.stem + "_points.json")

    model = {
        "schema_version": "1.0",
        "message_type": "visionops_box_model_3d",
        "model_type": "calibrated_3d_cuboid",
        "coordinate_frame": "camera_color_optical_frame",
        "origin_definition": "front_left_top_inner_corner",
        "origin_camera_mm": [float(v) for v in core["origin_camera_mm"]],
        "axes_camera": core["axes_camera"],
        "rotation_camera_from_box_rows": rotation.astype(float).tolist(),
        "inner_size_mm": {"width": width_mm, "height": height_mm, "depth": depth_mm},
        "safety_margin_mm": _margins_from_box(ref),
        "camera_resolution": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
        "intrinsics": {key: float(intr[key]) for key in ("fx", "fy", "cx", "cy")},
        "calibration": {
            "status": "calibrated_8point_manual_opening_bottom",
            "method": "bottom_depth_plane_plus_known_depth_opening_ray_intersections_8point",
            "created_at_local": now,
            "source": source_meta,
            "nominal_inner_size_mm": {
                "width": width_mm, "height": height_mm, "depth": depth_mm,
                "source": "production_reference_or_cli_override",
            },
            "input": {
                "manual_opening_corners_uv": dict(zip(("TL", "TR", "BR", "BL"), [list(p) for p in points[:4]])),
                "manual_bottom_corners_uv": dict(zip(("TL", "TR", "BR", "BL"), [list(p) for p in points[4:]])),
                "bottom_depth_patch_radius_px": int(args.patch_radius_px),
                "bottom_depth_keep_window_mm": float(args.patch_depth_window_mm),
                "opening_depth_policy": "ray_intersection_with_plane_parallel_to_bottom_offset_by_nominal_depth",
            },
            "stability": {
                "plane_residual_median_mm": float(diag["floor_plane"]["residual_median_mm"]),
                "plane_residual_p95_mm": float(diag["floor_plane"]["residual_p95_mm"]),
                "fitted_point_count": int(diag["floor_plane"]["inlier_count"]),
                "fitted_point_ratio": float(diag["floor_plane"]["inlier_ratio"]),
            },
            "derived": {
                "opening_center_camera_mm": diag["opening_center_camera_mm"],
                "bottom_center_camera_mm": diag["bottom_center_camera_mm"],
                "box_z_vs_camera_z_deg": float(diag["box_z_vs_camera_z_deg"]),
                "box_x_yaw_in_image_deg": float(diag["box_x_yaw_in_image_deg"]),
                "manual_top_edge_length_projected_mm": float(diag["manual_top_edge_length_projected_mm"]),
                "manual_left_edge_length_projected_mm": float(diag["manual_left_edge_length_projected_mm"]),
                "opening_corner_fit_error_median_mm": float(diag["opening_corner_fit_error_median_mm"]),
                "opening_corner_fit_error_max_mm": float(diag["opening_corner_fit_error_max_mm"]),
                "bottom_corner_model_error_median_mm": float(diag["bottom_corner_model_error_median_mm"]),
                "bottom_corner_model_error_max_mm": float(diag["bottom_corner_model_error_max_mm"]),
            },
        },
    }

    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    points_path.write_text(json.dumps({"labels": POINT_LABELS, "points_uv": [list(p) for p in points], "source": source_meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay = _overlay_fit(rgb, points, corners_camera, intr, diag)
    if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError("cannot save overlay %s" % overlay_path)

    print("\n" + "=" * 78)
    print("M39 8-POINT BOX CALIBRATION RESULT")
    print("=" * 78)
    print("source                  : %s" % source_meta.get("mode"))
    print("resolution              : %dx%d" % (rgb.shape[1], rgb.shape[0]))
    print("intrinsics              : fx=%.3f fy=%.3f cx=%.3f cy=%.3f" % (intr["fx"], intr["fy"], intr["cx"], intr["cy"]))
    print("inner size mm           : %.1f x %.1f x %.1f" % (width_mm, height_mm, depth_mm))
    print("floor plane residual    : median=%.3f mm p95=%.3f mm" % (diag["floor_plane"]["residual_median_mm"], diag["floor_plane"]["residual_p95_mm"]))
    print("opening corner fit err  : median=%.3f mm max=%.3f mm" % (diag["opening_corner_fit_error_median_mm"], diag["opening_corner_fit_error_max_mm"]))
    print("bottom corner model err : median=%.3f mm max=%.3f mm" % (diag["bottom_corner_model_error_median_mm"], diag["bottom_corner_model_error_max_mm"]))
    print("box Z vs camera Z       : %.3f deg" % diag["box_z_vs_camera_z_deg"])
    print("box X image yaw         : %.3f deg" % diag["box_x_yaw_in_image_deg"])
    print("candidate JSON          : %s" % output_path)
    print("overlay                 : %s" % overlay_path)
    print("click record            : %s" % points_path)

    production_box = ref_path
    if args.install:
        backup = production_box.with_name(production_box.name + ".bak_" + now)
        shutil.copy2(str(production_box), str(backup))
        shutil.copy2(str(output_path), str(production_box))
        print("BACKUP old box model    : %s" % backup)
        print("INSTALLED               : %s" % production_box)
    else:
        print("NOT installed. Inspect overlay, then rerun with --install or copy candidate manually.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print("\nCancelled: %s" % error)
        raise SystemExit(130)
