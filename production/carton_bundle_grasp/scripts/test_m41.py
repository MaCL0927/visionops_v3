#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified M41.x test entry point.

Subcommands:
  performance   watch live online timing
  status        print App/shared-depth status
  offline       validate RGB-D dataset geometry without RKNN
  selftest      exercise the M41.2 ROI-snapshot shared-depth reader locally
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_bundle_grasp.config import load_config
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.algorithm import CartonBundleGraspAlgorithm
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.local_ipc import (
    SHARED_DEPTH_HEADER,
    SHARED_DEPTH_HEADER_SIZE,
    SHARED_DEPTH_MAGIC,
    SHARED_DEPTH_PIXEL_UINT16_MM,
    SHARED_DEPTH_STATE_RUNNING,
    SHARED_DEPTH_VERSION,
    SharedDepthReader,
)


def _fetch(url: str, timeout: float = 2.0) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _value(document: Dict[str, Any], key: str) -> float:
    raw = document.get(key)
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def command_performance(args: argparse.Namespace) -> int:
    header = (
        " frame/result              rt  cls | depth snap vec dep plane ray rect geom | "
        "post age pts try transport fps"
    )
    print(header)
    print("-" * len(header))
    last_result = None
    while True:
        try:
            decision = _fetch(args.url)
            result_id = str(decision.get("result_id") or "-")
            if result_id != last_result or args.once:
                timing = decision.get("app_timing") if isinstance(decision.get("app_timing"), dict) else {}
                producer = decision.get("producer") if isinstance(decision.get("producer"), dict) else {}
                fps = producer.get("detection_fps") or producer.get("actual_fps") or 0.0
                print(
                    " {:22s} {:4.0f} {:4.1f} | {:5.1f} {:4.1f} {:4.1f} {:3.1f} {:5.1f} {:3.1f} {:4.1f} {:5.1f} | "
                    "{:4.1f} {:4.0f} {:3d} {:3d} {:18s} {:.2f}".format(
                        result_id[-22:],
                        _value(timing, "runtime_http_ms"),
                        _value(timing, "classify_ms"),
                        _value(timing, "depth_sample_deproject_ms"),
                        _value(timing, "depth_snapshot_copy_ms"),
                        _value(timing, "depth_vectorized_sample_ms"),
                        _value(timing, "depth_vectorized_deproject_ms"),
                        _value(timing, "plane_fit_ms"),
                        _value(timing, "corner_ray_build_ms") + _value(timing, "ray_plane_intersection_ms"),
                        _value(timing, "rectangle_reconstruct_ms"),
                        _value(timing, "geometry_3d_ms"),
                        _value(timing, "postprocess_stage_ms"),
                        _value(timing, "pipeline_age_ms"),
                        int(timing.get("depth_point_count") or 0),
                        int(timing.get("depth_snapshot_attempts") or 0),
                        str(timing.get("depth_transport") or "-"),
                        float(fps),
                    )
                )
                last_result = result_id
        except Exception as error:
            print("ERROR:", error)
        if args.once:
            return 0
        time.sleep(max(0.1, float(args.interval)))


def command_status(args: argparse.Namespace) -> int:
    document = _fetch(args.url)
    transport = (((document.get("ipc") or {}).get("camera_bridge") or {}).get("shared_depth"))
    output = {
        "version": document.get("version"),
        "continuous_enabled": document.get("continuous_enabled"),
        "producer": document.get("producer"),
        "shared_depth": transport,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _load_dataset_item(root: Path, label_path: Path, algorithm: CartonBundleGraspAlgorithm) -> Dict[str, Any]:
    stem = label_path.stem
    image = cv2.imread(str(root / "images" / (stem + ".jpg")))
    depth = cv2.imread(str(root / "depth" / (stem + ".png")), cv2.IMREAD_UNCHANGED)
    meta = json.loads((root / "meta" / (stem + ".json")).read_text(encoding="utf-8"))
    if image is None or depth is None:
        raise RuntimeError("missing RGB/depth for {}".format(stem))
    h, w = image.shape[:2]
    values = [float(value) for value in label_path.read_text(encoding="utf-8").split()]
    if len(values) < 9:
        raise RuntimeError("invalid segmentation label for {}".format(stem))
    polygon = [[values[i] * w, values[i + 1] * h] for i in range(1, len(values), 2)]
    runtime = {
        "image": {"width": w, "height": h},
        "detections": [{
            "id": stem,
            "class_id": int(values[0]),
            "class_name": "surface",
            "score": 1.0,
            "mask": {"source": "proto", "polygon": [polygon]},
        }],
    }
    classified = algorithm.classify(runtime)
    if not classified.items:
        return {"capture": stem, "status": "mask_rejected"}
    item = classified.items[0]

    points = np.asarray([[u, v, u, v] for u, v in item["plane_sample_points"]], dtype=np.float64)
    depth_h, depth_w = depth.shape[:2]
    sample_x = SharedDepthReader._map_pixel(points[:, 0], w, depth_w)
    sample_y = SharedDepthReader._map_pixel(points[:, 1], h, depth_h)
    radius_x = max(0, int(round(algorithm.depth_radius_px * float(max(1, depth_w - 1)) / float(max(1, w - 1)))))
    radius_y = max(0, int(round(algorithm.depth_radius_px * float(max(1, depth_h - 1)) / float(max(1, h - 1)))))
    z, valid_counts = SharedDepthReader._vectorized_depths(
        depth,
        sample_x,
        sample_y,
        0,
        0,
        radius_x,
        radius_y,
        algorithm.depth_percentile,
        algorithm.depth_min_valid_pixels,
        algorithm.min_depth_mm,
        algorithm.max_depth_mm,
        depth_w,
        depth_h,
    )
    intr = meta["depth"]["intrinsics_saved"]
    fx, fy, cx, cy = [float(intr[key]) for key in ("fx", "fy", "cx", "cy")]
    project_x = SharedDepthReader._map_coordinate(points[:, 2], w, depth_w)
    project_y = SharedDepthReader._map_coordinate(points[:, 3], h, depth_h)
    positions = np.stack(((project_x - cx) * z / fx, (project_y - cy) * z / fy, z), axis=1)
    positions[valid_counts < algorithm.depth_min_valid_pixels, :] = 0.0

    plane = algorithm.fit_plane(positions.tolist())
    rays = algorithm.corner_rays_from_intrinsics(item["quad"], intr, w, h, depth_w, depth_h)
    corners = algorithm.intersect_corner_rays(rays, plane)
    rectangle = algorithm.reconstruct_rectangle(item["quad"], corners, plane)
    return {
        "capture": stem,
        "status": "ok",
        "plane_samples": len(item["plane_sample_points"]),
        "observed_length_mm": round(float(rectangle["observed_length_mm"]), 3),
        "observed_width_mm": round(float(rectangle["observed_width_mm"]), 3),
        "plane_rms_mm": round(float(plane["rms_mm"]), 3),
        "plane_inlier_ratio": round(float(plane["inlier_ratio"]), 6),
        "z_ref_mm": round(float(plane["z_ref_mm"]), 3),
    }


def command_offline(args: argparse.Namespace) -> int:
    root = Path(args.dataset)
    algorithm = CartonBundleGraspAlgorithm(load_config(args.config)["carton_bundle_grasp"]["algorithm"])
    rows: List[Dict[str, Any]] = []
    for label_path in sorted((root / "labels").glob("*.txt")):
        try:
            rows.append(_load_dataset_item(root, label_path, algorithm))
        except Exception as error:
            rows.append({"capture": label_path.stem, "status": "error", "message": str(error)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0 if rows and all(row.get("status") == "ok" for row in rows) else 1


def command_selftest(_args: argparse.Namespace) -> int:
    name = "/visionops_m41_2_selftest_{}".format(os.getpid())
    path = "/dev/shm/" + name.lstrip("/")
    width, height = 64, 48
    stride = width * 2
    capacity = stride * height
    total = SHARED_DEPTH_HEADER_SIZE + 2 * capacity
    values: List[Any] = [
        SHARED_DEPTH_MAGIC, SHARED_DEPTH_VERSION, SHARED_DEPTH_HEADER_SIZE,
        total, capacity, capacity,
        width, height, stride, SHARED_DEPTH_PIXEL_UINT16_MM, 2,
        SHARED_DEPTH_STATE_RUNNING, 0, 1, 1, 0, 0, 0,
        123, int(time.time() * 1000), os.getpid(), 456, 0,
        100.0, 100.0, 32.0, 24.0,
    ] + [0] * 12
    header = SHARED_DEPTH_HEADER.pack(*values)
    frame = np.full((height, width), 1000, dtype="<u2")
    with open(path, "wb") as stream:
        stream.truncate(total)
        stream.seek(0)
        stream.write(header)
        stream.seek(SHARED_DEPTH_HEADER_SIZE)
        stream.write(frame.tobytes())
        stream.write(frame.tobytes())
    try:
        reader = SharedDepthReader(name, 500)
        points = [[10 + index % 10, 10 + index // 10, 10 + index % 10, 10 + index // 10] for index in range(96)]
        samples, response = reader.sample_deproject(points, width, height, 2, 50.0, 1, 100, 5000)
        assert len(samples) == 96
        assert all(item["depth_mm"] == 1000 for item in samples)
        assert int(response["snapshot_attempts"]) == 1
        print(json.dumps({
            "status": "PASS",
            "mode": response["mode"],
            "point_count": len(samples),
            "snapshot_attempts": response["snapshot_attempts"],
            "snapshot_roi_px": response["snapshot_roi_px"],
            "snapshot_roi_bytes": response["snapshot_roi_bytes"],
            "snapshot_copy_ms": round(float(response["snapshot_copy_ms"]), 4),
            "vectorized_sample_ms": round(float(response["vectorized_sample_ms"]), 4),
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified M41.2 test tool")
    sub = parser.add_subparsers(dest="command", required=True)

    perf = sub.add_parser("performance", help="watch live timing")
    perf.add_argument("--url", default="http://127.0.0.1:19215/api/app/latest_decision")
    perf.add_argument("--interval", type=float, default=1.0)
    perf.add_argument("--once", action="store_true")
    perf.set_defaults(func=command_performance)

    status = sub.add_parser("status", help="show App/shared-depth state")
    status.add_argument("--url", default="http://127.0.0.1:19215/api/app/status")
    status.set_defaults(func=command_status)

    offline = sub.add_parser("offline", help="validate saved RGB-D dataset")
    offline.add_argument("dataset", help="folder containing images/depth/labels/meta")
    offline.add_argument("--config", default="production/carton_bundle_grasp/config/line.yaml")
    offline.set_defaults(func=command_offline)

    selftest = sub.add_parser("selftest", help="test ROI-snapshot shared-memory path")
    selftest.set_defaults(func=command_selftest)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
