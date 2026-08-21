#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline M41 geometry check for VisionOps RGB-D capture folders.

Expected folders: images/, depth/, labels/, meta/.  YOLO segmentation labels are
used as the top-face mask, so no RKNN model is required for this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_bundle_grasp.config import load_config
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.algorithm import CartonBundleGraspAlgorithm


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline M41 RGB-D geometry validation")
    parser.add_argument("dataset", help="folder containing images/depth/labels/meta")
    parser.add_argument("--config", default="production/carton_bundle_grasp/config/line.yaml")
    args = parser.parse_args()
    root = Path(args.dataset)
    algorithm = CartonBundleGraspAlgorithm(load_config(args.config)["carton_bundle_grasp"]["algorithm"])
    rows: List[Dict[str, object]] = []
    for label_path in sorted((root / "labels").glob("*.txt")):
        stem = label_path.stem
        image = cv2.imread(str(root / "images" / (stem + ".jpg")))
        depth = cv2.imread(str(root / "depth" / (stem + ".png")), cv2.IMREAD_UNCHANGED)
        meta = json.loads((root / "meta" / (stem + ".json")).read_text(encoding="utf-8"))
        if image is None or depth is None:
            continue
        h, w = image.shape[:2]
        values = [float(value) for value in label_path.read_text(encoding="utf-8").split()]
        if len(values) < 9:
            continue
        polygon = [[values[i] * w, values[i + 1] * h] for i in range(1, len(values), 2)]
        runtime = {
            "image": {"width": w, "height": h},
            "detections": [{
                "id": stem,
                "class_id": int(values[0]),
                "class_name": "carton_bundle_top",
                "score": 1.0,
                "mask": {"source": "proto", "polygon": [polygon]},
            }],
        }
        classified = algorithm.classify(runtime)
        if not classified.items:
            rows.append({"capture": stem, "status": "mask_rejected"})
            continue
        item = classified.items[0]
        intr = meta["depth"]["intrinsics_saved"]
        fx, fy, cx, cy = [float(intr[key]) for key in ("fx", "fy", "cx", "cy")]
        positions = []
        samples = []
        radius = algorithm.depth_radius_px
        for u, v in item["plane_sample_points"]:
            x, y = int(round(u)), int(round(v))
            roi = depth[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
            valid = roi[(roi >= algorithm.min_depth_mm) & (roi <= algorithm.max_depth_mm)]
            z = float(np.percentile(valid, algorithm.depth_percentile)) if valid.size else 0.0
            point = [(u - cx) * z / fx, (v - cy) * z / fy, z] if z else [0.0, 0.0, 0.0]
            positions.append(point)
            samples.append({"depth_valid": bool(z), "position_camera": point})
        plane = algorithm.fit_plane(positions)
        z_ref = float(plane["z_ref_mm"])
        rays = [[(u - cx) * z_ref / fx, (v - cy) * z_ref / fy, z_ref] for u, v in item["quad"]]
        corners = algorithm.intersect_corner_rays(rays, plane)
        rectangle = algorithm.reconstruct_rectangle(item["quad"], corners, plane)
        rows.append({
            "capture": stem,
            "status": "ok",
            "observed_length_mm": round(float(rectangle["observed_length_mm"]), 3),
            "observed_width_mm": round(float(rectangle["observed_width_mm"]), 3),
            "plane_rms_mm": round(float(plane["rms_mm"]), 3),
            "plane_inlier_ratio": round(float(plane["inlier_ratio"]), 6),
            "z_ref_mm": round(float(plane["z_ref_mm"]), 3),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
