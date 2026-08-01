"""Interactive/non-interactive calibration of the fixed 3-D box model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore

from ..box_calibration import calibrate_box_model, draw_calibration_overlay
from ..io_utils import CapturePaths, load_rgb_depth_meta, resolve_intrinsics, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M34_new.4：用空箱同步RGB-D标定三维纸箱内壁")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="输出box_model.json")
    parser.add_argument("--points-json", type=Path, help="无GUI模式下提供ROI和点击点")
    parser.add_argument("--overlay", type=Path, help="标定投影图，默认与模型同目录")
    parser.add_argument("--rear-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--bottom-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--rear-corners", nargs=8, type=float, metavar=("TLX", "TLY", "TRX", "TRY", "BRX", "BRY", "BLX", "BLY"))
    parser.add_argument("--front-bottom-edge", nargs=4, type=float, metavar=("LX", "LY", "RX", "RY"))
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--ransac-iterations", type=int, default=1200)
    parser.add_argument("--inlier-threshold-mm", type=float, default=4.0)
    return parser


def _select_roi(image, title: str) -> Tuple[int, int, int, int]:
    roi = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    values = tuple(int(v) for v in roi)
    if values[2] <= 0 or values[3] <= 0:
        raise RuntimeError(f"未选择{title}")
    return values


def _collect_points(image, title: str, labels: Sequence[str]) -> List[List[float]]:
    display = image.copy()
    points: List[List[float]] = []

    def callback(event, x, y, flags, userdata):
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points.append([float(x), float(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, callback)
    while True:
        canvas = display.copy()
        for index, point in enumerate(points):
            p = (int(point[0]), int(point[1]))
            cv2.circle(canvas, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, labels[index], (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        next_label = labels[len(points)] if len(points) < len(labels) else "ENTER confirm"
        cv2.putText(canvas, f"Left click: {next_label}; right click: undo; Esc: cancel", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == len(labels):
            break
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消标定")
    cv2.destroyWindow(title)
    return points


def _reshape_pairs(values: Sequence[float]) -> List[List[float]]:
    return [[float(values[index]), float(values[index + 1])] for index in range(0, len(values), 2)]


def _resolve_inputs(args, rgb) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if args.points_json:
        payload = json.loads(args.points_json.read_text(encoding="utf-8"))
    rear_roi = args.rear_roi or payload.get("rear_roi_xywh")
    bottom_roi = args.bottom_roi or payload.get("bottom_roi_xywh")
    rear_corners = _reshape_pairs(args.rear_corners) if args.rear_corners else payload.get("rear_corners_uv_tl_tr_br_bl")
    front_bottom = _reshape_pairs(args.front_bottom_edge) if args.front_bottom_edge else payload.get("front_bottom_edge_uv_left_right")
    if not args.no_gui:
        rear_roi = rear_roi or _select_roi(rgb, "1/4 Select rear-wall plane ROI")
        bottom_roi = bottom_roi or _select_roi(rgb, "2/4 Select box-bottom plane ROI")
        rear_corners = rear_corners or _collect_points(rgb, "3/4 Rear corners", ["rear TL", "rear TR", "rear BR", "rear BL"])
        front_bottom = front_bottom or _collect_points(rgb, "4/4 Front opening bottom edge", ["front bottom L", "front bottom R"])
    if rear_roi is None or bottom_roi is None or rear_corners is None or front_bottom is None:
        raise ValueError("无GUI模式必须提供rear ROI、bottom ROI、rear corners和front bottom edge")
    return {
        "rear_roi_xywh": [int(v) for v in rear_roi],
        "bottom_roi_xywh": [int(v) for v in bottom_roi],
        "rear_corners_uv_tl_tr_br_bl": [[float(v) for v in point] for point in rear_corners],
        "front_bottom_edge_uv_left_right": [[float(v) for v in point] for point in front_bottom],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths = CapturePaths(args.rgb.stem, args.rgb, args.depth, args.meta)
    rgb, depth, meta = load_rgb_depth_meta(paths)
    intrinsics = resolve_intrinsics(meta, rgb.shape[1], rgb.shape[0])
    inputs = _resolve_inputs(args, rgb)
    model = calibrate_box_model(
        depth,
        intrinsics,
        inputs["rear_roi_xywh"],
        inputs["bottom_roi_xywh"],
        inputs["rear_corners_uv_tl_tr_br_bl"],
        inputs["front_bottom_edge_uv_left_right"],
        source={
            "capture_id": args.rgb.stem,
            "rgb_filename": args.rgb.name,
            "depth_filename": args.depth.name,
            "meta_filename": args.meta.name,
        },
        ransac_iterations=args.ransac_iterations,
        inlier_threshold_mm=args.inlier_threshold_mm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, model.to_dict())
    overlay_path = args.overlay or args.output.with_name(args.output.stem + "_overlay.jpg")
    overlay = draw_calibration_overlay(
        rgb,
        model,
        intrinsics,
        inputs["rear_roi_xywh"],
        inputs["bottom_roi_xywh"],
        inputs["rear_corners_uv_tl_tr_br_bl"],
        inputs["front_bottom_edge_uv_left_right"],
    )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"无法写入标定投影图: {overlay_path}")
    print("3D箱体标定完成")
    print("model:", args.output)
    print("overlay:", overlay_path)
    print("inner_size_mm:", model.inner_size_mm.tolist())
    print("origin_camera_mm:", model.origin_camera_mm.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
