"""M35.2 debug view with complete pre-grasp motion checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .geometry import project_point
from .segmentation import SegmentationInstance


def depth_colormap(depth: np.ndarray, minimum_mm: float = 150.0, maximum_mm: float = 3000.0) -> np.ndarray:
    valid = (depth >= minimum_mm) & (depth <= maximum_mm)
    display = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid].astype(np.float32)
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if high <= low:
            high = low + 1.0
        normalized = np.clip((depth.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
        display[valid] = normalized[valid].astype(np.uint8)
    colored = cv2.applyColorMap(255 - display, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _contours(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _uv(value):
    if not value:
        return None
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def _draw_polygon(image: np.ndarray, points, color, thickness: int = 1) -> None:
    if not points:
        return
    polygon = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], True, color, thickness, cv2.LINE_AA)


def draw_overlay(
    rgb_bgr: np.ndarray,
    instances: Sequence[SegmentationInstance],
    scene: Dict[str, Any],
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    output = rgb_bgr.copy()
    overlay = output.copy()
    for instance in instances:
        if instance.class_name == "foam_ring":
            overlay[instance.mask] = (80, 180, 80)
        elif instance.class_name == "ring_mouth":
            overlay[instance.mask] = (180, 100, 50)
    output = cv2.addWeighted(overlay, 0.20, output, 0.80, 0.0)
    for instance in instances:
        color = (60, 220, 60) if instance.class_name == "foam_ring" else (255, 160, 40)
        cv2.drawContours(output, _contours(instance.mask), -1, color, 1, cv2.LINE_AA)

    box_model = scene.get("box_wall_model") or {}
    if bool(box_model.get("enabled")):
        if str(box_model.get("model") or "") == "calibrated_3d_cuboid":
            front_polygon = box_model.get("front_polygon_uv")
            rear_polygon = box_model.get("rear_polygon_uv")
            if front_polygon:
                _draw_polygon(output, front_polygon, (0, 0, 255), 2)
            if rear_polygon:
                _draw_polygon(output, rear_polygon, (0, 220, 0), 2)
            for edge in box_model.get("edge_lines_uv") or []:
                if len(edge) >= 2:
                    p1 = _uv(edge[0])
                    p2 = _uv(edge[1])
                    if p1 is not None and p2 is not None:
                        cv2.line(output, p1, p2, (255, 220, 0), 1, cv2.LINE_AA)
            anchor = _uv(front_polygon[0]) if front_polygon else None
            if anchor is not None:
                size = box_model.get("inner_size_mm") or {}
                cv2.putText(
                    output,
                    "3D BOX W=%.0f H=%.0f D=%.0f" % (
                        float(size.get("width", 0.0)),
                        float(size.get("height", 0.0)),
                        float(size.get("depth", 0.0)),
                    ),
                    (anchor[0] + 5, max(14, anchor[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            box_polygon = box_model.get("inner_polygon_uv")
            if box_polygon:
                _draw_polygon(output, box_polygon, (255, 220, 0), 2)

    for item in scene.get("instances", []):
        center_uv = (item.get("mouth_ellipse") or {}).get("center_uv")
        if not center_uv:
            continue
        center = _uv(center_uv)
        if center is None:
            continue
        eligible = bool(item.get("eligible"))
        selected = bool(item.get("selected"))
        ring_color = (0, 255, 255) if selected else ((0, 220, 0) if eligible else (0, 0, 255))
        cv2.circle(output, center, 5, ring_color, -1, cv2.LINE_AA)
        pose_source = str((item.get("pose") or {}).get("normal_source") or "depth_plane")
        source_tag = "E" if pose_source == "ellipse_stabilized" else "D"
        best = (item.get("grasp") or {}).get("best_clock_candidate") or {}
        best_hour = best.get("clock_hour")
        label = "R%d %s z=%.0f tilt=%.1f[%s] best=%s" % (
            int(item.get("ring_instance_id", -1)),
            "SELECT" if selected else ("OK" if eligible else "REJECT"),
            float((item.get("ring_center_camera_mm") or [0, 0, 0])[2]),
            float(item.get("tilt_deg") or 0.0),
            source_tag,
            str(best_hour) if best_hour is not None else "-",
        )
        cv2.putText(output, label, (center[0] + 7, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ring_color, 1, cv2.LINE_AA)

        candidates = (item.get("grasp") or {}).get("clock_candidates") or []
        best_index = best.get("clock_index")
        for candidate in candidates:
            inner = _uv(candidate.get("inner_boundary_uv"))
            outer = _uv(candidate.get("outer_boundary_uv"))
            if inner is None or outer is None:
                continue
            midpoint = (int(round((inner[0] + outer[0]) * 0.5)), int(round((inner[1] + outer[1]) * 0.5)))
            valid = bool(candidate.get("valid"))
            warnings = candidate.get("warnings") or []
            evaluation_stage = str(candidate.get("evaluation_stage") or "full")
            deferred = evaluation_stage == "deferred"
            color = (128, 128, 128) if deferred else ((0, 200, 0) if valid else (0, 0, 220))
            if valid and warnings:
                color = (0, 165, 255)
            is_best = best_index is not None and int(candidate.get("clock_index", -1)) == int(best_index)
            if is_best:
                color = (0, 255, 255)
            radius = 6 if is_best else 4
            cv2.circle(output, midpoint, radius, color, -1, cv2.LINE_AA)
            candidate_label = str(candidate.get("clock_hour", "?"))
            if deferred:
                candidate_label += "d"
            reject_reasons = candidate.get("rejection_reasons") or []
            if any(("box_wall" in str(reason) or "box_3d" in str(reason)) for reason in reject_reasons):
                candidate_label += "W"
            if any("neighbor_3d" in str(reason) for reason in reject_reasons):
                candidate_label += "N"
            elif any("neighbor_2d" in str(warning) for warning in (candidate.get("warnings") or [])):
                candidate_label += "n"
            if any("full_gripper_static" in str(reason) for reason in reject_reasons):
                candidate_label += "G"
            if any("full_gripper_motion" in str(reason) for reason in reject_reasons):
                candidate_label += "P"
            cv2.putText(
                output,
                candidate_label,
                (midpoint[0] + 4, midpoint[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
            if is_best:
                cv2.line(output, inner, outer, color, 2, cv2.LINE_AA)
                cv2.circle(output, inner, 4, (255, 100, 0), -1, cv2.LINE_AA)
                cv2.circle(output, outer, 4, (0, 100, 255), -1, cv2.LINE_AA)
                _draw_polygon(output, candidate.get("inner_finger_sweep_polygon_uv"), (255, 100, 0), 2)
                _draw_polygon(output, candidate.get("outer_finger_sweep_polygon_uv"), (0, 100, 255), 2)
                full_static_preview = candidate.get("full_gripper_static") or {}
                component_colors = {
                    "contact_block": (255, 180, 0),
                    "moving_finger": (220, 120, 0),
                    "palm": (180, 0, 180),
                    "mounting_disk": (255, 0, 180),
                    "pneumatic_fitting": (0, 180, 255),
                    "robot_wrist": (180, 180, 255),
                }
                for component in ((full_static_preview.get("_debug") or {}).get("components") or full_static_preview.get("components") or []):
                    _draw_polygon(
                        output,
                        component.get("projection_uv"),
                        component_colors.get(str(component.get("group")), (200, 200, 200)),
                        1,
                    )
                full_motion_preview = candidate.get("full_gripper_motion") or {}
                motion_points = (full_motion_preview.get("_debug") or {}).get("fingertip_midpoint_path_camera_mm") or []
                motion_uv = []
                for point in motion_points:
                    uv = project_point(np.asarray(point, dtype=np.float64), intrinsics)
                    if uv is not None:
                        motion_uv.append((int(round(uv[0])), int(round(uv[1]))))
                if len(motion_uv) >= 2:
                    cv2.polylines(
                        output,
                        [np.asarray(motion_uv, dtype=np.int32).reshape(-1, 1, 2)],
                        False,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(output, motion_uv[0], 3, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(output, motion_uv[-1], 3, (0, 255, 255), -1, cv2.LINE_AA)
                box_wall = candidate.get("box_wall") or {}
                box_clearance = box_wall.get("clearance_mm")
                if box_clearance is None:
                    box_clearance = box_wall.get("minimum_clearance_mm")
                neighbor_3d = candidate.get("neighbor_3d") or {}
                neighbor_clearance = neighbor_3d.get("minimum_clearance_mm")
                full_static = candidate.get("full_gripper_static") or {}
                static_clearance_values = [
                    value for value in (
                        full_static.get("box_minimum_safety_clearance_mm"),
                        full_static.get("neighbor_minimum_clearance_mm"),
                    )
                    if value is not None
                ]
                static_text = (
                    "%.1f" % min(float(value) for value in static_clearance_values)
                    if static_clearance_values else str(full_static.get("status") or "off")
                )
                full_motion = candidate.get("full_gripper_motion") or {}
                motion_clearance_values = [
                    value for value in (
                        full_motion.get("box_minimum_safety_clearance_mm"),
                        full_motion.get("neighbor_minimum_clearance_mm"),
                    )
                    if value is not None
                ]
                motion_text = (
                    "%.1f" % min(float(value) for value in motion_clearance_values)
                    if motion_clearance_values else str(full_motion.get("status") or "off")
                )
                score_text = "clock=%s score=%.1f rim=%.1f gap=%.1f box=%s nbr=%s full=%s path=%s" % (
                    candidate.get("clock_hour"),
                    float(candidate.get("score", 0.0)),
                    float(candidate.get("wall_thickness_mm", 0.0)),
                    float(candidate.get("target_closing_gap_mm", 0.0)),
                    ("%.1f" % float(box_clearance)) if box_clearance is not None else str(box_wall.get("status") or "off"),
                    ("%.1f" % float(neighbor_clearance)) if neighbor_clearance is not None else str(neighbor_3d.get("status") or "off"),
                    static_text,
                    motion_text,
                )
                cv2.putText(output, score_text, (center[0] + 7, center[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        center_3d = item.get("ring_center_camera_mm")
        approach = item.get("approach_vector_camera")
        if center_3d and approach:
            start = np.asarray(center_3d, dtype=np.float64)
            end = start + np.asarray(approach, dtype=np.float64) * 60.0
            end_uv = project_point(end, intrinsics)
            if end_uv:
                cv2.arrowedLine(
                    output,
                    center,
                    (int(round(end_uv[0])), int(round(end_uv[1]))),
                    ring_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.2,
                )

        if not eligible:
            reasons = item.get("rejection_reasons") or []
            if reasons:
                cv2.putText(
                    output,
                    str(reasons[0]),
                    (center[0] + 7, center[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.33,
                    ring_color,
                    1,
                    cv2.LINE_AA,
                )
    return output




def _vector3(value: Any) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _fixed_axis_rod_endpoints(
    center_camera: np.ndarray,
    axis_toward_camera: np.ndarray,
    rod_length_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return fixed-length near/far endpoints around the ring center.

    ``axis_toward_camera`` is a signed unit vector.  The endpoint in its
    positive direction is therefore the near endpoint, while the endpoint in
    the negative direction is the far endpoint.  The function deliberately
    does not stretch or shorten the rod: its projected 2-D length must retain
    the relationship with the 3-D tilt angle.
    """
    length = float(rod_length_mm)
    if not np.isfinite(length) or length <= 0.0:
        return None
    half = 0.5 * length
    near = center_camera + axis_toward_camera * half
    far = center_camera - axis_toward_camera * half
    if not np.isfinite(near).all() or not np.isfinite(far).all():
        return None
    if float(near[2]) <= 1.0 or float(far[2]) <= 1.0:
        return None
    return near, far


def _draw_dashed_line(
    image: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length_px: float,
    gap_length_px: float,
) -> None:
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    delta = second - first
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return
    unit = delta / length
    dash = max(1.0, float(dash_length_px))
    gap = max(0.0, float(gap_length_px))
    cursor = 0.0
    while cursor < length:
        segment_end = min(length, cursor + dash)
        point_a = tuple(np.rint(first + unit * cursor).astype(int).tolist())
        point_b = tuple(np.rint(first + unit * segment_end).astype(int).tolist())
        cv2.line(image, point_a, point_b, color, thickness, cv2.LINE_AA)
        cursor = segment_end + gap


def _put_outlined_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.40,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_paired_axis_overlay(
    rgb_bgr: np.ndarray,
    instances: Sequence[SegmentationInstance],
    scene: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Render M35.4 directed fixed-length 3-D axis rods in the RGB image.

    Only successfully paired ``foam_ring`` / ``ring_mouth`` instances with a
    valid ``ring_axis_toward_camera`` are included.  Each 3-D rod has a fixed
    physical length before projection, so its 2-D length changes with the ring
    tilt.  The near half is a solid red arrow and filled endpoint; the far half
    is a dashed cyan line and open endpoint.
    """
    cfg = dict(config or {})
    rod_length_mm = float(cfg.get("rod_length_mm", 80.0))
    minimum_projected_rod_px = float(cfg.get("minimum_projected_rod_px", 1.0))
    mask_alpha = float(np.clip(float(cfg.get("mask_alpha", 0.18)), 0.0, 1.0))
    line_thickness = max(1, int(cfg.get("line_thickness", 3)))
    dash_length_px = float(cfg.get("far_dash_length_px", 9.0))
    gap_length_px = float(cfg.get("far_gap_length_px", 6.0))
    endpoint_radius_px = max(3, int(cfg.get("endpoint_radius_px", 7)))
    draw_vector_values = bool(cfg.get("draw_vector_values", True))
    draw_depth_values = bool(cfg.get("draw_depth_values", True))

    near_color = (0, 0, 255)
    far_color = (255, 190, 0)
    center_color = (255, 255, 255)
    warning_color = (255, 0, 255)

    output = rgb_bgr.copy()
    tint = output.copy()
    height, width = output.shape[:2]
    instance_by_id = {int(instance.instance_id): instance for instance in instances}
    valid_items: List[Tuple[Mapping[str, Any], SegmentationInstance, SegmentationInstance]] = []
    for item in scene.get("instances", []) or []:
        ring_id = int(item.get("ring_instance_id", -1))
        mouth_id = int(item.get("mouth_instance_id", -1))
        ring = instance_by_id.get(ring_id)
        mouth = instance_by_id.get(mouth_id)
        center_camera = _vector3(item.get("ring_center_camera_mm"))
        axis_camera = _vector3(item.get("ring_axis_toward_camera"))
        if ring is None or mouth is None or center_camera is None or axis_camera is None:
            continue
        valid_items.append((item, ring, mouth))
        tint[ring.mask] = (70, 180, 70)
        tint[mouth.mask] = (0, 140, 230)

    if valid_items and mask_alpha > 0.0:
        output = cv2.addWeighted(tint, mask_alpha, output, 1.0 - mask_alpha, 0.0)

    diagnostics: List[Dict[str, Any]] = []
    for item, ring, mouth in valid_items:
        ring_id = int(item.get("ring_instance_id", -1))
        mouth_id = int(item.get("mouth_instance_id", -1))
        cv2.drawContours(output, _contours(ring.mask), -1, (60, 220, 60), 2, cv2.LINE_AA)
        cv2.drawContours(output, _contours(mouth.mask), -1, (0, 165, 255), 2, cv2.LINE_AA)

        center_camera = np.asarray(item["ring_center_camera_mm"], dtype=np.float64).reshape(3)
        axis_camera = np.asarray(item["ring_axis_toward_camera"], dtype=np.float64).reshape(3)
        axis_camera /= max(float(np.linalg.norm(axis_camera)), 1e-12)
        endpoints = _fixed_axis_rod_endpoints(center_camera, axis_camera, rod_length_mm)
        center_uv_float = project_point(center_camera, intrinsics)
        near_camera = endpoints[0] if endpoints is not None else None
        far_camera = endpoints[1] if endpoints is not None else None
        near_uv_float = project_point(near_camera, intrinsics) if near_camera is not None else None
        far_uv_float = project_point(far_camera, intrinsics) if far_camera is not None else None

        depth_delta_mm = (
            float(far_camera[2] - near_camera[2])
            if near_camera is not None and far_camera is not None
            else None
        )
        if depth_delta_mm is None:
            depth_order_status = "unavailable"
        elif depth_delta_mm > 0.5:
            depth_order_status = "toward_camera"
        elif depth_delta_mm < -0.5:
            depth_order_status = "inconsistent_axis_sign"
        else:
            depth_order_status = "parallel_to_image_plane"

        row: Dict[str, Any] = {
            "ring_instance_id": ring_id,
            "mouth_instance_id": mouth_id,
            "center_camera_mm": center_camera.astype(float).tolist(),
            "axis_toward_camera": axis_camera.astype(float).tolist(),
            "tilt_deg": float(item.get("tilt_deg") or 0.0),
            "normal_source": str((item.get("pose") or {}).get("normal_source") or "unknown"),
            "rod_length_mm": float(rod_length_mm),
            "center_uv": list(center_uv_float) if center_uv_float is not None else None,
            "near_camera_mm": near_camera.astype(float).tolist() if near_camera is not None else None,
            "far_camera_mm": far_camera.astype(float).tolist() if far_camera is not None else None,
            "near_uv": list(near_uv_float) if near_uv_float is not None else None,
            "far_uv": list(far_uv_float) if far_uv_float is not None else None,
            "near_depth_mm": float(near_camera[2]) if near_camera is not None else None,
            "far_depth_mm": float(far_camera[2]) if far_camera is not None else None,
            "depth_delta_far_minus_near_mm": depth_delta_mm,
            "depth_order_status": depth_order_status,
            "status": "projection_failed",
            "projected_rod_length_px": None,
        }
        if center_uv_float is None or near_uv_float is None or far_uv_float is None:
            diagnostics.append(row)
            continue

        center_uv = tuple(np.rint(center_uv_float).astype(int).tolist())
        near_uv = tuple(np.rint(near_uv_float).astype(int).tolist())
        far_uv = tuple(np.rint(far_uv_float).astype(int).tolist())
        projected_length_px = float(
            np.linalg.norm(np.asarray(near_uv_float, dtype=np.float64) - np.asarray(far_uv_float, dtype=np.float64))
        )
        row["projected_rod_length_px"] = projected_length_px
        row["status"] = "near_optical_axis" if projected_length_px < minimum_projected_rod_px else "drawn"
        if depth_order_status == "inconsistent_axis_sign":
            row["status"] = "drawn_sign_warning"

        cv2.circle(output, center_uv, 5, center_color, -1, cv2.LINE_AA)
        cv2.circle(output, center_uv, 5, (20, 20, 20), 1, cv2.LINE_AA)

        if projected_length_px < minimum_projected_rod_px:
            # The rod is close to the optical axis.  Draw the far open endpoint
            # first and the near solid endpoint on top, preserving front/back
            # ordering without inventing an arbitrary image direction.
            cv2.circle(output, center_uv, endpoint_radius_px + 5, far_color, 2, cv2.LINE_AA)
            cv2.circle(output, center_uv, endpoint_radius_px - 1, near_color, -1, cv2.LINE_AA)
            cv2.circle(output, center_uv, endpoint_radius_px - 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            _draw_dashed_line(
                output,
                far_uv_float,
                center_uv_float,
                far_color,
                line_thickness,
                dash_length_px,
                gap_length_px,
            )
            cv2.arrowedLine(
                output,
                center_uv,
                near_uv,
                near_color,
                line_thickness,
                cv2.LINE_AA,
                tipLength=0.24,
            )
            cv2.circle(output, far_uv, endpoint_radius_px, far_color, 2, cv2.LINE_AA)
            cv2.circle(output, near_uv, endpoint_radius_px, near_color, -1, cv2.LINE_AA)
            cv2.circle(output, near_uv, endpoint_radius_px, (255, 255, 255), 1, cv2.LINE_AA)

        source_tag = "E" if row["normal_source"] == "ellipse_stabilized" else "D"
        sign_tag = " SIGN?" if depth_order_status == "inconsistent_axis_sign" else ""
        line1 = "R%d-M%d t=%.1f L2D=%.1f dz=%.0f[%s]%s" % (
            ring_id,
            mouth_id,
            row["tilt_deg"],
            projected_length_px,
            float(depth_delta_mm or 0.0),
            source_tag,
            sign_tag,
        )
        ring_x1, ring_y1, ring_x2, _ = ring.bbox_xyxy
        label_x = max(3, min(width - 4, int(ring_x1)))
        label_y = int(ring_y1) - 8
        if label_y < 18:
            label_y = min(height - 28, int(ring_y1) + 18)
        _put_outlined_text(output, line1, (label_x, label_y), 0.36)
        if draw_vector_values or draw_depth_values:
            fragments: List[str] = []
            if draw_vector_values:
                fragments.append("n=(%+.2f,%+.2f,%+.2f)" % tuple(axis_camera.tolist()))
            if draw_depth_values and near_camera is not None and far_camera is not None:
                fragments.append("Z=%.0f/%.0f" % (near_camera[2], far_camera[2]))
            _put_outlined_text(output, " ".join(fragments), (label_x, label_y + 15), 0.33)
        if depth_order_status == "inconsistent_axis_sign":
            cv2.circle(output, center_uv, endpoint_radius_px + 10, warning_color, 2, cv2.LINE_AA)
        diagnostics.append(row)

    summary = "M35.4 directed fixed-length 3D rods: %d / matched pairs: %d / L=%.0fmm" % (
        len(valid_items),
        int(scene.get("matched_pairs", len(valid_items)) or 0),
        rod_length_mm,
    )
    _put_outlined_text(output, summary, (8, 20), 0.50)
    _put_outlined_text(
        output,
        "red solid=near/toward camera, cyan dashed=far; L2D reflects 3D direction and perspective",
        (8, 40),
        0.38,
    )
    return output, diagnostics
