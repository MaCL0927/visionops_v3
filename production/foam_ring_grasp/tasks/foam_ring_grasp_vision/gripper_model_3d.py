"""M35.2 complete gripper geometry, static pose and pre-grasp motion checks.

The grasp-local frame is the same frame returned to the robot side:

* +X: finger closing axis, from the inner finger to the outer finger.
* +Y: lateral axis completing a right-handed frame.
* +Z: approach axis, from pregrasp toward the ring.
* origin: midpoint between the two fingertips at the commanded insertion depth.

All parts upstream of the fingertips therefore have negative local Z.
M35.1 final static-pose checks are retained. M35.2 additionally checks the
complete motion from pregrasp through closing on the target. Post-grasp lift is
intentionally omitted by task definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .box_model_3d import BoxModel3D


@dataclass(frozen=True)
class CollisionPrimitive:
    name: str
    group: str
    kind: str
    center_local_mm: np.ndarray
    rotation_local: np.ndarray
    size_mm: Optional[np.ndarray] = None
    radius_mm: Optional[float] = None
    length_mm: Optional[float] = None
    extra_margin_mm: float = 0.0
    allow_target_contact: bool = False


@dataclass(frozen=True)
class StaticGripperModel:
    opening_mm: float
    finger_angle_deg: float
    pivot_to_tip_axial_mm: float
    disk_lower_z_mm: float
    disk_upper_z_mm: float
    components: Tuple[CollisionPrimitive, ...]
    diagnostics: Dict[str, Any]


def _float(raw: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _int(raw: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(raw.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} is a zero vector")
    return vector / norm


def _obb_rotation_from_link(direction_local: np.ndarray) -> np.ndarray:
    z_axis = _unit(direction_local, "finger_link_direction")
    y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = _unit(np.cross(y_axis, z_axis), "finger_link_x")
    y_axis = _unit(np.cross(z_axis, x_axis), "finger_link_y")
    return np.column_stack((x_axis, y_axis, z_axis))


def _identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _rotation_with_local_z_aligned_to(axis_world: Sequence[float]) -> np.ndarray:
    """Return a right-handed rotation whose local +Z axis matches axis_world.

    The cylinder primitives in this module are parameterized along their local
    +Z axis. This helper lets us mount cylindrical components cleanly on any of
    the gripper body faces while keeping the rest of the collision pipeline
    unchanged.
    """
    z_axis = _unit(axis_world, "cylinder_axis_world")
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, reference))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(reference, z_axis)
    if float(np.linalg.norm(x_axis)) <= 1e-12:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = np.cross(reference, z_axis)
    x_axis = _unit(x_axis, "cylinder_axis_x")
    y_axis = _unit(np.cross(z_axis, x_axis), "cylinder_axis_y")
    return np.column_stack((x_axis, y_axis, z_axis))


def build_static_gripper_model(
    opening_mm: float,
    geometry_cfg: Mapping[str, Any],
    collision_cfg: Optional[Mapping[str, Any]] = None,
) -> StaticGripperModel:
    """Build the measured final-pose gripper model for one commanded gap.

    New measurements supplied for M35.1 take precedence over the older M34
    finger-only approximations.
    """
    collision_cfg = collision_cfg or {}
    opening = float(opening_mm)

    contact = dict(geometry_cfg.get("contact_block") or {})
    moving = dict(geometry_cfg.get("moving_finger") or {})
    palm = dict(geometry_cfg.get("palm") or {})
    disk = dict(geometry_cfg.get("mounting_disk") or {})
    fitting = dict(geometry_cfg.get("pneumatic_fitting") or {})
    wrist = dict(geometry_cfg.get("robot_wrist") or {})
    kinematics = dict(geometry_cfg.get("finger_kinematics") or {})
    margins = dict(collision_cfg.get("component_extra_margin_mm") or {})

    contact_width = _float(contact, "width_x_mm", 15.0)
    contact_thickness = _float(contact, "thickness_y_mm", 20.0)
    contact_length = _float(contact, "length_z_mm", 35.0)
    link_width = _float(moving, "width_x_mm", 16.0)
    link_thickness = _float(moving, "thickness_y_mm", 20.0)
    link_length = _float(moving, "pivot_to_tip_mm", 90.0)
    pivot_separation = _float(kinematics, "pivot_center_separation_mm", 36.0)
    pivot_from_disk_lower = _float(kinematics, "pivot_center_from_disk_lower_mm", 55.0)

    # The measured opening is the distance between the inward-facing black
    # contact surfaces. The contact-block center is half a block-width farther
    # from the gripper centerline.
    tip_center_abs_x = 0.5 * opening + 0.5 * contact_width
    pivot_abs_x = 0.5 * pivot_separation
    lateral_delta = tip_center_abs_x - pivot_abs_x
    if abs(lateral_delta) >= link_length:
        raise ValueError(
            "invalid finger geometry: opening requires a lateral pivot-to-tip "
            f"offset {lateral_delta:.3f} mm for a {link_length:.3f} mm link"
        )
    axial = math.sqrt(max(0.0, link_length * link_length - lateral_delta * lateral_delta))
    finger_angle = math.degrees(math.asin(float(np.clip(lateral_delta / link_length, -1.0, 1.0))))

    pivot_z = -axial
    disk_lower_z = pivot_z - pivot_from_disk_lower
    disk_thickness = _float(disk, "length_z_mm", 20.0)
    disk_upper_z = disk_lower_z - disk_thickness

    components: List[CollisionPrimitive] = []
    for sign, side in ((-1.0, "inner"), (1.0, "outer")):
        pivot = np.asarray([sign * pivot_abs_x, 0.0, pivot_z], dtype=np.float64)
        tip = np.asarray([sign * tip_center_abs_x, 0.0, 0.0], dtype=np.float64)
        direction = tip - pivot
        rotation = _obb_rotation_from_link(direction)
        link_center = 0.5 * (pivot + tip)
        contact_center = tip - rotation[:, 2] * (0.5 * contact_length)
        components.append(
            CollisionPrimitive(
                name=f"{side}_moving_finger",
                group="moving_finger",
                kind="obb",
                center_local_mm=link_center,
                rotation_local=rotation,
                size_mm=np.asarray([link_width, link_thickness, link_length], dtype=np.float64),
                extra_margin_mm=float(margins.get("moving_finger", 4.0)),
            )
        )
        components.append(
            CollisionPrimitive(
                name=f"{side}_contact_block",
                group="contact_block",
                kind="obb",
                center_local_mm=contact_center,
                rotation_local=rotation,
                size_mm=np.asarray([contact_width, contact_thickness, contact_length], dtype=np.float64),
                extra_margin_mm=float(margins.get("contact_block", 2.0)),
                allow_target_contact=True,
            )
        )

    palm_length = _float(palm, "length_z_mm", 60.0)
    palm_center = np.asarray([0.0, 0.0, disk_lower_z + 0.5 * palm_length], dtype=np.float64)
    components.append(
        CollisionPrimitive(
            name="palm",
            group="palm",
            kind="obb",
            center_local_mm=palm_center,
            rotation_local=_identity_rotation(),
            size_mm=np.asarray(
                [
                    _float(palm, "width_x_mm", 50.0),
                    _float(palm, "thickness_y_mm", 35.0),
                    palm_length,
                ],
                dtype=np.float64,
            ),
            extra_margin_mm=float(margins.get("palm", 6.0)),
        )
    )

    components.append(
        CollisionPrimitive(
            name="mounting_disk",
            group="mounting_disk",
            kind="cylinder",
            center_local_mm=np.asarray([0.0, 0.0, disk_lower_z - 0.5 * disk_thickness], dtype=np.float64),
            rotation_local=_identity_rotation(),
            radius_mm=0.5 * _float(disk, "diameter_mm", 70.0),
            length_mm=disk_thickness,
            extra_margin_mm=float(margins.get("mounting_disk", 8.0)),
        )
    )

    if bool(fitting.get("enabled", True)):
        fitting_length = _float(fitting, "protrusion_length_mm", 30.0)
        fitting_radius = 0.5 * _float(fitting, "diameter_mm", 10.0)
        palm_half_x = 0.5 * _float(palm, "width_x_mm", 50.0)
        palm_half_y = 0.5 * _float(palm, "thickness_y_mm", 35.0)

        direction_name = str(fitting.get("axis") or "positive_x").lower()
        if direction_name in {"positive_x", "+x", "x", "right"}:
            axis_world = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            fitting_center_base = np.asarray([palm_half_x + 0.5 * fitting_length, 0.0, 0.0], dtype=np.float64)
            spacing_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            spacing_default_key = "center_spacing_y_mm"
        elif direction_name in {"negative_x", "-x", "left"}:
            axis_world = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
            fitting_center_base = np.asarray([-(palm_half_x + 0.5 * fitting_length), 0.0, 0.0], dtype=np.float64)
            spacing_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            spacing_default_key = "center_spacing_y_mm"
        elif direction_name in {"positive_y", "+y", "y", "front"}:
            axis_world = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            fitting_center_base = np.asarray([0.0, palm_half_y + 0.5 * fitting_length, 0.0], dtype=np.float64)
            spacing_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            spacing_default_key = "center_spacing_x_mm"
        elif direction_name in {"negative_y", "-y", "back"}:
            axis_world = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
            fitting_center_base = np.asarray([0.0, -(palm_half_y + 0.5 * fitting_length), 0.0], dtype=np.float64)
            spacing_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            spacing_default_key = "center_spacing_x_mm"
        else:
            raise ValueError(f"unsupported pneumatic_fitting.axis: {direction_name}")

        fitting_rotation = _rotation_with_local_z_aligned_to(axis_world)
        fitting_z = disk_lower_z + _float(fitting, "center_from_disk_lower_mm", 25.0)
        fitting_count = max(1, int(round(_float(fitting, "count", 2.0))))
        spacing = _float(
            fitting,
            spacing_default_key,
            _float(fitting, "center_spacing_mm", 20.0),
        )
        if fitting_count <= 1:
            fitting_offsets = [0.0]
        else:
            start = -0.5 * spacing * float(fitting_count - 1)
            fitting_offsets = [start + spacing * float(i) for i in range(fitting_count)]
        for idx, offset in enumerate(fitting_offsets):
            name = "pneumatic_fitting" if fitting_count == 1 else f"pneumatic_fitting_{idx + 1}"
            center = fitting_center_base + spacing_axis * float(offset)
            center[2] = fitting_z
            components.append(
                CollisionPrimitive(
                    name=name,
                    group="pneumatic_fitting",
                    kind="cylinder",
                    center_local_mm=center,
                    rotation_local=fitting_rotation,
                    radius_mm=fitting_radius,
                    length_mm=fitting_length,
                    extra_margin_mm=float(margins.get("pneumatic_fitting", 8.0)),
                )
            )

    if bool(wrist.get("enabled", True)):
        wrist_length = _float(wrist, "length_z_mm", 200.0)
        components.append(
            CollisionPrimitive(
                name="robot_wrist",
                group="robot_wrist",
                kind="cylinder",
                center_local_mm=np.asarray([0.0, 0.0, disk_upper_z - 0.5 * wrist_length], dtype=np.float64),
                rotation_local=_identity_rotation(),
                radius_mm=0.5 * _float(wrist, "diameter_mm", 100.0),
                length_mm=wrist_length,
                extra_margin_mm=float(margins.get("robot_wrist", 10.0)),
            )
        )

    diagnostics = {
        "frame_origin": "fingertip_midpoint_at_final_inserted_pose",
        "frame_axes": "+X closing, +Y lateral, +Z approach/toward object",
        "opening_reference": "distance_between_inward_facing_contact_surfaces",
        "opening_mm": opening,
        "contact_tip_center_abs_x_mm": tip_center_abs_x,
        "pivot_center_abs_x_mm": pivot_abs_x,
        "pivot_to_tip_lateral_mm": lateral_delta,
        "pivot_to_tip_axial_mm": axial,
        "finger_angle_deg_from_tool_z": finger_angle,
        "pivot_z_mm": pivot_z,
        "disk_lower_z_mm": disk_lower_z,
        "disk_upper_z_mm": disk_upper_z,
        "disk_upper_to_tip_mm": -disk_upper_z,
        "disk_lower_to_tip_mm": -disk_lower_z,
    }
    return StaticGripperModel(
        opening_mm=opening,
        finger_angle_deg=finger_angle,
        pivot_to_tip_axial_mm=axial,
        disk_lower_z_mm=disk_lower_z,
        disk_upper_z_mm=disk_upper_z,
        components=tuple(components),
        diagnostics=diagnostics,
    )


def _sample_obb_surface(primitive: CollisionPrimitive, resolution: int) -> np.ndarray:
    assert primitive.size_mm is not None
    half = 0.5 * np.asarray(primitive.size_mm, dtype=np.float64)
    n = max(2, int(resolution))
    coordinates = [np.linspace(-half[index], half[index], n) for index in range(3)]
    rows: List[List[float]] = []
    for axis in range(3):
        other = [idx for idx in range(3) if idx != axis]
        for sign in (-1.0, 1.0):
            for first in coordinates[other[0]]:
                for second in coordinates[other[1]]:
                    point = [0.0, 0.0, 0.0]
                    point[axis] = sign * half[axis]
                    point[other[0]] = float(first)
                    point[other[1]] = float(second)
                    rows.append(point)
    local = np.asarray(rows, dtype=np.float64)
    return local @ primitive.rotation_local.T + primitive.center_local_mm


def _sample_cylinder_surface(primitive: CollisionPrimitive, radial_samples: int, axial_samples: int) -> np.ndarray:
    assert primitive.radius_mm is not None and primitive.length_mm is not None
    angles = np.linspace(0.0, 2.0 * math.pi, max(8, int(radial_samples)), endpoint=False)
    z_values = np.linspace(-0.5 * primitive.length_mm, 0.5 * primitive.length_mm, max(2, int(axial_samples)))
    rows: List[List[float]] = []
    for z in z_values:
        for angle in angles:
            rows.append([primitive.radius_mm * math.cos(angle), primitive.radius_mm * math.sin(angle), z])
    # Include cap interior samples, not only the edge, so a wall crossing a cap
    # is visible in the conservative static check.
    for z in (-0.5 * primitive.length_mm, 0.5 * primitive.length_mm):
        for ratio in (0.0, 0.5, 1.0):
            for angle in angles:
                rows.append([ratio * primitive.radius_mm * math.cos(angle), ratio * primitive.radius_mm * math.sin(angle), z])
    local = np.asarray(rows, dtype=np.float64)
    return local @ primitive.rotation_local.T + primitive.center_local_mm


def sample_component_surface_local(
    primitive: CollisionPrimitive,
    obb_resolution: int = 4,
    cylinder_radial_samples: int = 24,
    cylinder_axial_samples: int = 5,
) -> np.ndarray:
    if primitive.kind == "obb":
        return _sample_obb_surface(primitive, obb_resolution)
    if primitive.kind == "cylinder":
        return _sample_cylinder_surface(primitive, cylinder_radial_samples, cylinder_axial_samples)
    raise ValueError(f"unsupported primitive kind: {primitive.kind}")


def _component_to_camera(
    primitive: CollisionPrimitive,
    origin_camera_mm: np.ndarray,
    rotation_camera_from_grasp: np.ndarray,
) -> Dict[str, Any]:
    center_camera = rotation_camera_from_grasp @ primitive.center_local_mm + origin_camera_mm
    rotation_camera = rotation_camera_from_grasp @ primitive.rotation_local
    return {
        "primitive": primitive,
        "center_camera_mm": center_camera,
        "rotation_camera": rotation_camera,
    }


def _component_points_camera(
    primitive: CollisionPrimitive,
    origin_camera_mm: np.ndarray,
    rotation_camera_from_grasp: np.ndarray,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    points_local = sample_component_surface_local(
        primitive,
        obb_resolution=_int(cfg, "obb_surface_resolution", 4),
        cylinder_radial_samples=_int(cfg, "cylinder_radial_samples", 24),
        cylinder_axial_samples=_int(cfg, "cylinder_axial_samples", 5),
    )
    return points_local @ rotation_camera_from_grasp.T + origin_camera_mm


def _wall_clearances(points_box: np.ndarray, size_mm: np.ndarray) -> Dict[str, np.ndarray]:
    width, height, depth = np.asarray(size_mm, dtype=np.float64).tolist()
    return {
        "left": points_box[:, 0],
        "right": width - points_box[:, 0],
        "top": points_box[:, 1],
        "bottom": height - points_box[:, 1],
        "back": depth - points_box[:, 2],
    }


def _check_component_against_box(
    primitive: CollisionPrimitive,
    points_camera: np.ndarray,
    model: Optional[BoxModel3D],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    if model is None:
        return {
            "component": primitive.name,
            "group": primitive.group,
            "status": "unconfigured",
            "minimum_clearance_mm": None,
            "minimum_safety_clearance_mm": None,
            "nearest_wall": None,
        }
    points_box = model.camera_to_box(points_camera)
    front_tolerance = _float(cfg, "front_entry_tolerance_mm", 2.0)
    active = points_box[:, 2] >= -front_tolerance
    if not np.any(active):
        return {
            "component": primitive.name,
            "group": primitive.group,
            "status": "outside_front",
            "active_point_count": 0,
            "minimum_clearance_mm": None,
            "minimum_safety_clearance_mm": None,
            "nearest_wall": None,
            "physical_intersection": False,
            "safety_margin_violation": False,
        }
    q = points_box[active]
    clearances = _wall_clearances(q, model.inner_size_mm)
    physical_rows: List[Tuple[float, str, int]] = []
    safe_rows: List[Tuple[float, str, int]] = []
    for wall, values in clearances.items():
        margin = float(model.safety_margin_mm.get(wall, 0.0)) + float(primitive.extra_margin_mm)
        for index, value in enumerate(values.tolist()):
            physical_rows.append((float(value), wall, index))
            safe_rows.append((float(value) - margin, wall, index))
    physical_min, physical_wall, physical_index = min(physical_rows, key=lambda row: row[0])
    safe_min, safe_wall, safe_index = min(safe_rows, key=lambda row: row[0])
    physical_intersection = physical_min < 0.0
    safety_violation = safe_min < 0.0 and not physical_intersection
    status = "intersects" if physical_intersection else ("too_close" if safety_violation else "clear")
    selected_index = physical_index if physical_intersection else safe_index
    active_camera = points_camera[active]
    active_box = points_box[active]
    return {
        "component": primitive.name,
        "group": primitive.group,
        "status": status,
        "active_point_count": int(np.count_nonzero(active)),
        "minimum_clearance_mm": float(physical_min),
        "minimum_safety_clearance_mm": float(safe_min),
        "nearest_wall": str(physical_wall if physical_intersection else safe_wall),
        "nearest_point_camera_mm": active_camera[selected_index].astype(float).tolist(),
        "nearest_point_box_mm": active_box[selected_index].astype(float).tolist(),
        "physical_intersection": bool(physical_intersection),
        "safety_margin_violation": bool(safety_violation),
        "component_extra_margin_mm": float(primitive.extra_margin_mm),
    }


def _points_to_obb_distance(
    points_camera: np.ndarray,
    center_camera: np.ndarray,
    rotation_camera: np.ndarray,
    size_mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    local = (points_camera - center_camera) @ rotation_camera
    half = 0.5 * np.asarray(size_mm, dtype=np.float64)
    outside = np.maximum(np.abs(local) - half.reshape(1, 3), 0.0)
    distances = np.linalg.norm(outside, axis=1)
    inside = np.all(np.abs(local) <= half.reshape(1, 3), axis=1)
    return distances, inside


def _points_to_cylinder_distance(
    points_camera: np.ndarray,
    center_camera: np.ndarray,
    rotation_camera: np.ndarray,
    radius_mm: float,
    length_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    local = (points_camera - center_camera) @ rotation_camera
    radial = np.linalg.norm(local[:, :2], axis=1)
    radial_out = np.maximum(radial - float(radius_mm), 0.0)
    axial_out = np.maximum(np.abs(local[:, 2]) - 0.5 * float(length_mm), 0.0)
    distances = np.sqrt(radial_out * radial_out + axial_out * axial_out)
    inside = (radial <= float(radius_mm)) & (np.abs(local[:, 2]) <= 0.5 * float(length_mm))
    return distances, inside


def _check_component_against_neighbors(
    primitive: CollisionPrimitive,
    center_camera: np.ndarray,
    rotation_camera: np.ndarray,
    neighbor_clouds: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    minimum_points = max(1, _int(cfg, "minimum_points_per_instance", 12))
    minimum_collision_points = max(1, _int(cfg, "minimum_collision_points", 4))
    tolerance = _float(cfg, "intersection_tolerance_mm", 1.5)
    minimum_clearance = _float(cfg, "minimum_clearance_mm", 4.0)
    checks: List[Dict[str, Any]] = []
    colliding_ids = set()
    for cloud in neighbor_clouds:
        points = cloud.get("points_camera")
        if not isinstance(points, np.ndarray) or len(points) < minimum_points:
            continue
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if primitive.kind == "obb":
            assert primitive.size_mm is not None
            distances, inside = _points_to_obb_distance(points, center_camera, rotation_camera, primitive.size_mm)
        else:
            assert primitive.radius_mm is not None and primitive.length_mm is not None
            distances, inside = _points_to_cylinder_distance(
                points,
                center_camera,
                rotation_camera,
                primitive.radius_mm,
                primitive.length_mm,
            )
        collision_mask = distances <= tolerance
        collision_count = int(np.count_nonzero(collision_mask))
        collision = collision_count >= minimum_collision_points
        if collision:
            colliding_ids.add(int(cloud["instance_id"]))
        sorted_distances = np.sort(distances)
        kth = min(minimum_collision_points - 1, len(sorted_distances) - 1)
        nearest_index = int(np.argmin(distances))
        checks.append(
            {
                "component": primitive.name,
                "group": primitive.group,
                "neighbor_instance_id": int(cloud["instance_id"]),
                "neighbor_point_count": int(len(points)),
                "inside_point_count": int(np.count_nonzero(inside)),
                "collision_point_count": collision_count,
                "collision": bool(collision),
                "raw_minimum_clearance_mm": float(distances[nearest_index]),
                "robust_minimum_clearance_mm": float(sorted_distances[kth]),
                "nearest_point_camera_mm": points[nearest_index].astype(float).tolist(),
            }
        )
    if not neighbor_clouds:
        return {
            "component": primitive.name,
            "group": primitive.group,
            "status": "clear",
            "reason": "no_neighbor_instances",
            "minimum_clearance_mm": None,
            "colliding_instance_ids": [],
            "checks": [],
        }
    if not checks:
        return {
            "component": primitive.name,
            "group": primitive.group,
            "status": "unknown",
            "reason": "insufficient_neighbor_depth_support",
            "minimum_clearance_mm": None,
            "colliding_instance_ids": [],
            "checks": [],
        }
    nearest = min(checks, key=lambda row: float(row["robust_minimum_clearance_mm"]))
    robust_min = float(nearest["robust_minimum_clearance_mm"])
    status = "intersects" if colliding_ids else ("too_close" if robust_min < minimum_clearance else "clear")
    return {
        "component": primitive.name,
        "group": primitive.group,
        "status": status,
        "minimum_clearance_mm": robust_min,
        "raw_minimum_clearance_mm": min(float(row["raw_minimum_clearance_mm"]) for row in checks),
        "required_clearance_mm": minimum_clearance,
        "nearest_instance_id": int(nearest["neighbor_instance_id"]),
        "colliding_instance_ids": sorted(colliding_ids),
        "checks": checks if bool(cfg.get("include_instance_checks_in_json", False)) else [],
    }


def _project_hull(points_camera: np.ndarray, intrinsics: Optional[Mapping[str, float]]) -> Optional[List[List[float]]]:
    if not intrinsics or len(points_camera) == 0:
        return None
    z = points_camera[:, 2]
    valid = z > 1e-6
    if int(np.count_nonzero(valid)) < 3:
        return None
    points = points_camera[valid]
    u = float(intrinsics["fx"]) * points[:, 0] / points[:, 2] + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * points[:, 1] / points[:, 2] + float(intrinsics["cy"])
    hull = cv2.convexHull(np.rint(np.column_stack((u, v))).astype(np.int32).reshape(-1, 1, 2))
    return hull.reshape(-1, 2).astype(float).tolist()


def check_full_gripper_static_final_pose(
    origin_camera_mm: Sequence[float],
    closing_axis_camera: Sequence[float],
    lateral_axis_camera: Sequence[float],
    approach_axis_camera: Sequence[float],
    opening_mm: float,
    box_model: Optional[BoxModel3D],
    neighbor_clouds: Sequence[Mapping[str, Any]],
    geometry_cfg: Mapping[str, Any],
    collision_cfg: Mapping[str, Any],
    intrinsics: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    enabled = bool(collision_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "status": "disabled", "static_final_pose_only": True}

    x_axis = _unit(closing_axis_camera, "closing_axis_camera")
    z_axis = _unit(approach_axis_camera, "approach_axis_camera")
    x_axis = _unit(x_axis - z_axis * float(np.dot(x_axis, z_axis)), "closing_axis_orthogonal")
    y_hint = _unit(lateral_axis_camera, "lateral_axis_camera")
    y_axis = _unit(y_hint - x_axis * float(np.dot(y_hint, x_axis)) - z_axis * float(np.dot(y_hint, z_axis)), "lateral_axis_orthogonal")
    # Rebuild a guaranteed right-handed frame.
    y_axis = _unit(np.cross(z_axis, x_axis), "lateral_axis_right_handed")
    x_axis = _unit(np.cross(y_axis, z_axis), "closing_axis_right_handed")
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    origin = np.asarray(origin_camera_mm, dtype=np.float64).reshape(3)

    try:
        model = build_static_gripper_model(opening_mm, geometry_cfg, collision_cfg)
    except Exception as error:
        return {
            "enabled": True,
            "status": "unconfigured",
            "reason": str(error),
            "static_final_pose_only": True,
            "hard_reject_on_unconfigured": bool(collision_cfg.get("hard_reject_on_unconfigured", True)),
        }

    component_rows: List[Dict[str, Any]] = []
    box_rows: List[Dict[str, Any]] = []
    neighbor_rows: List[Dict[str, Any]] = []
    for primitive in model.components:
        transformed = _component_to_camera(primitive, origin, rotation)
        points_camera = _component_points_camera(primitive, origin, rotation, collision_cfg)
        box_check = _check_component_against_box(primitive, points_camera, box_model, collision_cfg)
        neighbor_check = _check_component_against_neighbors(
            primitive,
            np.asarray(transformed["center_camera_mm"], dtype=np.float64),
            np.asarray(transformed["rotation_camera"], dtype=np.float64),
            neighbor_clouds,
            collision_cfg,
        )
        box_rows.append(box_check)
        neighbor_rows.append(neighbor_check)
        component_rows.append(
            {
                "name": primitive.name,
                "group": primitive.group,
                "kind": primitive.kind,
                "center_local_mm": primitive.center_local_mm.astype(float).tolist(),
                "center_camera_mm": np.asarray(transformed["center_camera_mm"], dtype=float).tolist(),
                "rotation_camera_rows": np.asarray(transformed["rotation_camera"], dtype=float).tolist(),
                "size_mm": primitive.size_mm.astype(float).tolist() if primitive.size_mm is not None else None,
                "radius_mm": primitive.radius_mm,
                "length_mm": primitive.length_mm,
                "extra_margin_mm": float(primitive.extra_margin_mm),
                "projection_uv": _project_hull(points_camera, intrinsics),
                "box": box_check,
                "neighbor": neighbor_check,
            }
        )

    box_intersections = [row for row in box_rows if row.get("status") == "intersects"]
    box_close = [row for row in box_rows if row.get("status") == "too_close"]
    box_unconfigured = [row for row in box_rows if row.get("status") == "unconfigured"]
    neighbor_intersections = [row for row in neighbor_rows if row.get("status") == "intersects"]
    neighbor_close = [row for row in neighbor_rows if row.get("status") == "too_close"]
    neighbor_unknown = [row for row in neighbor_rows if row.get("status") == "unknown"]

    if box_unconfigured:
        box_status = "unconfigured"
    elif box_intersections:
        box_status = "intersects"
    elif box_close:
        box_status = "too_close"
    else:
        box_status = "clear"
    if neighbor_intersections:
        neighbor_status = "intersects"
    elif neighbor_close:
        neighbor_status = "too_close"
    elif neighbor_unknown:
        neighbor_status = "unknown"
    else:
        neighbor_status = "clear"

    box_finite = [float(row["minimum_clearance_mm"]) for row in box_rows if row.get("minimum_clearance_mm") is not None]
    box_safe_finite = [float(row["minimum_safety_clearance_mm"]) for row in box_rows if row.get("minimum_safety_clearance_mm") is not None]
    neighbor_finite = [float(row["minimum_clearance_mm"]) for row in neighbor_rows if row.get("minimum_clearance_mm") is not None]
    priorities = {"unconfigured": 4, "intersects": 3, "too_close": 2, "unknown": 1, "clear": 0, "outside_front": 0}
    worst_box = max(
        box_rows,
        key=lambda row: (
            priorities.get(str(row.get("status")), -1),
            -float(row.get("minimum_safety_clearance_mm") if row.get("minimum_safety_clearance_mm") is not None else 1e12),
        ),
        default=None,
    )
    worst_neighbor = max(
        neighbor_rows,
        key=lambda row: (
            priorities.get(str(row.get("status")), -1),
            -float(row.get("minimum_clearance_mm") if row.get("minimum_clearance_mm") is not None else 1e12),
        ),
        default=None,
    )

    hard_box = (
        (box_status == "intersects" and bool(collision_cfg.get("hard_reject_box_intersection", True)))
        or (box_status == "too_close" and bool(collision_cfg.get("hard_reject_box_clearance", True)))
        or (box_status == "unconfigured" and bool(collision_cfg.get("hard_reject_on_unconfigured", True)))
    )
    hard_neighbor = (
        (neighbor_status == "intersects" and bool(collision_cfg.get("hard_reject_neighbor_intersection", True)))
        or (neighbor_status == "too_close" and bool(collision_cfg.get("hard_reject_neighbor_clearance", True)))
        or (neighbor_status == "unknown" and bool(collision_cfg.get("hard_reject_neighbor_unknown", False)))
    )
    status = "rejected" if (hard_box or hard_neighbor) else ("warning" if box_status != "clear" or neighbor_status != "clear" else "clear")

    mount_origin = rotation @ np.asarray([0.0, 0.0, model.disk_upper_z_mm], dtype=np.float64) + origin
    disk_lower_origin = rotation @ np.asarray([0.0, 0.0, model.disk_lower_z_mm], dtype=np.float64) + origin
    mount_transform = np.eye(4, dtype=np.float64)
    mount_transform[:3, :3] = rotation
    mount_transform[:3, 3] = mount_origin

    return {
        "enabled": True,
        "status": status,
        "static_final_pose_only": True,
        "dynamic_sweeps_checked": False,
        "opening_mm": float(opening_mm),
        "finger_angle_deg": float(model.finger_angle_deg),
        "pivot_to_tip_axial_mm": float(model.pivot_to_tip_axial_mm),
        "box_status": box_status,
        "neighbor_status": neighbor_status,
        "box_minimum_clearance_mm": min(box_finite) if box_finite else None,
        "box_minimum_safety_clearance_mm": min(box_safe_finite) if box_safe_finite else None,
        "neighbor_minimum_clearance_mm": min(neighbor_finite) if neighbor_finite else None,
        "box_worst_component": worst_box.get("component") if worst_box else None,
        "box_nearest_wall": worst_box.get("nearest_wall") if worst_box else None,
        "neighbor_worst_component": worst_neighbor.get("component") if worst_neighbor else None,
        "neighbor_nearest_instance_id": worst_neighbor.get("nearest_instance_id") if worst_neighbor else None,
        "box_colliding_components": [row["component"] for row in box_intersections],
        "box_too_close_components": [row["component"] for row in box_close],
        "neighbor_colliding_components": [row["component"] for row in neighbor_intersections],
        "neighbor_too_close_components": [row["component"] for row in neighbor_close],
        "neighbor_colliding_instance_ids": sorted(
            {
                int(value)
                for row in neighbor_rows
                for value in (row.get("colliding_instance_ids") or [])
            }
        ),
        "hard_reject": bool(hard_box or hard_neighbor),
        "hard_reject_box": bool(hard_box),
        "hard_reject_neighbor": bool(hard_neighbor),
        "model_diagnostics": model.diagnostics,
        "mounting_interface_frame_camera": {
            "definition": "mounting_disk_upper_surface_center",
            "origin_camera_mm": mount_origin.astype(float).tolist(),
            "disk_lower_center_camera_mm": disk_lower_origin.astype(float).tolist(),
            "x_closing_axis_camera": x_axis.astype(float).tolist(),
            "y_lateral_axis_camera": y_axis.astype(float).tolist(),
            "z_toward_fingertips_camera": z_axis.astype(float).tolist(),
            "rotation_matrix_rows": rotation.astype(float).tolist(),
            "T_camera_mount_rows": mount_transform.astype(float).tolist(),
            "mount_to_fingertip_midpoint_mm": float(-model.disk_upper_z_mm),
        },
        "component_count": len(component_rows),
        "component_summaries": [
            {
                "name": row["name"],
                "group": row["group"],
                "kind": row["kind"],
                "center_camera_mm": row["center_camera_mm"],
                "box_status": (row.get("box") or {}).get("status"),
                "box_clearance_mm": (row.get("box") or {}).get("minimum_clearance_mm"),
                "box_safety_clearance_mm": (row.get("box") or {}).get("minimum_safety_clearance_mm"),
                "neighbor_status": (row.get("neighbor") or {}).get("status"),
                "neighbor_clearance_mm": (row.get("neighbor") or {}).get("minimum_clearance_mm"),
            }
            for row in component_rows
        ],
        "components": component_rows if bool(collision_cfg.get("include_component_details_in_json", False)) else [],
        "_debug": {"components": component_rows},
        "limitations": [
            "final_static_pose_only",
            "approach_open_insert_close_and_lift_sweeps_not_checked",
            "robot_joint_reachability_not_checked",
            "hand_eye_transform_not_applied",
        ],
    }


def _motion_status_priority(status: str) -> int:
    return {
        "unconfigured": 5,
        "intersects": 4,
        "too_close": 3,
        "unknown": 2,
        "warning": 1,
        "clear": 0,
        "outside_front": 0,
        "disabled": 0,
    }.get(str(status), -1)


def _motion_stage_samples(
    stage: str,
    origin_start: np.ndarray,
    origin_end: np.ndarray,
    opening_start_mm: float,
    opening_end_mm: float,
    sample_count: int,
) -> List[Dict[str, Any]]:
    count = max(2, int(sample_count))
    rows: List[Dict[str, Any]] = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, count)):
        origin = (1.0 - float(fraction)) * origin_start + float(fraction) * origin_end
        opening = (1.0 - float(fraction)) * float(opening_start_mm) + float(fraction) * float(opening_end_mm)
        rows.append(
            {
                "stage": str(stage),
                "sample_index": int(index),
                "sample_count": int(count),
                "fraction": float(fraction),
                "origin_camera_mm": np.asarray(origin, dtype=np.float64),
                "opening_mm": float(opening),
            }
        )
    return rows


def check_full_gripper_pregrasp_motion(
    final_origin_camera_mm: Sequence[float],
    closing_axis_camera: Sequence[float],
    lateral_axis_camera: Sequence[float],
    approach_axis_camera: Sequence[float],
    target_closing_gap_mm: float,
    approach_opening_mm: float,
    pregrasp_offset_mm: float,
    open_start_offset_mm: float,
    rim_insert_depth_mm: float,
    box_model: Optional[BoxModel3D],
    neighbor_clouds: Sequence[Mapping[str, Any]],
    geometry_cfg: Mapping[str, Any],
    static_collision_cfg: Mapping[str, Any],
    motion_cfg: Mapping[str, Any],
    intrinsics: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Check the complete gripper from pregrasp until the wall is pinched.

    The checked motion is intentionally limited to the period before the target
    has been grasped:

    ``travel small opening -> pre-open -> open approach -> insert -> close``.

    Post-grasp lift, target transport, robot joint reachability and hand-eye
    conversion are outside M35.2 by user request.
    """

    if not bool(motion_cfg.get("enabled", False)):
        return {
            "enabled": False,
            "status": "disabled",
            "motion_scope": "pregrasp_to_grasp_only",
            "post_grasp_lift_checked": False,
        }

    try:
        x_axis = _unit(closing_axis_camera, "closing_axis_camera")
        z_axis = _unit(approach_axis_camera, "approach_axis_camera")
        x_axis = _unit(x_axis - z_axis * float(np.dot(x_axis, z_axis)), "closing_axis_orthogonal")
        y_axis = _unit(np.cross(z_axis, x_axis), "lateral_axis_right_handed")
        x_axis = _unit(np.cross(y_axis, z_axis), "closing_axis_right_handed")
        # Validate the supplied lateral hint without allowing it to flip the
        # right-handed grasp frame.
        y_hint = _unit(lateral_axis_camera, "lateral_axis_camera")
        if float(np.dot(y_axis, y_hint)) < 0.0:
            y_axis = -y_axis
            x_axis = -x_axis
        rotation = np.column_stack((x_axis, y_axis, z_axis))

        final_origin = np.asarray(final_origin_camera_mm, dtype=np.float64).reshape(3)
        pregrasp_offset = float(pregrasp_offset_mm)
        open_start_offset = float(open_start_offset_mm)
        insert_depth = float(rim_insert_depth_mm)
        if pregrasp_offset <= 0.0:
            raise ValueError("pregrasp_offset_mm must be positive")
        if open_start_offset < 0.0 or open_start_offset > pregrasp_offset:
            raise ValueError("open_start_offset_mm must be within [0, pregrasp_offset_mm]")
        if insert_depth < 0.0:
            raise ValueError("rim_insert_depth_mm must be non-negative")

        measured = geometry_cfg.get("finger_kinematics") or {}
        minimum_gap = _float(measured, "measured_minimum_gap_mm", 10.0)
        maximum_gap = _float(measured, "measured_maximum_gap_mm", 75.0)
        travel_opening = _float(motion_cfg, "travel_opening_mm", minimum_gap + 0.5)
        target_opening = float(target_closing_gap_mm)
        approach_opening = float(approach_opening_mm)
        for name, opening in (
            ("travel_opening_mm", travel_opening),
            ("approach_opening_mm", approach_opening),
            ("target_closing_gap_mm", target_opening),
        ):
            if opening < minimum_gap - 1e-6 or opening > maximum_gap + 1e-6:
                raise ValueError(
                    f"{name}={opening:.3f} outside measured opening range "
                    f"[{minimum_gap:.3f}, {maximum_gap:.3f}]"
                )

        check_cfg = dict(static_collision_cfg)
        check_cfg.update(dict(motion_cfg))
        check_cfg["enabled"] = True
        # Pose details are aggregated below; do not duplicate full component
        # payloads at every sample unless explicitly requested.
        include_pose_components = bool(motion_cfg.get("include_pose_component_details_in_json", False))
        if not include_pose_components:
            check_cfg["include_component_details_in_json"] = False
            check_cfg["include_instance_checks_in_json"] = False

        # Opening/closing occurs with the robot mounting interface fixed. The
        # symmetric pivot arc changes the fingertip midpoint slightly along Z,
        # so stage interpolation is performed in mounting-interface space and
        # converted back to the fingertip-origin convention used by the static
        # checker at every sample.
        target_model = build_static_gripper_model(target_opening, geometry_cfg, check_cfg)
        approach_model = build_static_gripper_model(approach_opening, geometry_cfg, check_cfg)
        travel_model = build_static_gripper_model(travel_opening, geometry_cfg, check_cfg)

        final_mount_origin = (
            final_origin
            + rotation @ np.asarray([0.0, 0.0, target_model.disk_upper_z_mm], dtype=np.float64)
        )
        final_open_tip_origin = (
            final_mount_origin
            - rotation @ np.asarray([0.0, 0.0, approach_model.disk_upper_z_mm], dtype=np.float64)
        )
        rim_open_tip_origin = final_open_tip_origin - z_axis * insert_depth
        rim_open_mount_origin = (
            rim_open_tip_origin
            + rotation @ np.asarray([0.0, 0.0, approach_model.disk_upper_z_mm], dtype=np.float64)
        )
        open_start_open_tip_origin = rim_open_tip_origin - z_axis * open_start_offset
        open_start_mount_origin = (
            open_start_open_tip_origin
            + rotation @ np.asarray([0.0, 0.0, approach_model.disk_upper_z_mm], dtype=np.float64)
        )
        pregrasp_travel_tip_origin = rim_open_tip_origin - z_axis * pregrasp_offset
        pregrasp_mount_origin = (
            pregrasp_travel_tip_origin
            + rotation @ np.asarray([0.0, 0.0, travel_model.disk_upper_z_mm], dtype=np.float64)
        )

        stage_definitions = [
            (
                "travel_small_opening",
                pregrasp_mount_origin,
                open_start_mount_origin,
                travel_opening,
                travel_opening,
                _int(motion_cfg, "travel_sample_count", 7),
            ),
            (
                "preopen_near_target",
                open_start_mount_origin,
                open_start_mount_origin,
                travel_opening,
                approach_opening,
                _int(motion_cfg, "open_sample_count", 6),
            ),
            (
                "approach_open",
                open_start_mount_origin,
                rim_open_mount_origin,
                approach_opening,
                approach_opening,
                _int(motion_cfg, "approach_sample_count", 7),
            ),
            (
                "insert_open",
                rim_open_mount_origin,
                final_mount_origin,
                approach_opening,
                approach_opening,
                _int(motion_cfg, "insert_sample_count", 5),
            ),
            (
                "close_on_rim",
                final_mount_origin,
                final_mount_origin,
                approach_opening,
                target_opening,
                _int(motion_cfg, "close_sample_count", 7),
            ),
        ]

        stop_on_hard = bool(motion_cfg.get("stop_on_first_hard_reject", True))
        include_pose_details = bool(motion_cfg.get("include_pose_details_in_json", False))
        pose_rows: List[Dict[str, Any]] = []
        stage_rows: List[Dict[str, Any]] = []
        all_box_rows: List[Dict[str, Any]] = []
        all_neighbor_rows: List[Dict[str, Any]] = []
        stopped_early = False

        for stage, origin_start, origin_end, opening_start, opening_end, count in stage_definitions:
            samples = _motion_stage_samples(
                stage,
                origin_start,
                origin_end,
                opening_start,
                opening_end,
                count,
            )
            evaluated: List[Dict[str, Any]] = []
            for sample in samples:
                current_model = build_static_gripper_model(
                    float(sample["opening_mm"]),
                    geometry_cfg,
                    check_cfg,
                )
                mount_origin = np.asarray(sample["origin_camera_mm"], dtype=np.float64)
                tip_origin = (
                    mount_origin
                    - rotation @ np.asarray([0.0, 0.0, current_model.disk_upper_z_mm], dtype=np.float64)
                )
                pose_result = check_full_gripper_static_final_pose(
                    tip_origin,
                    x_axis,
                    y_axis,
                    z_axis,
                    float(sample["opening_mm"]),
                    box_model,
                    neighbor_clouds,
                    geometry_cfg,
                    check_cfg,
                    intrinsics=intrinsics,
                )
                pose_row = {
                    "stage": stage,
                    "sample_index": int(sample["sample_index"]),
                    "sample_count": int(sample["sample_count"]),
                    "fraction": float(sample["fraction"]),
                    "origin_camera_mm": tip_origin.astype(float).tolist(),
                    "mounting_interface_origin_camera_mm": mount_origin.astype(float).tolist(),
                    "opening_mm": float(sample["opening_mm"]),
                    "status": pose_result.get("status"),
                    "box_status": pose_result.get("box_status"),
                    "neighbor_status": pose_result.get("neighbor_status"),
                    "box_minimum_clearance_mm": pose_result.get("box_minimum_clearance_mm"),
                    "box_minimum_safety_clearance_mm": pose_result.get("box_minimum_safety_clearance_mm"),
                    "neighbor_minimum_clearance_mm": pose_result.get("neighbor_minimum_clearance_mm"),
                    "box_worst_component": pose_result.get("box_worst_component"),
                    "box_nearest_wall": pose_result.get("box_nearest_wall"),
                    "neighbor_worst_component": pose_result.get("neighbor_worst_component"),
                    "neighbor_nearest_instance_id": pose_result.get("neighbor_nearest_instance_id"),
                    "box_colliding_components": pose_result.get("box_colliding_components") or [],
                    "neighbor_colliding_components": pose_result.get("neighbor_colliding_components") or [],
                    "neighbor_colliding_instance_ids": pose_result.get("neighbor_colliding_instance_ids") or [],
                    "hard_reject_box": bool(pose_result.get("hard_reject_box")),
                    "hard_reject_neighbor": bool(pose_result.get("hard_reject_neighbor")),
                }
                if include_pose_components:
                    pose_row["component_summaries"] = pose_result.get("component_summaries") or []
                    pose_row["components"] = pose_result.get("components") or []
                evaluated.append(pose_row)
                pose_rows.append(pose_row)
                all_box_rows.append(pose_row)
                all_neighbor_rows.append(pose_row)
                if stop_on_hard and (pose_row["hard_reject_box"] or pose_row["hard_reject_neighbor"]):
                    stopped_early = True
                    break

            box_statuses = [str(row.get("box_status") or "disabled") for row in evaluated]
            neighbor_statuses = [str(row.get("neighbor_status") or "disabled") for row in evaluated]
            stage_box_status = max(box_statuses, key=_motion_status_priority, default="disabled")
            stage_neighbor_status = max(neighbor_statuses, key=_motion_status_priority, default="disabled")
            stage_hard_box = any(bool(row.get("hard_reject_box")) for row in evaluated)
            stage_hard_neighbor = any(bool(row.get("hard_reject_neighbor")) for row in evaluated)
            if stage_hard_box or stage_hard_neighbor:
                stage_status = "rejected"
            elif stage_box_status not in {"clear", "outside_front", "disabled"} or stage_neighbor_status not in {"clear", "disabled"}:
                stage_status = "warning"
            else:
                stage_status = "clear"
            finite_box = [
                float(row["box_minimum_clearance_mm"])
                for row in evaluated
                if row.get("box_minimum_clearance_mm") is not None
            ]
            finite_box_safe = [
                float(row["box_minimum_safety_clearance_mm"])
                for row in evaluated
                if row.get("box_minimum_safety_clearance_mm") is not None
            ]
            finite_neighbor = [
                float(row["neighbor_minimum_clearance_mm"])
                for row in evaluated
                if row.get("neighbor_minimum_clearance_mm") is not None
            ]
            worst_pose = max(
                evaluated,
                key=lambda row: (
                    1 if row.get("hard_reject_box") or row.get("hard_reject_neighbor") else 0,
                    _motion_status_priority(str(row.get("box_status") or "disabled")),
                    _motion_status_priority(str(row.get("neighbor_status") or "disabled")),
                    -float(
                        row.get("box_minimum_safety_clearance_mm")
                        if row.get("box_minimum_safety_clearance_mm") is not None
                        else 1e12
                    ),
                    -float(
                        row.get("neighbor_minimum_clearance_mm")
                        if row.get("neighbor_minimum_clearance_mm") is not None
                        else 1e12
                    ),
                ),
                default=None,
            )
            stage_rows.append(
                {
                    "stage": stage,
                    "status": stage_status,
                    "box_status": stage_box_status,
                    "neighbor_status": stage_neighbor_status,
                    "planned_sample_count": int(max(2, count)),
                    "evaluated_sample_count": int(len(evaluated)),
                    "opening_start_mm": float(opening_start),
                    "opening_end_mm": float(opening_end),
                    "path_length_mm": float(np.linalg.norm(origin_end - origin_start)),
                    "box_minimum_clearance_mm": min(finite_box) if finite_box else None,
                    "box_minimum_safety_clearance_mm": min(finite_box_safe) if finite_box_safe else None,
                    "neighbor_minimum_clearance_mm": min(finite_neighbor) if finite_neighbor else None,
                    "worst_sample_index": worst_pose.get("sample_index") if worst_pose else None,
                    "worst_box_component": worst_pose.get("box_worst_component") if worst_pose else None,
                    "worst_box_wall": worst_pose.get("box_nearest_wall") if worst_pose else None,
                    "worst_neighbor_component": worst_pose.get("neighbor_worst_component") if worst_pose else None,
                    "worst_neighbor_instance_id": worst_pose.get("neighbor_nearest_instance_id") if worst_pose else None,
                    "hard_reject_box": bool(stage_hard_box),
                    "hard_reject_neighbor": bool(stage_hard_neighbor),
                }
            )
            if stopped_early:
                break

        box_status = max(
            (str(row.get("box_status") or "disabled") for row in pose_rows),
            key=_motion_status_priority,
            default="disabled",
        )
        neighbor_status = max(
            (str(row.get("neighbor_status") or "disabled") for row in pose_rows),
            key=_motion_status_priority,
            default="disabled",
        )
        hard_box = any(bool(row.get("hard_reject_box")) for row in pose_rows)
        hard_neighbor = any(bool(row.get("hard_reject_neighbor")) for row in pose_rows)
        if hard_box or hard_neighbor:
            status = "rejected"
        elif box_status not in {"clear", "outside_front", "disabled"} or neighbor_status not in {"clear", "disabled"}:
            status = "warning"
        else:
            status = "clear"

        finite_box = [
            float(row["box_minimum_clearance_mm"])
            for row in pose_rows
            if row.get("box_minimum_clearance_mm") is not None
        ]
        finite_box_safe = [
            float(row["box_minimum_safety_clearance_mm"])
            for row in pose_rows
            if row.get("box_minimum_safety_clearance_mm") is not None
        ]
        finite_neighbor = [
            float(row["neighbor_minimum_clearance_mm"])
            for row in pose_rows
            if row.get("neighbor_minimum_clearance_mm") is not None
        ]
        worst_pose = max(
            pose_rows,
            key=lambda row: (
                1 if row.get("hard_reject_box") or row.get("hard_reject_neighbor") else 0,
                _motion_status_priority(str(row.get("box_status") or "disabled")),
                _motion_status_priority(str(row.get("neighbor_status") or "disabled")),
                -float(
                    row.get("box_minimum_safety_clearance_mm")
                    if row.get("box_minimum_safety_clearance_mm") is not None
                    else 1e12
                ),
                -float(
                    row.get("neighbor_minimum_clearance_mm")
                    if row.get("neighbor_minimum_clearance_mm") is not None
                    else 1e12
                ),
            ),
            default=None,
        )

        result: Dict[str, Any] = {
            "enabled": True,
            "status": status,
            "motion_scope": "pregrasp_to_grasp_only",
            "pregrasp_path_checked": True,
            "post_grasp_lift_checked": False,
            "target_transport_checked": False,
            "orientation_mode": "fixed_grasp_orientation_during_pregrasp_motion",
            "travel_opening_mm": float(travel_opening),
            "approach_opening_mm": float(approach_opening),
            "target_closing_gap_mm": float(target_opening),
            "pregrasp_offset_mm": float(pregrasp_offset),
            "open_start_offset_mm": float(open_start_offset),
            "rim_insert_depth_mm": float(insert_depth),
            "box_status": box_status,
            "neighbor_status": neighbor_status,
            "box_minimum_clearance_mm": min(finite_box) if finite_box else None,
            "box_minimum_safety_clearance_mm": min(finite_box_safe) if finite_box_safe else None,
            "neighbor_minimum_clearance_mm": min(finite_neighbor) if finite_neighbor else None,
            "worst_stage": worst_pose.get("stage") if worst_pose else None,
            "worst_sample_index": worst_pose.get("sample_index") if worst_pose else None,
            "box_worst_component": worst_pose.get("box_worst_component") if worst_pose else None,
            "box_nearest_wall": worst_pose.get("box_nearest_wall") if worst_pose else None,
            "neighbor_worst_component": worst_pose.get("neighbor_worst_component") if worst_pose else None,
            "neighbor_nearest_instance_id": worst_pose.get("neighbor_nearest_instance_id") if worst_pose else None,
            "box_colliding_components": sorted(
                {
                    str(component)
                    for row in pose_rows
                    for component in (row.get("box_colliding_components") or [])
                }
            ),
            "neighbor_colliding_components": sorted(
                {
                    str(component)
                    for row in pose_rows
                    for component in (row.get("neighbor_colliding_components") or [])
                }
            ),
            "neighbor_colliding_instance_ids": sorted(
                {
                    int(instance_id)
                    for row in pose_rows
                    for instance_id in (row.get("neighbor_colliding_instance_ids") or [])
                }
            ),
            "hard_reject": bool(hard_box or hard_neighbor),
            "hard_reject_box": bool(hard_box),
            "hard_reject_neighbor": bool(hard_neighbor),
            "stopped_early": bool(stopped_early),
            "evaluated_pose_count": int(len(pose_rows)),
            "planned_pose_count": int(sum(max(2, int(row[5])) for row in stage_definitions)),
            "stage_summaries": stage_rows,
            "path_keyframes_camera": {
                "fingertip_midpoint": {
                    "pregrasp_travel_opening": pregrasp_travel_tip_origin.astype(float).tolist(),
                    "open_start_approach_opening": open_start_open_tip_origin.astype(float).tolist(),
                    "rim_approach_opening": rim_open_tip_origin.astype(float).tolist(),
                    "final_open_before_close": final_open_tip_origin.astype(float).tolist(),
                    "final_grasp_closed": final_origin.astype(float).tolist(),
                },
                "mounting_interface": {
                    "pregrasp": pregrasp_mount_origin.astype(float).tolist(),
                    "open_start": open_start_mount_origin.astype(float).tolist(),
                    "rim": rim_open_mount_origin.astype(float).tolist(),
                    "final_grasp": final_mount_origin.astype(float).tolist(),
                },
            },
            "limitations": [
                "post_grasp_lift_and_target_transport_intentionally_not_checked",
                "motion_before_pregrasp_keyframe_not_checked",
                "grasp_orientation_is_fixed_during_checked_motion",
                "robot_joint_reachability_not_checked",
                "hand_eye_transform_not_applied",
            ],
        }
        if include_pose_details:
            result["pose_checks"] = pose_rows
        result["_debug"] = {
            "pose_checks": pose_rows,
            "mounting_path_camera_mm": [
                row.get("mounting_interface_origin_camera_mm")
                for row in pose_rows
                if row.get("mounting_interface_origin_camera_mm") is not None
            ],
            "fingertip_midpoint_path_camera_mm": [row.get("origin_camera_mm") for row in pose_rows],
        }
        return result
    except Exception as error:
        return {
            "enabled": True,
            "status": "unconfigured",
            "reason": str(error),
            "motion_scope": "pregrasp_to_grasp_only",
            "post_grasp_lift_checked": False,
            "hard_reject_on_unconfigured": bool(motion_cfg.get("hard_reject_on_unconfigured", True)),
        }
