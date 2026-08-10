"""Rim-pinch geometry tests through M35.2 complete pre-grasp motion checks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    analyze_scene,
    associate_ring_mouths,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.offline_validate import main
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _instance(instance_id: int, class_id: int, name: str, mask: np.ndarray) -> SegmentationInstance:
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=class_id,
        class_name=name,
        confidence=0.95,
        mask=mask.astype(bool),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _config() -> GeometryConfig:
    return GeometryConfig(
        {
            "classes": {"foam_ring": "foam_ring", "ring_mouth": "ring_mouth"},
            "association": {
                "use_filled_outer_envelope": True,
                "envelope_close_px": 2,
                "envelope_dilate_px": 2,
                "minimum_component_area_ratio": 0.03,
                "minimum_envelope_containment": 0.55,
                "minimum_center_inside": True,
                "maximum_normalized_center_distance": 0.60,
                "minimum_mouth_to_ring_area_ratio": 0.02,
                "maximum_mouth_to_ring_area_ratio": 0.70,
            },
            "depth": {
                "minimum_mm": 150,
                "maximum_mm": 3000,
                "mask_erode_px": 1,
                "mouth_exclusion_px": 1,
                "front_band_expand_ratio": 0.45,
                "minimum_front_band_px": 5,
                "maximum_front_band_px": 28,
                "minimum_valid_points": 40,
                "local_obstacle_margin_mm": 8.0,
                "maximum_front_obstacle_ratio": 0.20,
                "local_obstacle_observable_max_tilt_deg": 30.0,
            },
            "plane": {
                "ransac_iterations": 150,
                "inlier_threshold_mm": 2.0,
                "minimum_inlier_ratio": 0.45,
                "random_seed": 7,
                "refine_with_svd": True,
            },
            "pose": {
                "normal_mode": "depth_plane",
                "normal_disagreement_warning_deg": 25.0,
            },
            "object_geometry": {
                "minimum_inner_diameter_mm": 30.0,
                "maximum_inner_diameter_mm": 120.0,
                "minimum_wall_thickness_mm": 8.0,
                "maximum_wall_thickness_mm": 55.0,
                "maximum_outer_search_radius_ratio": 1.6,
                "maximum_ring_mask_gap_px": 1,
                "physical_size_hard_reject": False,
            },
            "gripper": {
                "mode": "rim_pinch",
                "finger_thickness_mm": 17.0,
                "finger_width_mm": 20.0,
                "opening_reference": "inner_gap",
                "minimum_opening_mm": 10.0,
                "maximum_opening_mm": 80.0,
                "opening_safety_margin_mm": 3.0,
                "wall_compression_mm": 1.5,
                "rim_insert_depth_mm": 20.0,
                "pregrasp_offset_mm": 90.0,
                "preopen_clearance_mm": 6.0,
                "clock_position_count": 12,
                "geometry_candidate_max_tilt_deg": 45.0,
                "robot_safe_max_tilt_deg": 30.0,
                "top_layer_tolerance_mm": 15.0,
            },
            "box_wall": {
                "enabled": False,
            },
            "candidate": {
                "minimum_inner_finger_mouth_containment": 0.55,
                "maximum_other_ring_overlap_ratio": 0.12,
                "minimum_image_border_margin_px": 2,
                "minimum_neighbor_clearance_mm": 0.0,
                "hard_reject_front_obstacle": False,
            },
            "quality": {
                "minimum_depth_valid_ratio": 0.25,
                "minimum_ellipse_points": 12,
                "minimum_mouth_area_px": 80,
                "minimum_ring_area_px": 300,
            },
            "robot_interface": {"camera_frame_id": "camera_color_optical_frame"},
        }
    )


def _synthetic_scene(with_neighbor: bool = False):
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 860
    instances = [_instance(0, 0, "foam_ring", ring), _instance(1, 1, "ring_mouth", mouth)]
    if with_neighbor:
        neighbor = np.zeros_like(ring)
        cv2.circle(neighbor, (245, 130), 28, 1, -1)
        depth[neighbor.astype(bool)] = 790
        instances.append(_instance(2, 0, "foam_ring", neighbor))
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    return instances, depth, intrinsics


def test_donut_mask_association_uses_filled_envelope() -> None:
    height, width = 200, 240
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (120, 100), 42, 1, -1)
    cv2.circle(ring, (120, 100), 22, 0, -1)
    cv2.circle(mouth, (120, 100), 20, 1, -1)
    ring_instance = _instance(0, 0, "foam_ring", ring)
    mouth_instance = _instance(1, 1, "ring_mouth", mouth)
    matches, unmatched = associate_ring_mouths([ring_instance], [mouth_instance], _config())
    assert len(matches) == 1
    assert not unmatched
    assert matches[0][2]["raw_mask_containment"] == 0.0
    assert matches[0][2]["envelope_containment"] > 0.95


def test_twelve_rim_pinch_candidates_and_robot_frame_are_generated() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    scene = analyze_scene(instances, depth, intrinsics, _config())
    assert scene["matched_pairs"] == 1
    assert scene["eligible_count"] == 1
    item = scene["instances"][0]
    candidates = item["grasp"]["clock_candidates"]
    assert len(candidates) == 12
    assert [candidate["clock_hour"] for candidate in candidates] == [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    best = item["grasp"]["best_clock_candidate"]
    assert best is not None
    assert best["valid"] is True
    assert best["wall_thickness_mm"] == pytest.approx(44.0, abs=4.0)
    assert best["target_closing_gap_mm"] == pytest.approx(best["wall_thickness_mm"] - 3.0, abs=0.1)
    frame = best["grasp_frame_camera"]
    rotation = np.asarray(frame["rotation_matrix_rows"], dtype=np.float64)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert frame["inner_finger_side"] == "negative_x"
    assert frame["outer_finger_side"] == "positive_x"
    assert scene["robot_candidate"]["robot_ready"] is False


def test_neighbor_on_three_oclock_reduces_or_rejects_that_candidate() -> None:
    instances, depth, intrinsics = _synthetic_scene(with_neighbor=True)
    scene = analyze_scene(instances, depth, intrinsics, _config())
    item = next(row for row in scene["instances"] if row["ring_instance_id"] == 0)
    candidates = {candidate["clock_hour"]: candidate for candidate in item["grasp"]["clock_candidates"]}
    three = candidates[3]
    nine = candidates[9]
    assert (not three["valid"]) or three["score"] < nine["score"]




def test_box_wall_rejects_outer_finger_toward_wall() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    raw = _config().raw
    raw["box_wall"] = {
        "enabled": True,
        "model": "normalized_inner_polygon",
        # Put a conservative left inner wall close to the ring. The 9-o'clock
        # outer finger should cross it while 3-o'clock remains available.
        "inner_polygon_normalized": [
            [0.43, 0.08],
            [0.96, 0.08],
            [0.96, 0.92],
            [0.43, 0.92],
        ],
        "hard_reject": True,
        "minimum_outer_finger_containment": 0.98,
        "minimum_inner_finger_containment": 0.90,
        "minimum_wall_clearance_mm": 0.0,
        "clearance_percentile": 10.0,
    }
    scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw))
    item = scene["instances"][0]
    candidates = {candidate["clock_hour"]: candidate for candidate in item["grasp"]["clock_candidates"]}
    nine = candidates[9]
    three = candidates[3]
    assert nine["box_wall_status"] == "intersects"
    assert "finger_sweep_intersects_box_wall" in nine["rejection_reasons"]
    assert nine["valid"] is False
    assert three["box_wall_status"] in {"clear", "too_close"}
    assert scene["box_wall_model"]["enabled"] is True


def _circle_polygon(cx: float, cy: float, radius: float, width: int, height: int, count: int = 64) -> str:
    values = []
    for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        values.extend([(cx + radius * np.cos(angle)) / width, (cy + radius * np.sin(angle)) / height])
    return " ".join("%.8f" % value for value in values)


def test_cli_writes_geometry_and_robot_candidate(tmp_path: Path) -> None:
    width, height = 360, 260
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "depth").mkdir()
    (data / "meta").mkdir()
    labels = tmp_path / "labels"
    labels.mkdir()
    capture_id = "visionops_test_rim_pinch"
    rgb = np.full((height, width, 3), 160, dtype=np.uint8)
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 860
    cv2.imwrite(str(data / "images" / (capture_id + ".jpg")), rgb)
    cv2.imwrite(str(data / "depth" / (capture_id + ".png")), depth)
    meta = {
        "depth": {
            "aligned_to_color": True,
            "calibration_ready": True,
            "intrinsics_saved": {
                "fx": 400.0,
                "fy": 400.0,
                "cx": 180.0,
                "cy": 130.0,
                "width": width,
                "height": height,
            },
        }
    }
    (data / "meta" / (capture_id + ".json")).write_text(json.dumps(meta), encoding="utf-8")
    (labels / (capture_id + ".txt")).write_text(
        "0 " + _circle_polygon(180, 130, 42, width, height) + "\n"
        "1 " + _circle_polygon(180, 130, 20, width, height) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "line.yaml"
    import yaml  # type: ignore

    config_path.write_text(yaml.safe_dump(_config().raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    output = tmp_path / "out"
    rc = main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(data),
            "--capture-id",
            capture_id,
            "--labels-dir",
            str(labels),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    frame_dir = output / capture_id
    assert (frame_dir / "geometry_result.json").exists()
    assert (frame_dir / "robot_grasp_candidate.json").exists()
    assert (frame_dir / "geometry_overlay.jpg").exists()
    assert (frame_dir / "paired_axis_overlay.jpg").exists()
    assert (frame_dir / "paired_axis_projection.json").exists()
    axis_payload = json.loads((frame_dir / "paired_axis_projection.json").read_text(encoding="utf-8"))
    assert axis_payload["stage"] == "M35.4"
    assert axis_payload["visualization_mode"] == "directed_fixed_length_3d_axis_rod_projection"
    assert axis_payload["rod_length_mm"] == pytest.approx(80.0)
    assert axis_payload["axes"][0]["near_depth_mm"] < axis_payload["axes"][0]["far_depth_mm"]
    payload = json.loads((frame_dir / "robot_grasp_candidate.json").read_text(encoding="utf-8"))
    assert payload["message_type"] == "foam_ring_rim_pinch_grasp_candidate"
    assert payload["robot_ready"] is False
    assert payload["grasp_frame_camera"]["x_closing_axis_camera"]
    assert payload["grasp_frame_camera"]["z_approach_axis_camera"]

    disabled_raw = _config().raw
    disabled_raw["axis_direction"] = {"enabled": False, "rod_length_mm": 80.0}
    config_path.write_text(yaml.safe_dump(disabled_raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    disabled_output = tmp_path / "out_axis_disabled"
    disabled_rc = main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(data),
            "--capture-id",
            capture_id,
            "--labels-dir",
            str(labels),
            "--output",
            str(disabled_output),
        ]
    )
    assert disabled_rc == 0
    disabled_frame_dir = disabled_output / capture_id
    assert not (disabled_frame_dir / "paired_axis_overlay.jpg").exists()
    assert not (disabled_frame_dir / "paired_axis_projection.json").exists()
    disabled_geometry = json.loads((disabled_frame_dir / "geometry_result.json").read_text(encoding="utf-8"))
    assert disabled_geometry["axis_direction_diagnostics"]["enabled"] is False
    # Core grasp geometry remains available because it still needs the pose normal.
    assert (disabled_frame_dir / "robot_grasp_candidate.json").exists()


def test_soft_wall_closing_gap_can_clamp_near_mechanical_minimum() -> None:
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 27, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 840
    instances = [_instance(0, 0, "foam_ring", ring), _instance(1, 1, "ring_mouth", mouth)]
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    raw = _config().raw
    raw["gripper"].update(
        {
            "closing_limit_margin_mm": 0.5,
            "approach_limit_margin_mm": 3.0,
            "minimum_contact_compression_mm": 0.5,
            "maximum_contact_compression_mm": 3.0,
        }
    )
    scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw))
    item = scene["instances"][0]
    valid = [candidate for candidate in item["grasp"]["clock_candidates"] if candidate["valid"]]
    assert valid
    best = item["grasp"]["best_clock_candidate"]
    assert best["target_closing_gap_mm"] >= 10.5
    assert 0.5 <= best["actual_wall_compression_each_side_mm"] <= 3.0
    assert "target_closing_gap_out_of_range" not in best["rejection_reasons"]


def test_calibrated_3d_box_swept_prism_clear_and_collision() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import (
        BoxModel3D,
        check_swept_prism_against_box,
    )

    model = BoxModel3D(
        origin_camera_mm=np.asarray([-100.0, -100.0, 500.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([200.0, 200.0, 200.0]),
        safety_margin_mm={"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0, "back": 5.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )
    clear = check_swept_prism_against_box(
        model,
        [0.0, 0.0, 450.0],
        [0.0, 0.0, 600.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        10.0,
        10.0,
        stage="approach",
    )
    assert clear["status"] == "clear"
    collision = check_swept_prism_against_box(
        model,
        [-98.0, 0.0, 520.0],
        [-98.0, 0.0, 600.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        10.0,
        10.0,
        stage="insert",
    )
    assert collision["status"] == "intersects"
    assert collision["nearest_wall"] == "left"


def test_calibrated_3d_box_model_roundtrip_and_projection() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import (
        box_model_from_dict,
        box_projection,
    )

    payload = {
        "model_type": "calibrated_3d_cuboid",
        "coordinate_frame": "camera_color_optical_frame",
        "origin_camera_mm": [-100.0, -80.0, 500.0],
        "rotation_camera_from_box_rows": np.eye(3).tolist(),
        "inner_size_mm": {"width": 200.0, "height": 160.0, "depth": 300.0},
        "safety_margin_mm": {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0, "back": 5.0},
        "camera_resolution": {"width": 360, "height": 260},
    }
    model = box_model_from_dict(payload)
    points_box = np.asarray([[0.0, 0.0, 0.0], [200.0, 160.0, 300.0]])
    points_camera = model.box_to_camera(points_box)
    assert model.camera_to_box(points_camera) == pytest.approx(points_box, abs=1e-8)
    projection = box_projection(model, {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0})
    assert len(projection["front_polygon_uv"]) == 4
    assert len(projection["rear_polygon_uv"]) == 4
    assert len(projection["edge_lines_uv"]) == 4


def _neighbor_3d_config(raw: dict) -> GeometryConfig:
    raw["neighbor_3d"] = {
        "enabled": True,
        "minimum_depth_mm": 150,
        "maximum_depth_mm": 3000,
        "mask_erode_px": 0,
        "point_stride": 1,
        "minimum_points_per_instance": 4,
        "minimum_total_points": 4,
        "maximum_points_per_instance": 5000,
        "target_surface_exclusion_enabled": True,
        "target_surface_exclusion_dilate_px": 2,
        "target_surface_exclusion_mm": 12.0,
        "minimum_collision_points": 4,
        "intersection_tolerance_mm": 1.5,
        "minimum_clearance_mm": 3.0,
        "hard_reject_on_intersection": True,
        "hard_reject_on_clearance": True,
        "hard_reject_on_unknown": False,
        "score_saturation_mm": 30.0,
        "unknown_score": 0.35,
    }
    raw["candidate"]["neighbor_2d_overlap_mode"] = "warning_only"
    raw["candidate"]["neighbor_2d_clearance_mode"] = "warning_only"
    return GeometryConfig(raw)


def _overlapping_neighbor_scene(neighbor_depth_mm: int):
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    neighbor = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    cv2.circle(neighbor, (225, 130), 28, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 860
    depth[neighbor.astype(bool)] = int(neighbor_depth_mm)
    instances = [
        _instance(0, 0, "foam_ring", ring),
        _instance(1, 1, "ring_mouth", mouth),
        _instance(2, 0, "foam_ring", neighbor),
    ]
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    return instances, depth, intrinsics


def test_neighbor_3d_rejects_same_depth_collision_despite_2d_warning_only() -> None:
    instances, depth, intrinsics = _overlapping_neighbor_scene(805)
    scene = analyze_scene(instances, depth, intrinsics, _neighbor_3d_config(_config().raw))
    item = next(row for row in scene["instances"] if row["ring_instance_id"] == 0)
    candidates = {candidate["clock_hour"]: candidate for candidate in item["grasp"]["clock_candidates"]}
    three = candidates[3]
    assert three["other_ring_overlap_ratio"] > 0.9
    assert "neighbor_2d_overlap_warning" in three["warnings"]
    assert three["neighbor_3d_status"] == "intersects"
    assert "neighbor_3d_finger_collision" in three["rejection_reasons"]
    assert three["valid"] is False
    assert 2 in three["neighbor_3d"]["colliding_instance_ids"]


def test_neighbor_3d_keeps_projected_overlap_when_neighbor_is_behind_sweep() -> None:
    instances, depth, intrinsics = _overlapping_neighbor_scene(950)
    scene = analyze_scene(instances, depth, intrinsics, _neighbor_3d_config(_config().raw))
    item = next(row for row in scene["instances"] if row["ring_instance_id"] == 0)
    candidates = {candidate["clock_hour"]: candidate for candidate in item["grasp"]["clock_candidates"]}
    three = candidates[3]
    assert three["other_ring_overlap_ratio"] > 0.9
    assert "neighbor_2d_overlap_warning" in three["warnings"]
    assert three["neighbor_3d_status"] == "clear"
    assert three["neighbor_3d_clearance_mm"] > 100.0
    assert "neighbor_3d_finger_collision" not in three["rejection_reasons"]
    assert three["valid"] is True


def test_target_surface_exclusion_removes_duplicate_visible_surface_points() -> None:
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 860
    instances = [
        _instance(0, 0, "foam_ring", ring),
        _instance(1, 1, "ring_mouth", mouth),
        # Simulate an overlapping duplicate/bleeding PT mask.
        _instance(2, 0, "foam_ring", ring.copy()),
    ]
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    scene = analyze_scene(instances, depth, intrinsics, _neighbor_3d_config(_config().raw))
    item = next(row for row in scene["instances"] if row["ring_instance_id"] == 0)
    summary = item["neighbor_3d_point_clouds"]
    neighbor = summary["instances"][0]
    assert neighbor["removed_target_surface_point_count"] > 1000
    best = item["grasp"]["best_clock_candidate"]
    assert best is not None
    assert best["neighbor_3d_status"] == "clear"
    assert "neighbor_3d_finger_collision" not in best["rejection_reasons"]


def _m35_geometry_config() -> dict:
    return {
        "contact_block": {
            "width_x_mm": 15.0,
            "thickness_y_mm": 20.0,
            "length_z_mm": 35.0,
            "intended_ring_engagement_mm": 20.0,
        },
        "moving_finger": {
            "pivot_to_tip_mm": 90.0,
            "width_x_mm": 16.0,
            "thickness_y_mm": 20.0,
        },
        "finger_kinematics": {
            "pivot_center_separation_mm": 36.0,
            "pivot_center_from_disk_lower_mm": 55.0,
        },
        "palm": {"length_z_mm": 60.0, "width_x_mm": 50.0, "thickness_y_mm": 35.0},
        "mounting_disk": {"diameter_mm": 70.0, "length_z_mm": 20.0},
        "pneumatic_fitting": {
            "enabled": True,
            "axis": "positive_x",
            "center_from_disk_lower_mm": 25.0,
            "protrusion_length_mm": 30.0,
            "diameter_mm": 10.0,
            "count": 2,
            "center_spacing_y_mm": 20.0,
        },
        "robot_wrist": {"enabled": True, "diameter_mm": 100.0, "length_z_mm": 200.0},
    }


def _m35_collision_config() -> dict:
    return {
        "enabled": True,
        "obb_surface_resolution": 3,
        "cylinder_radial_samples": 16,
        "cylinder_axial_samples": 4,
        "front_entry_tolerance_mm": 2.0,
        "component_extra_margin_mm": {
            "contact_block": 0.0,
            "moving_finger": 0.0,
            "palm": 0.0,
            "mounting_disk": 0.0,
            "pneumatic_fitting": 0.0,
            "robot_wrist": 0.0,
        },
        "minimum_points_per_instance": 4,
        "minimum_collision_points": 4,
        "intersection_tolerance_mm": 1.5,
        "minimum_clearance_mm": 4.0,
        "hard_reject_box_intersection": True,
        "hard_reject_box_clearance": True,
        "hard_reject_neighbor_intersection": True,
        "hard_reject_neighbor_clearance": True,
        "hard_reject_neighbor_unknown": False,
        "hard_reject_on_unconfigured": True,
    }


def test_m35_1_measured_gripper_geometry_reconstructs_dimensions() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        build_static_gripper_model,
    )

    model_min = build_static_gripper_model(10.0, _m35_geometry_config(), _m35_collision_config())
    model_max = build_static_gripper_model(75.0, _m35_geometry_config(), _m35_collision_config())
    assert model_min.diagnostics["disk_lower_to_tip_mm"] == pytest.approx(144.83, abs=0.5)
    assert model_min.diagnostics["disk_upper_to_tip_mm"] == pytest.approx(164.83, abs=0.5)
    assert model_min.finger_angle_deg == pytest.approx(-3.50, abs=0.5)
    assert model_max.finger_angle_deg == pytest.approx(17.47, abs=0.7)
    names = {component.name for component in model_min.components}
    assert names == {
        "inner_moving_finger",
        "outer_moving_finger",
        "inner_contact_block",
        "outer_contact_block",
        "palm",
        "mounting_disk",
        "pneumatic_fitting_1",
        "pneumatic_fitting_2",
        "robot_wrist",
    }
    fittings = sorted(
        [component for component in model_min.components if component.group == "pneumatic_fitting"],
        key=lambda component: float(component.center_local_mm[1]),
    )
    assert len(fittings) == 2
    # Side-mounted fittings: along +X (right side of the front view),
    # horizontally separated in the side view, so they share the same Z but
    # differ along the palm-thickness axis Y.
    palm_half_x = 0.5 * _m35_geometry_config()["palm"]["width_x_mm"]
    fitting_length = _m35_geometry_config()["pneumatic_fitting"]["protrusion_length_mm"]
    expected_center_x = palm_half_x + 0.5 * fitting_length
    assert fittings[0].center_local_mm[0] == pytest.approx(expected_center_x, abs=1e-6)
    assert fittings[1].center_local_mm[0] == pytest.approx(expected_center_x, abs=1e-6)
    assert fittings[0].center_local_mm[1] == pytest.approx(-10.0, abs=1e-6)
    assert fittings[1].center_local_mm[1] == pytest.approx(10.0, abs=1e-6)
    assert fittings[0].center_local_mm[2] == pytest.approx(fittings[1].center_local_mm[2], abs=1e-6)
    # The thinner fingers should remain centered within the thicker palm when
    # viewed from the side.
    palm = next(component for component in model_min.components if component.name == "palm")
    for name in ("inner_moving_finger", "outer_moving_finger", "inner_contact_block", "outer_contact_block"):
        component = next(item for item in model_min.components if item.name == name)
        assert component.center_local_mm[1] == pytest.approx(palm.center_local_mm[1], abs=1e-6)


def test_m35_1_complete_gripper_static_box_clear_and_collision() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import BoxModel3D
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        check_full_gripper_static_final_pose,
    )

    large_box = BoxModel3D(
        origin_camera_mm=np.asarray([-220.0, -220.0, 350.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([440.0, 440.0, 450.0]),
        safety_margin_mm={"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "back": 0.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )
    clear = check_full_gripper_static_final_pose(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        large_box,
        [],
        _m35_geometry_config(),
        _m35_collision_config(),
    )
    assert clear["status"] == "clear"
    assert clear["box_status"] == "clear"
    assert clear["component_count"] == 9
    mount = clear["mounting_interface_frame_camera"]
    assert mount["mount_to_fingertip_midpoint_mm"] == pytest.approx(164.8, abs=0.6)
    assert mount["origin_camera_mm"][2] < 650.0

    narrow_box = BoxModel3D(
        origin_camera_mm=np.asarray([-40.0, -220.0, 350.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([80.0, 440.0, 450.0]),
        safety_margin_mm={"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "back": 0.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )
    collision = check_full_gripper_static_final_pose(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        narrow_box,
        [],
        _m35_geometry_config(),
        _m35_collision_config(),
    )
    assert collision["status"] == "rejected"
    assert collision["box_status"] in {"intersects", "too_close"}
    assert collision["hard_reject_box"] is True
    assert collision["box_worst_component"] in {"mounting_disk", "robot_wrist", "pneumatic_fitting", "pneumatic_fitting_1", "pneumatic_fitting_2", "palm"}


def test_m35_1_complete_gripper_static_neighbor_hits_palm() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import BoxModel3D
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        build_static_gripper_model,
        check_full_gripper_static_final_pose,
    )

    model = build_static_gripper_model(20.0, _m35_geometry_config(), _m35_collision_config())
    palm = next(component for component in model.components if component.name == "palm")
    origin = np.asarray([0.0, 0.0, 650.0])
    palm_center = palm.center_local_mm + origin
    offsets = np.asarray(
        [[x, y, z] for x in (-2.0, 0.0, 2.0) for y in (-2.0, 0.0, 2.0) for z in (-2.0, 0.0, 2.0)],
        dtype=np.float64,
    )
    cloud = {
        "instance_id": 99,
        "retained_point_count": len(offsets),
        "points_camera": palm_center.reshape(1, 3) + offsets,
    }
    box = BoxModel3D(
        origin_camera_mm=np.asarray([-220.0, -220.0, 350.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([440.0, 440.0, 450.0]),
        safety_margin_mm={"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "back": 0.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )
    result = check_full_gripper_static_final_pose(
        origin,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        box,
        [cloud],
        _m35_geometry_config(),
        _m35_collision_config(),
    )
    assert result["status"] == "rejected"
    assert result["neighbor_status"] == "intersects"
    assert result["neighbor_worst_component"] == "palm"
    assert 99 in result["neighbor_colliding_instance_ids"]


def _m352_motion_config() -> dict:
    return {
        "enabled": True,
        "motion_scope": "pregrasp_to_grasp_only",
        "travel_opening_mm": 10.5,
        "open_start_offset_mm": 25.0,
        "travel_sample_count": 4,
        "open_sample_count": 5,
        "approach_sample_count": 4,
        "insert_sample_count": 3,
        "close_sample_count": 5,
        "obb_surface_resolution": 3,
        "cylinder_radial_samples": 16,
        "cylinder_axial_samples": 4,
        "minimum_points_per_instance": 4,
        "minimum_collision_points": 4,
        "intersection_tolerance_mm": 1.5,
        "minimum_clearance_mm": 4.0,
        "hard_reject_box_intersection": True,
        "hard_reject_box_clearance": True,
        "hard_reject_neighbor_intersection": True,
        "hard_reject_neighbor_clearance": True,
        "hard_reject_neighbor_unknown": False,
        "hard_reject_on_unconfigured": True,
        "stop_on_first_hard_reject": True,
        "include_pose_details_in_json": False,
    }


def _large_m352_box():
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import BoxModel3D

    return BoxModel3D(
        origin_camera_mm=np.asarray([-250.0, -250.0, 300.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([500.0, 500.0, 600.0]),
        safety_margin_mm={"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "back": 0.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )


def test_m35_2_pregrasp_motion_checks_only_until_grasp() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        check_full_gripper_pregrasp_motion,
    )

    result = check_full_gripper_pregrasp_motion(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        55.0,
        90.0,
        25.0,
        20.0,
        _large_m352_box(),
        [],
        _m35_geometry_config(),
        _m35_collision_config(),
        _m352_motion_config(),
    )
    assert result["status"] == "clear"
    assert result["pregrasp_path_checked"] is True
    assert result["post_grasp_lift_checked"] is False
    assert [row["stage"] for row in result["stage_summaries"]] == [
        "travel_small_opening",
        "preopen_near_target",
        "approach_open",
        "insert_open",
        "close_on_rim",
    ]
    assert "lift" not in " ".join(row["stage"] for row in result["stage_summaries"])
    fingertips = result["path_keyframes_camera"]["fingertip_midpoint"]
    mounts = result["path_keyframes_camera"]["mounting_interface"]
    assert fingertips["final_grasp_closed"] == pytest.approx([0.0, 0.0, 650.0])
    assert fingertips["rim_approach_opening"][2] == pytest.approx(
        fingertips["final_open_before_close"][2] - 20.0, abs=1e-6
    )
    preopen = next(row for row in result["stage_summaries"] if row["stage"] == "preopen_near_target")
    assert preopen["path_length_mm"] == pytest.approx(0.0, abs=1e-9)


def test_m35_2_neighbor_can_block_preopen_while_final_static_pose_is_clear() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        build_static_gripper_model,
        check_full_gripper_pregrasp_motion,
        check_full_gripper_static_final_pose,
    )

    geometry = _m35_geometry_config()
    static_cfg = _m35_collision_config()
    approach_opening = 55.0
    model = build_static_gripper_model(approach_opening, geometry, static_cfg)
    outer_contact = next(component for component in model.components if component.name == "outer_contact_block")
    # Final fingertip midpoint is z=650; rim is z=630 and pre-open starts at
    # z=605. Put an obstacle on the outer contact block at the pre-open pose.
    preopen_origin = np.asarray([0.0, 0.0, 560.0])
    obstacle_center = preopen_origin + outer_contact.center_local_mm
    offsets = np.asarray(
        [[x, y, z] for x in (-2.0, 0.0, 2.0) for y in (-2.0, 0.0, 2.0) for z in (-2.0, 0.0, 2.0)],
        dtype=np.float64,
    )
    cloud = {
        "instance_id": 77,
        "retained_point_count": len(offsets),
        "points_camera": obstacle_center.reshape(1, 3) + offsets,
    }
    final_static = check_full_gripper_static_final_pose(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        _large_m352_box(),
        [cloud],
        geometry,
        static_cfg,
    )
    assert final_static["neighbor_status"] == "clear"

    motion = check_full_gripper_pregrasp_motion(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        approach_opening,
        90.0,
        70.0,
        20.0,
        _large_m352_box(),
        [cloud],
        geometry,
        static_cfg,
        _m352_motion_config(),
    )
    assert motion["status"] == "rejected"
    assert motion["neighbor_status"] in {"intersects", "too_close"}
    assert motion["hard_reject_neighbor"] is True
    assert motion["worst_stage"] in {"preopen_near_target", "approach_open"}
    assert motion["neighbor_nearest_instance_id"] == 77


def test_m35_2_opening_sweep_can_hit_box_when_final_closed_pose_fits() -> None:
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.box_model_3d import BoxModel3D
    from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.gripper_model_3d import (
        check_full_gripper_pregrasp_motion,
        check_full_gripper_static_final_pose,
    )

    geometry = _m35_geometry_config()
    geometry["mounting_disk"] = {**geometry["mounting_disk"], "diameter_mm": 60.0}
    geometry["pneumatic_fitting"] = {**geometry["pneumatic_fitting"], "enabled": False}
    geometry["robot_wrist"] = {**geometry["robot_wrist"], "enabled": False}
    static_cfg = _m35_collision_config()
    box = BoxModel3D(
        origin_camera_mm=np.asarray([-37.0, -200.0, 300.0]),
        rotation_camera_from_box=np.eye(3),
        inner_size_mm=np.asarray([74.0, 400.0, 600.0]),
        safety_margin_mm={"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "back": 0.0},
        camera_frame_id="camera_color_optical_frame",
        camera_resolution=(360, 260),
        intrinsics=None,
        calibration={},
    )
    final_static = check_full_gripper_static_final_pose(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        10.5,
        box,
        [],
        geometry,
        static_cfg,
    )
    assert final_static["box_status"] == "clear"

    motion = check_full_gripper_pregrasp_motion(
        [0.0, 0.0, 650.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        10.5,
        60.0,
        90.0,
        25.0,
        20.0,
        box,
        [],
        geometry,
        static_cfg,
        _m352_motion_config(),
    )
    assert motion["status"] == "rejected"
    assert motion["box_status"] == "intersects"
    assert motion["worst_stage"] in {"preopen_near_target", "approach_open", "insert_open"}


def test_m3641_staged_mode_limits_full_collision_candidates_and_reports_timing() -> None:
    instances, depth, intrinsics = _synthetic_scene(with_neighbor=True)
    raw = _config().raw
    raw["geometry_optimization"] = {
        "enabled": True,
        "mode": "staged",
        "initial_full_candidate_budget": 2,
        "maximum_full_candidate_budget": 3,
        "minimum_valid_full_candidates": 1,
        "cache_neighbor_point_clouds": True,
        "skip_rejected_pairs": True,
    }
    scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw))
    optimization = scene["geometry_optimization"]
    assert optimization["mode"] == "staged"
    assert optimization["light_candidate_count"] == 12
    assert 1 <= optimization["full_candidate_evaluated_count"] <= 3
    assert optimization["neighbor_base_cache"]["instance_count"] == 2
    assert scene["timing_ms"]["pair_geometry_initial_ms"] >= 0.0
    assert scene["timing_ms"]["full_candidate_evaluation_ms"] >= 0.0
    item = scene["instances"][0]
    candidates = item["grasp"]["clock_candidates"]
    assert len(candidates) == 12
    assert sum(bool(candidate.get("full_evaluated")) for candidate in candidates) == optimization["full_candidate_evaluated_count"]
    assert any(candidate.get("evaluation_stage") == "deferred" for candidate in candidates)
    assert item["candidate_evaluation"]["deferred_count"] > 0


def test_m3641_missing_optimization_section_keeps_exhaustive_compatibility() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    scene = analyze_scene(instances, depth, intrinsics, _config())
    assert scene["geometry_optimization"]["mode"] == "exhaustive"
    item = scene["instances"][0]
    candidates = item["grasp"]["clock_candidates"]
    assert len(candidates) == 12
    assert all(candidate.get("full_evaluated") is True for candidate in candidates)
    assert all(candidate.get("evaluation_stage") == "full" for candidate in candidates)
    assert scene["eligible_count"] == 1


def _first_valid_config(raw: dict) -> GeometryConfig:
    raw["geometry_optimization"] = {
        "enabled": True,
        "mode": "first_valid",
        "skip_rejected_pairs": True,
        "prefer_top_layer": True,
        "cache_neighbor_point_clouds": True,
        "stop_after_first_valid_target": True,
        "stop_after_first_valid_candidate": True,
        "maximum_pairs_to_fully_analyze": 3,
        "maximum_full_candidates_per_pair": 12,
    }
    raw["clock_search"] = {
        "mode": "adaptive_8_plus_4",
        "primary_clock_hours": [12, 2, 3, 5, 6, 8, 9, 11],
        "fallback_to_remaining": True,
    }
    raw["pair_preselection"] = {
        "depth_sample_stride": 4,
        "ring_erode_px": 1,
        "mouth_exclusion_px": 2,
    }
    return GeometryConfig(raw)


def _two_pair_scene():
    height, width = 260, 520
    depth = np.zeros((height, width), dtype=np.uint16)
    instances = []
    for pair_index, (cx, z) in enumerate(((135, 760), (385, 920))):
        ring = np.zeros((height, width), dtype=np.uint8)
        mouth = np.zeros_like(ring)
        cv2.circle(ring, (cx, 130), 42, 1, -1)
        cv2.circle(mouth, (cx, 130), 20, 1, -1)
        depth[ring.astype(bool)] = z
        depth[mouth.astype(bool)] = z + 60
        instances.extend(
            [
                _instance(pair_index * 2, 0, "foam_ring", ring),
                _instance(pair_index * 2 + 1, 1, "ring_mouth", mouth),
            ]
        )
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": width / 2.0, "cy": height / 2.0}
    return instances, depth, intrinsics


def test_m3642_first_valid_analyzes_only_first_successful_pair() -> None:
    instances, depth, intrinsics = _two_pair_scene()
    scene = analyze_scene(
        instances,
        depth,
        intrinsics,
        _first_valid_config(_config().raw),
    )
    optimization = scene["geometry_optimization"]
    assert scene["matched_pairs"] == 2
    assert scene["eligible_count"] == 1
    assert optimization["mode"] == "first_valid"
    assert optimization["fully_analyzed_pair_count"] == 1
    assert optimization["deferred_pair_count"] == 1
    assert optimization["early_exit_triggered"] is True
    assert optimization["full_candidate_evaluated_count"] == 1
    analyzed = [item for item in scene["instances"] if item.get("processing_status") != "deferred"]
    deferred = [item for item in scene["instances"] if item.get("processing_status") == "deferred"]
    assert len(analyzed) == 1
    assert len(deferred) == 1
    assert deferred[0]["deferred_reason"] == "after_first_valid_target"
    candidates = analyzed[0]["grasp"]["clock_candidates"]
    assert len(candidates) == 8
    assert {candidate.get("search_batch") for candidate in candidates} == {"primary"}


def test_m386_first_valid_compares_top_three_full_candidates_before_accept(monkeypatch) -> None:
    import production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry as geometry_module

    light_scores = {12: 100.0, 2: 90.0, 3: 80.0, 5: 70.0, 6: 60.0, 8: 50.0, 9: 40.0, 11: 30.0}
    full_scores = {12: 50.0, 2: 95.0, 3: 70.0}

    def fake_clock_candidate(clock, *args, evaluation_level="full", **kwargs):
        hour = int(clock.get("clock_hour"))
        light = str(evaluation_level).lower() != "full"
        return {
            **dict(clock),
            "evaluation_stage": "light" if light else "full",
            "full_evaluated": not light,
            "light_valid": True,
            "valid": not light,
            "score": light_scores.get(hour, 1.0) if light else full_scores.get(hour, 1.0),
            "light_score": light_scores.get(hour, 1.0),
            "warnings": [],
            "rejection_reasons": [],
            "neighbor_3d": {"status": "clear"},
            "full_gripper_static": {"status": "clear"},
            "full_gripper_motion": {"status": "clear"},
        }

    monkeypatch.setattr(geometry_module, "_clock_candidate", fake_clock_candidate)
    instances, depth, intrinsics = _synthetic_scene()
    raw = _config().raw
    config = _first_valid_config(raw)
    config.raw["geometry_optimization"]["minimum_full_candidates_before_accept"] = 3
    config.raw["geometry_optimization"]["maximum_full_candidates_per_pair"] = 6
    scene = analyze_scene(instances, depth, intrinsics, config)
    optimization = scene["geometry_optimization"]
    assert optimization["full_candidate_evaluated_count"] == 3
    assert optimization["full_candidate_valid_count"] == 3
    assert scene["selected_clock_hour"] == 2


def test_m3642_adaptive_clock_adds_four_fallbacks_only_after_primary_failure(monkeypatch) -> None:
    import production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry as geometry_module

    def fake_clock_candidate(clock, *args, evaluation_level="full", **kwargs):
        light = str(evaluation_level).lower() != "full"
        valid = (not light) and str(clock.get("search_batch")) == "fallback"
        return {
            **dict(clock),
            "evaluation_stage": "light" if light else "full",
            "full_evaluated": not light,
            "light_valid": True,
            "valid": bool(valid),
            "score": 90.0 if valid else 50.0,
            "light_score": 80.0,
            "warnings": [],
            "rejection_reasons": [] if valid or light else ["synthetic_primary_failure"],
            "neighbor_3d": {"status": "clear" if valid else "rejected"},
            "full_gripper_static": {"status": "clear" if valid else "rejected"},
            "full_gripper_motion": {"status": "clear" if valid else "rejected"},
        }

    monkeypatch.setattr(geometry_module, "_clock_candidate", fake_clock_candidate)
    instances, depth, intrinsics = _synthetic_scene()
    scene = analyze_scene(
        instances,
        depth,
        intrinsics,
        _first_valid_config(_config().raw),
    )
    optimization = scene["geometry_optimization"]
    assert optimization["adaptive_fallback_used"] is True
    assert optimization["primary_light_candidate_count"] == 8
    assert optimization["fallback_light_candidate_count"] == 4
    assert optimization["full_candidate_evaluated_count"] == 9
    assert optimization["full_candidate_valid_count"] == 1
    assert optimization["first_valid_candidate_search_batch"] == "fallback"
    candidates = scene["instances"][0]["grasp"]["clock_candidates"]
    assert len(candidates) == 12
    assert sum(candidate.get("search_batch") == "fallback" for candidate in candidates) == 4
    assert scene["selected_clock_search_batch"] == "fallback"


def test_m3642_command_mode_keeps_staged_and_exhaustive_compatibility() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    raw = _config().raw
    raw["geometry_optimization"] = {"enabled": True, "mode": "staged"}
    staged_scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw))
    assert staged_scene["geometry_optimization"]["mode"] == "staged"
    assert len(staged_scene["instances"][0]["grasp"]["clock_candidates"]) == 12

    raw2 = _config().raw
    raw2["geometry_optimization"] = {"enabled": False, "mode": "exhaustive"}
    exhaustive_scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw2))
    assert exhaustive_scene["geometry_optimization"]["mode"] == "exhaustive"
    assert len(exhaustive_scene["instances"][0]["grasp"]["clock_candidates"]) == 12


def test_m3923_preferred_1_to_3_clock_directions_are_evaluated_before_higher_score_nonpreferred(monkeypatch) -> None:
    import production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry as geometry_module

    # Deliberately give 12 o'clock the best geometry score. M39.2.3 should
    # still fully evaluate the empirically reachable 1/2/3 o'clock group first.
    light_scores = {12: 100.0, 1: 20.0, 2: 30.0, 3: 10.0, 5: 90.0, 6: 80.0, 8: 70.0, 9: 60.0, 11: 50.0}
    full_scores = {1: 60.0, 2: 75.0, 3: 55.0, 12: 99.0}

    def fake_clock_candidate(clock, *args, evaluation_level="full", **kwargs):
        hour = int(clock.get("clock_hour"))
        light = str(evaluation_level).lower() != "full"
        return {
            **dict(clock),
            "evaluation_stage": "light" if light else "full",
            "full_evaluated": not light,
            "light_valid": True,
            "valid": not light,
            "score": light_scores.get(hour, 1.0) if light else full_scores.get(hour, 50.0),
            "light_score": light_scores.get(hour, 1.0),
            "warnings": [],
            "rejection_reasons": [],
            "neighbor_3d": {"status": "clear"},
            "full_gripper_static": {"status": "clear"},
            "full_gripper_motion": {"status": "clear"},
        }

    monkeypatch.setattr(geometry_module, "_clock_candidate", fake_clock_candidate)
    instances, depth, intrinsics = _synthetic_scene()
    raw = _config().raw
    config = _first_valid_config(raw)
    config.raw["clock_search"].update({
        "preferred_clock_hours": [1, 2, 3],
        "prefer_preferred_clock": True,
        "promote_preferred_to_primary": True,
    })
    config.raw["geometry_optimization"]["minimum_full_candidates_before_accept"] = 3
    config.raw["geometry_optimization"]["maximum_full_candidates_per_pair"] = 6

    scene = analyze_scene(instances, depth, intrinsics, config)
    candidates = scene["instances"][0]["grasp"]["clock_candidates"]
    evaluated_hours = {
        int(candidate["clock_hour"])
        for candidate in candidates
        if bool(candidate.get("full_evaluated"))
    }
    assert evaluated_hours == {1, 2, 3}
    assert scene["selected_clock_hour"] == 2
    assert scene["selected_clock_preferred"] is True
    assert scene["preferred_clock_hours"] == [1, 2, 3]


def test_m3923_nonpreferred_direction_remains_fallback_when_all_preferred_are_invalid(monkeypatch) -> None:
    import production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry as geometry_module

    light_scores = {12: 100.0, 1: 30.0, 2: 20.0, 3: 10.0, 5: 90.0, 6: 80.0, 8: 70.0, 9: 60.0, 11: 50.0}

    def fake_clock_candidate(clock, *args, evaluation_level="full", **kwargs):
        hour = int(clock.get("clock_hour"))
        light = str(evaluation_level).lower() != "full"
        valid = (not light) and hour not in {1, 2, 3}
        return {
            **dict(clock),
            "evaluation_stage": "light" if light else "full",
            "full_evaluated": not light,
            "light_valid": True,
            "valid": bool(valid),
            "score": light_scores.get(hour, 1.0) if light else (90.0 if valid else 5.0),
            "light_score": light_scores.get(hour, 1.0),
            "warnings": [],
            "rejection_reasons": [] if valid or light else ["synthetic_preferred_invalid"],
            "neighbor_3d": {"status": "clear" if valid else "rejected"},
            "full_gripper_static": {"status": "clear" if valid else "rejected"},
            "full_gripper_motion": {"status": "clear" if valid else "rejected"},
        }

    monkeypatch.setattr(geometry_module, "_clock_candidate", fake_clock_candidate)
    instances, depth, intrinsics = _synthetic_scene()
    raw = _config().raw
    config = _first_valid_config(raw)
    config.raw["clock_search"].update({
        "preferred_clock_hours": [1, 2, 3],
        "prefer_preferred_clock": True,
        "promote_preferred_to_primary": True,
    })
    config.raw["geometry_optimization"]["minimum_full_candidates_before_accept"] = 3
    config.raw["geometry_optimization"]["maximum_full_candidates_per_pair"] = 6

    scene = analyze_scene(instances, depth, intrinsics, config)
    candidates = scene["instances"][0]["grasp"]["clock_candidates"]
    evaluated = [
        int(candidate["clock_hour"])
        for candidate in candidates
        if bool(candidate.get("full_evaluated"))
    ]
    assert {1, 2, 3}.issubset(set(evaluated))
    assert scene["selected_clock_hour"] == 12
    assert scene["selected_clock_preferred"] is False
