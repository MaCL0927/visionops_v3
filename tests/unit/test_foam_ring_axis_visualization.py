from __future__ import annotations

import cv2
import numpy as np
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.visualization import render_paired_axis_overlay


def _instance(instance_id: int, class_name: str, center: tuple[int, int], radius: int) -> SegmentationInstance:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=0 if class_name == "foam_ring" else 1,
        class_name=class_name,
        confidence=0.9,
        mask=mask.astype(bool),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _scene(axis: list[float], tilt_deg: float = 12.9) -> dict:
    return {
        "matched_pairs": 1,
        "instances": [
            {
                "ring_instance_id": 10,
                "mouth_instance_id": 11,
                "ring_center_camera_mm": [0.0, 0.0, 1000.0],
                "ring_axis_toward_camera": axis,
                "tilt_deg": tilt_deg,
                "pose": {"normal_source": "ellipse_stabilized"},
            }
        ],
    }


def test_axis_overlay_only_tints_successfully_paired_instances_and_keeps_fixed_3d_length() -> None:
    ring = _instance(10, "foam_ring", (80, 60), 24)
    mouth = _instance(11, "ring_mouth", (80, 60), 10)
    unmatched = _instance(12, "foam_ring", (25, 25), 8)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    scene = _scene([0.20, 0.10, -0.974679])
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}

    rendered, rows = render_paired_axis_overlay(
        image,
        [ring, mouth, unmatched],
        scene,
        intrinsics,
        {"rod_length_mm": 80.0},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "drawn"
    assert row["near_uv"] is not None
    assert row["far_uv"] is not None
    assert row["near_depth_mm"] < row["far_depth_mm"]
    assert row["depth_delta_far_minus_near_mm"] > 0.0
    assert row["depth_order_status"] == "toward_camera"
    assert np.linalg.norm(np.asarray(row["near_camera_mm"]) - np.asarray(row["far_camera_mm"])) == pytest.approx(80.0)
    assert row["projected_rod_length_px"] > 1.0
    # Unmatched object is not tinted or outlined.
    assert rendered[25, 25].tolist() == [0, 0, 0]
    # Paired object and centered rod are visible.
    assert int(rendered[60, 80].sum()) > 0


def test_near_optical_axis_uses_overlapped_near_far_marker_without_fake_direction() -> None:
    ring = _instance(10, "foam_ring", (80, 60), 24)
    mouth = _instance(11, "ring_mouth", (80, 60), 10)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}

    _, rows = render_paired_axis_overlay(
        image,
        [ring, mouth],
        _scene([0.0, 0.0, -1.0], 0.0),
        intrinsics,
        {"rod_length_mm": 80.0},
    )

    row = rows[0]
    assert row["status"] == "near_optical_axis"
    assert row["near_uv"] == pytest.approx(row["far_uv"])
    assert row["projected_rod_length_px"] == pytest.approx(0.0)
    assert row["near_depth_mm"] == pytest.approx(960.0)
    assert row["far_depth_mm"] == pytest.approx(1040.0)


def test_fixed_3d_rod_projection_gets_longer_as_tilt_increases() -> None:
    ring = _instance(10, "foam_ring", (80, 60), 24)
    mouth = _instance(11, "ring_mouth", (80, 60), 10)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 80.0, "cy": 60.0}
    cfg = {"rod_length_mm": 80.0, "minimum_projected_rod_px": 0.1}

    _, shallow_rows = render_paired_axis_overlay(
        image,
        [ring, mouth],
        _scene([0.173648, 0.0, -0.984808], 10.0),
        intrinsics,
        cfg,
    )
    _, steep_rows = render_paired_axis_overlay(
        image,
        [ring, mouth],
        _scene([0.707107, 0.0, -0.707107], 45.0),
        intrinsics,
        cfg,
    )

    assert steep_rows[0]["projected_rod_length_px"] > shallow_rows[0]["projected_rod_length_px"] * 3.0


def test_axis_sign_inconsistency_is_exposed_instead_of_silently_swapped() -> None:
    ring = _instance(10, "foam_ring", (80, 60), 24)
    mouth = _instance(11, "ring_mouth", (80, 60), 10)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 80.0, "cy": 60.0}

    _, rows = render_paired_axis_overlay(
        image,
        [ring, mouth],
        _scene([0.5, 0.0, 0.866025], 30.0),
        intrinsics,
        {"rod_length_mm": 80.0},
    )

    assert rows[0]["depth_order_status"] == "inconsistent_axis_sign"
    assert rows[0]["status"] == "drawn_sign_warning"
