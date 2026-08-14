from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision import online_processor as op
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.ring_prior_surface import reconstruct_ring_prior_surface


class _Geometry:
    def section(self, name):
        if name == "depth":
            return {"minimum_mm": 150.0, "maximum_mm": 3000.0}
        return {}


def _scene():
    return {
        "selected_ring_instance_id": 1,
        "robot_candidate": {"pose": [1, 2, 3]},
        "instances": [{
            "ring_instance_id": 1,
            "mouth_instance_id": 2,
            "pose_strategy": "m38_1_front_annulus",
            "tilt_deg": 1.886,
            "eligible": True,
            "m38_branch_a": {},
            "mouth_ellipse": {"center_uv": [20.0, 20.0]},
        }],
    }


def _instances():
    yy, xx = np.indices((40, 40))
    rr = np.hypot(xx - 20, yy - 20)
    return [
        SimpleNamespace(instance_id=1, mask=(rr >= 8) & (rr <= 13)),
        SimpleNamespace(instance_id=2, mask=rr <= 8),
    ]


def test_m3932_synthetic_ring_prior_reconstructs_front_surface():
    yy, xx = np.indices((120, 120))
    cx = cy = 60.0
    rr = np.hypot(xx - cx, yy - cy)
    mouth = rr <= 24
    ring = (rr >= 24) & (rr <= 34)
    fx = fy = 600.0
    gx = np.tan(np.radians(15.0))
    z = 750.0 / (1.0 - gx * (xx - cx) / fx)
    depth = np.zeros((120, 120), dtype=np.uint16)
    depth[ring] = np.rint(z[ring]).astype(np.uint16)
    theta = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    sector = np.floor(theta * 16 / (2 * np.pi)).astype(int)
    depth[ring & np.isin(sector, [0, 1, 8, 9])] += 70

    evidence = reconstruct_ring_prior_surface(
        depth,
        ring,
        mouth,
        (cx, cy),
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        box_x_camera=[1, 0, 0],
        box_y_camera=[0, 1, 0],
        box_z_inside_camera=[0, 0, 1],
        object_geometry={
            "nominal_inner_diameter_mm": 60.0,
            "nominal_outer_diameter_mm": 85.0,
            "nominal_wall_thickness_mm": 14.0,
            "axial_length_mm": 70.0,
        },
        config={"sector_count": 16, "sample_wall_start_ratio": 0.15, "sample_wall_end_ratio": 0.50},
    )
    assert evidence["surface"] is not None
    assert 9.0 <= evidence["surface"]["tilt_deg"] <= 22.0
    assert max(
        row["selected_candidate"]["depth_mm"]
        for row in evidence["surface"]["sector_samples"]
        if row.get("selected_candidate")
    ) < 800.0
    assert evidence["uses_absolute_box_floor_depth_for_identity"] is False


def test_m3932_online_error_is_diagnostic_only(monkeypatch):
    scene = _scene()
    before = deepcopy(scene["robot_candidate"])
    monkeypatch.setattr(op, "_box_reference_axes_camera", lambda _cfg: (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ))
    monkeypatch.setattr(op, "reconstruct_ring_prior_surface", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    summary = op._attach_m3932_ring_prior_diagnostics(
        scene,
        _instances(),
        np.full((40, 40), 750, dtype=np.uint16),
        {"fx": 600.0, "fy": 600.0, "cx": 20.0, "cy": 20.0},
        raw_config={"m39_3_2_ring_prior_surface": {"enabled": True}, "object_geometry": {}},
        geometry_config=_Geometry(),
    )
    evidence = scene["instances"][0]["m38_branch_a"]["m39_3_2_ring_prior_surface"]
    assert evidence["status"] == "ERROR"
    assert evidence["production_routing_enabled"] is False
    assert summary["classification_counts"]["ERROR"] == 1
    assert scene["robot_candidate"] == before
    assert scene["instances"][0]["tilt_deg"] == 1.886


def test_m3932_online_attach_never_routes(monkeypatch):
    scene = _scene()
    before = deepcopy(scene["robot_candidate"])
    monkeypatch.setattr(op, "_box_reference_axes_camera", lambda _cfg: (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ))
    monkeypatch.setattr(op, "reconstruct_ring_prior_surface", lambda *args, **kwargs: {
        "status": "RECONSTRUCTED",
        "classification": "TILTED",
        "surface": {"tilt_deg": 15.0},
        "production_routing_enabled": False,
    })
    summary = op._attach_m3932_ring_prior_diagnostics(
        scene,
        _instances(),
        np.full((40, 40), 750, dtype=np.uint16),
        {"fx": 600.0, "fy": 600.0, "cx": 20.0, "cy": 20.0},
        raw_config={"m39_3_2_ring_prior_surface": {"enabled": True}, "object_geometry": {}},
        geometry_config=_Geometry(),
    )
    evidence = scene["instances"][0]["m38_branch_a"]["m39_3_2_ring_prior_surface"]
    assert evidence["classification"] == "TILTED"
    assert evidence["production_routing_enabled"] is False
    assert evidence["production_final_tilt_deg"] == 1.886
    assert summary["classification_counts"]["TILTED"] == 1
    assert scene["robot_candidate"] == before
    assert scene["instances"][0]["tilt_deg"] == 1.886
