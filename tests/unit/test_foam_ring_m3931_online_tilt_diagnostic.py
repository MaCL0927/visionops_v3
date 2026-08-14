from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision import online_processor as op


class _Geometry:
    def section(self, name):
        if name == "depth":
            return {"minimum_mm": 150.0, "maximum_mm": 3000.0}
        return {}


def _scene():
    return {
        "selected_ring_instance_id": 1,
        "robot_candidate": {"pose": [1, 2, 3]},
        "instances": [
            {
                "ring_instance_id": 1,
                "mouth_instance_id": 2,
                "pose_strategy": "m38_1_front_annulus",
                "tilt_deg": 1.886,
                "eligible": True,
                "m38_branch_a": {"normal_source": "m39_2_9_calibrated_box_floor"},
                "mouth_ellipse": {"center_uv": [1.5, 1.5]},
                "_debug": {"m38_raw_annulus_mask": np.ones((4, 4), dtype=bool)},
            }
        ],
    }


def _instances():
    return [
        SimpleNamespace(instance_id=1, mask=np.ones((4, 4), dtype=bool)),
        SimpleNamespace(instance_id=2, mask=np.zeros((4, 4), dtype=bool)),
    ]


def test_m3931_disabled_does_not_change_production_scene():
    scene = _scene()
    before_candidate = deepcopy(scene["robot_candidate"])
    before_tilt = scene["instances"][0]["tilt_deg"]
    summary = op._attach_m3931_online_tilt_diagnostics(
        scene,
        _instances(),
        np.full((4, 4), 700, dtype=np.uint16),
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        raw_config={"m39_3_1_tilt_evidence": {"enabled": False}},
        geometry_config=_Geometry(),
    )
    assert summary["status"] == "disabled"
    assert scene["robot_candidate"] == before_candidate
    assert scene["instances"][0]["tilt_deg"] == before_tilt


def test_m3931_detector_error_is_diagnostic_only(monkeypatch):
    scene = _scene()
    before_candidate = deepcopy(scene["robot_candidate"])
    monkeypatch.setattr(op, "_box_reference_axes_camera", lambda _cfg: (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ))
    # Remove the raw mask so the detector fails inside the per-instance guard.
    scene["instances"][0]["_debug"] = {}
    summary = op._attach_m3931_online_tilt_diagnostics(
        scene,
        _instances(),
        np.full((4, 4), 700, dtype=np.uint16),
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        raw_config={
            "m39_3_1_tilt_evidence": {"enabled": True},
            "m39_3_0_tilt_evidence": {},
        },
        geometry_config=_Geometry(),
    )
    evidence = scene["instances"][0]["m38_branch_a"]["m39_3_1_tilt_evidence"]
    assert evidence["state"] == "ERROR"
    assert evidence["production_routing_enabled"] is False
    assert summary["state_counts"]["ERROR"] == 1
    assert scene["robot_candidate"] == before_candidate
    assert scene["instances"][0]["eligible"] is True
    assert scene["instances"][0]["tilt_deg"] == 1.886


def test_m3931_attaches_tilted_evidence_without_routing(monkeypatch):
    scene = _scene()
    before_candidate = deepcopy(scene["robot_candidate"])
    monkeypatch.setattr(op, "_box_reference_axes_camera", lambda _cfg: (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ))
    monkeypatch.setattr(op, "build_tilt_core_mask", lambda *args, **kwargs: np.ones((4, 4), dtype=bool))
    pts = np.array([[0.0, 0.0, 700.0], [1.0, 0.0, 700.0], [0.0, 1.0, 700.0]])
    pix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(op, "depth_pixels_to_points", lambda *args, **kwargs: (pts, pix))
    plane = SimpleNamespace(normal=np.array([0.2, 0.0, -0.98]), inlier_ratio=0.9, residual_p95_mm=2.0)
    monkeypatch.setattr(op, "fit_plane_ransac", lambda *args, **kwargs: plane)
    monkeypatch.setattr(op, "analyze_tilt_evidence", lambda *args, **kwargs: {
        "state": "TILTED",
        "confidence": "strong",
        "classification_reason": "test",
        "raw_plane_cue": {"tilt_deg": 15.0},
        "sector_gradient": {"sector_gradient_tilt_deg": 14.0, "predicted_peak_to_peak_mm": 25.0},
    })
    summary = op._attach_m3931_online_tilt_diagnostics(
        scene,
        _instances(),
        np.full((4, 4), 700, dtype=np.uint16),
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        raw_config={
            "m39_3_1_tilt_evidence": {"enabled": True, "detector_config_source": "m39_3_0_tilt_evidence"},
            "m39_3_0_tilt_evidence": {},
        },
        geometry_config=_Geometry(),
    )
    evidence = scene["instances"][0]["m38_branch_a"]["m39_3_1_tilt_evidence"]
    assert evidence["state"] == "TILTED"
    assert evidence["mode"] == "online_diagnostic_only"
    assert evidence["production_routing_enabled"] is False
    assert evidence["production_final_tilt_deg"] == 1.886
    assert summary["state_counts"]["TILTED"] == 1
    assert scene["robot_candidate"] == before_candidate
    assert scene["instances"][0]["tilt_deg"] == 1.886
