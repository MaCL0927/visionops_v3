from __future__ import annotations

from pathlib import Path

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    _resolve_pose_conflict_policy,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (
    HybridGraspConfig,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_template import (
    SideRingTemplateConfig,
    _depth_gradient_hard_rejection_required,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LINE_CONFIG = REPO_ROOT / "production/foam_ring_grasp/config/line.yaml"


def test_m3761_production_stoploss_configuration():
    raw = load_yaml(LINE_CONFIG)
    side = SideRingTemplateConfig.from_mapping(raw)
    hybrid = HybridGraspConfig.from_mapping(raw)

    assert side.multi_surface_depth_gradient_weight == 0.0
    assert side.multi_surface_depth_gradient_hard_reject_enabled is False
    assert hybrid.maximum_accurate_refinements_per_trigger == 0
    assert raw["pose"]["pose_conflict_policy"] == "fallback_to_m37"


def test_m3761_depth_gradient_is_diagnostic_by_default():
    config = SideRingTemplateConfig.from_mapping(
        {
            "side_ring_template": {
                "multi_surface_maximum_depth_gradient_axis_error_deg": 18.0,
            }
        }
    )
    assert config.multi_surface_depth_gradient_weight == 0.0
    assert config.multi_surface_depth_gradient_hard_reject_enabled is False
    assert not _depth_gradient_hard_rejection_required(
        mouth_present=False,
        axis_error_deg=65.0,
        config=config,
    )


def test_legacy_depth_gradient_hard_gate_can_be_reenabled_explicitly():
    config = SideRingTemplateConfig.from_mapping(
        {
            "side_ring_template": {
                "multi_surface_depth_gradient_hard_reject_enabled": True,
                "multi_surface_maximum_depth_gradient_axis_error_deg": 18.0,
            }
        }
    )
    assert _depth_gradient_hard_rejection_required(
        mouth_present=False,
        axis_error_deg=19.0,
        config=config,
    )
    assert not _depth_gradient_hard_rejection_required(
        mouth_present=True,
        axis_error_deg=80.0,
        config=config,
    )


def test_pose_conflict_policy_defaults_to_m37_handoff():
    assert _resolve_pose_conflict_policy({}) == "fallback_to_m37"
    assert (
        _resolve_pose_conflict_policy({"pose_conflict_policy": "fallback_to_m37"})
        == "fallback_to_m37"
    )
    assert (
        _resolve_pose_conflict_policy({"pose_conflict_hard_reject_enabled": True})
        == "hard_reject"
    )
    assert (
        _resolve_pose_conflict_policy({"pose_conflict_hard_reject_enabled": False})
        == "warn_only"
    )


def test_hybrid_default_disables_online_local_accurate():
    config = HybridGraspConfig.from_mapping({})
    assert config.maximum_accurate_refinements_per_trigger == 0
