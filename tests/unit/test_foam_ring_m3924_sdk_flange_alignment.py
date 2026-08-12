from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "production" / "foam_ring_grasp" / "scripts" / "test_m39_2_4_sdk_flange_alignment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("m3924_sdk_flange_alignment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pose_roundtrip_and_fixed_transform_recovery() -> None:
    m = _load_module()
    hand = [420.0, 120.0, 550.0, 0.20, -0.30, 0.10, 0.92]
    Tbh = m.pose7_to_T(hand)
    Ttool = np.array(
        [
            [0.997305, -0.047329, 0.056068, -170.52],
            [-0.066196, -0.910000, 0.409290, -31.54],
            [0.031650, -0.411899, -0.910680, 17.83],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Re-orthogonalize the example rotation so the pose conversion is exact.
    U, _, Vt = np.linalg.svd(Ttool[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    Ttool[:3, :3] = R
    Tactual = Tbh @ Ttool
    actual_pose = m.T_to_pose7(Tactual)
    recovered = m.invert_T(Tbh) @ m.pose7_to_T(actual_pose)
    np.testing.assert_allclose(recovered, Ttool, atol=1e-8)


def test_fit_transform_recovers_common_tool_transform() -> None:
    m = _load_module()
    base = np.eye(4, dtype=float)
    base[:3, 3] = [-170.0, -31.0, 18.0]
    angle = np.deg2rad(156.0)
    base[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    records = []
    perturb = [(-2.0, 1.0, 0.0), (0.0, 0.0, 1.0), (2.0, -1.0, -1.0)]
    for i, dt in enumerate(perturb):
        T = base.copy()
        T[:3, 3] += np.asarray(dt)
        records.append({"request_id": str(i), "T_hand_tcp_sdk_flange_mm": T.tolist()})
    fit = m.fit_transform(records, clock_hour=1)
    np.testing.assert_allclose(fit["translation_mm"], [-170.0, -31.0, 18.0], atol=1e-9)
    assert fit["sample_count"] == 3
    assert fit["fit_residuals"]["rotation_max_deg"] == pytest.approx(0.0, abs=1e-6)



def test_m3926_sdk_flange_alignment_default_clock_is_3() -> None:
    m = _load_module()
    args = m.build_parser().parse_args([])
    assert args.clock_hour == 3
