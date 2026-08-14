from __future__ import annotations

import math
import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.tilt_evidence import analyze_tilt_evidence


def _synthetic_ring(tilt_deg: float, *, noise_mm: float = 0.5, corrupt: dict[int, float] | None = None):
    rng = np.random.default_rng(7)
    sector_count = 16
    center_uv = (320.0, 320.0)
    radius_mm = 50.0
    points = []
    pixels = []
    slope = math.tan(math.radians(tilt_deg))
    for sector in range(sector_count):
        theta = (sector + 0.5) * 2.0 * math.pi / sector_count
        base_x = radius_mm * math.cos(theta)
        base_y = radius_mm * math.sin(theta)
        base_h = slope * base_x
        if corrupt and sector in corrupt:
            base_h += float(corrupt[sector])
        for j in range(10):
            points.append([
                base_x + rng.normal(0.0, 0.4),
                base_y + rng.normal(0.0, 0.4),
                700.0 + base_h + rng.normal(0.0, noise_mm),
            ])
            pixels.append([
                center_uv[0] + 50.0 * math.cos(theta) + rng.normal(0.0, 0.3),
                center_uv[1] + 50.0 * math.sin(theta) + rng.normal(0.0, 0.3),
            ])
    points = np.asarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    # h=gx*x+gy*y+c -> plane normal [gx, gy, -1]
    raw_normal = np.asarray([slope, 0.0, -1.0], dtype=np.float64)
    raw_normal /= np.linalg.norm(raw_normal)
    return points, pixels, center_uv, raw_normal


def _cfg():
    return {
        "sector_count": 16,
        "minimum_sector_points": 5,
        "sector_local_band_half_width_mm": 8.0,
        "harmonic_ransac_residual_mm": 4.0,
        "minimum_valid_sectors": 10,
        "minimum_harmonic_inliers": 5,
        "tilted_raw_plane_min_deg": 12.0,
        "tilted_sector_gradient_min_deg": 10.0,
        "tilted_peak_to_peak_min_mm": 18.0,
        "flat_sector_gradient_max_deg": 8.0,
        "flat_peak_to_peak_max_mm": 12.0,
        "flat_raw_plane_max_deg": 6.0,
        "severe_incoherence_residual_mm": 40.0,
        "low_raw_flat_override_max_residual_mm": 20.0,
    }


def test_m3930_flat_with_local_bad_sectors_stays_flat():
    points, pixels, center, normal = _synthetic_ring(0.0, corrupt={3: 35.0, 11: -30.0})
    result = analyze_tilt_evidence(
        points,
        pixels,
        center,
        box_x_camera=[1.0, 0.0, 0.0],
        box_y_camera=[0.0, 1.0, 0.0],
        box_z_inside_camera=[0.0, 0.0, 1.0],
        config=_cfg(),
        raw_plane_normal_toward_camera=normal,
        raw_plane_inlier_ratio=0.8,
        raw_plane_residual_p95_mm=3.0,
    )
    assert result["state"] == "FLAT"
    assert result["production_routing_enabled"] is False


def test_m3930_realistic_15deg_gradient_is_tilted():
    points, pixels, center, normal = _synthetic_ring(15.0)
    result = analyze_tilt_evidence(
        points,
        pixels,
        center,
        box_x_camera=[1.0, 0.0, 0.0],
        box_y_camera=[0.0, 1.0, 0.0],
        box_z_inside_camera=[0.0, 0.0, 1.0],
        config=_cfg(),
        raw_plane_normal_toward_camera=normal,
        raw_plane_inlier_ratio=0.85,
        raw_plane_residual_p95_mm=2.0,
    )
    assert result["state"] == "TILTED"
    assert result["sector_gradient"]["predicted_peak_to_peak_mm"] > 20.0
    assert 12.0 <= result["sector_gradient"]["sector_gradient_tilt_deg"] <= 18.0


def test_m3930_severely_incoherent_small_gradient_is_uncertain():
    # Alternating large sector offsets have no trustworthy ring-wide front face.
    corrupt = {sector: (70.0 if sector % 2 == 0 else -70.0) for sector in range(16)}
    points, pixels, center, normal = _synthetic_ring(0.0, noise_mm=0.4, corrupt=corrupt)
    result = analyze_tilt_evidence(
        points,
        pixels,
        center,
        box_x_camera=[1.0, 0.0, 0.0],
        box_y_camera=[0.0, 1.0, 0.0],
        box_z_inside_camera=[0.0, 0.0, 1.0],
        config=_cfg(),
        raw_plane_normal_toward_camera=normal,
        raw_plane_inlier_ratio=0.4,
        raw_plane_residual_p95_mm=20.0,
    )
    assert result["state"] == "UNCERTAIN"
