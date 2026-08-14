#!/usr/bin/env python3
"""Preflight checks for the cleaned foam_ring_grasp production package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.robot_pose_transform import RobotPoseTransformer

DEFAULT_CONFIG = REPO_ROOT / "production" / "foam_ring_grasp" / "config" / "line.yaml"
STATUS_URL = "http://127.0.0.1:19213/api/foam_ring/status"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _clock_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    flat = _mapping(raw.get("clock_search") or {}, "clock_search")
    tilted = _mapping(
        raw.get("m39_3_4_1_tilted_production_routing") or {},
        "m39_3_4_1_tilted_production_routing",
    )
    flat_pref = [int(v) for v in flat.get("preferred_clock_hours") or []]
    flat_primary = [int(v) for v in flat.get("primary_clock_hours") or []]
    tilted_pref = [int(v) for v in tilted.get("preferred_clock_hours") or []]
    tilted_fallback = [int(v) for v in tilted.get("fallback_clock_hours") or []]
    ok = (
        flat_pref == [3]
        and flat_primary == [3]
        and not bool(flat.get("fallback_to_remaining", True))
        and tilted_pref == [3]
        and not bool(tilted.get("fallback_enabled", True))
        and tilted_fallback == []
        and int(tilted.get("maximum_clock_candidates", 0)) == 1
    )
    if not ok:
        raise RuntimeError(
            "clock contract violated: production must be clock-3 only with all fallbacks disabled"
        )
    return {
        "status": "ok",
        "flat": {"preferred": flat_pref, "primary": flat_primary, "fallback": False},
        "tilted": {"preferred": tilted_pref, "fallback": False, "max_candidates": 1},
    }


def _visible_mouth_scope(raw: Mapping[str, Any]) -> dict[str, Any]:
    branch_a = _mapping(raw.get("m38_branch_a") or {}, "m38_branch_a")
    branch_b = _mapping(raw.get("m38_branch_b") or {}, "m38_branch_b")
    branch_d = _mapping(raw.get("m38_branch_d") or {}, "m38_branch_d")
    branch_c = _mapping(raw.get("m38_branch_c") or {}, "m38_branch_c")
    hybrid = _mapping(raw.get("hybrid_grasp") or {}, "hybrid_grasp")
    if not bool(branch_a.get("enabled", False)):
        raise RuntimeError("visible-mouth Branch A must remain enabled")
    if bool(branch_b.get("enabled", False)) or bool(branch_d.get("enabled", False)):
        raise RuntimeError("pre-M39.4 side/no-mouth branches B/D must remain disabled")
    if not bool(branch_c.get("enabled", False)) or not bool(branch_c.get("fast_terminate", False)):
        raise RuntimeError("pre-M39.4 no-mouth conservative reject must remain enabled")
    if bool(hybrid.get("legacy_m36_enabled", True)) or bool(hybrid.get("side_ring_fallback_enabled", True)):
        raise RuntimeError("legacy M36/M37 fallback must remain disabled in production")
    return {
        "status": "ok",
        "visible_mouth_branch": "enabled",
        "side_no_mouth": "reject_until_M39.4",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cleaned foam-ring production configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--service", action="store_true", help="also query the running 19213 service")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    raw = load_yaml(config_path)
    transformer = RobotPoseTransformer.from_mapping(raw, config_path)
    report: dict[str, Any] = {
        "status": "ok",
        "config": str(config_path),
        "task": raw.get("task"),
        "schema_version": raw.get("schema_version"),
        "clock_contract": _clock_contract(raw),
        "scope_contract": _visible_mouth_scope(raw),
        "robot_pose_transform": transformer.status(),
    }
    if args.service:
        response = requests.get(STATUS_URL, timeout=5.0)
        response.raise_for_status()
        service = response.json()
        if service.get("status") != "ok":
            raise RuntimeError(f"service status abnormal: {service}")
        report["service"] = service
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
