"""M38.6 one-shot direction/collision/contact-fix validation.

M36.5 reuses the same :class:`OnlineGeometryProcessor` in a persistent service;
this command intentionally starts and stops the processor once so the historical
single-trigger regression path remains available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep these aliases in this module so existing tests and local diagnosis can
# inject fakes without touching the persistent processor implementation.
from production.common.runtime_ipc import RuntimeIpcClient  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    OnlineGeometryError,
    OnlineGeometryProcessor,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.rgbd_cache import (  # noqa: E402
    RgbdFrameCache,
)


def run_once(
    *,
    config_path: Path,
    runtime_url: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exact_match_timeout_ms: int | None = None,
    geometry_mode: str | None = None,
) -> Dict[str, Any]:
    processor = OnlineGeometryProcessor(
        config_path=config_path,
        runtime_url=runtime_url,
        output_root=output_root,
        exact_match_timeout_ms=exact_match_timeout_ms,
        geometry_mode=geometry_mode,
        runtime_status_ttl_ms=0,
        client_factory=RuntimeIpcClient,
        cache_factory=RgbdFrameCache,
        analyze_fn=analyze_scene,
    )
    processor.start()
    try:
        return processor.process(
            save_debug=None,
            generate_overlay=False,
            stage="M38.6_direction_collision_contact_fix_online_geometry_once",
        ).payload
    finally:
        processor.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M38.6：M38.1前三方向比较、确认碰撞硬拒绝、分支C直观原因、纯侧面近端15%外接触点",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-url")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exact-match-timeout-ms", type=int)
    parser.add_argument(
        "--geometry-mode",
        choices=("first_valid", "staged", "exhaustive"),
        help="覆盖line.yaml中的geometry_optimization.mode，用于first_valid/staged/exhaustive对照",
    )
    parser.add_argument(
        "--print-full-json",
        action="store_true",
        help="终端打印完整结果；默认只打印精简摘要",
    )
    return parser


def _summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else {}
    timing = payload.get("timing_ms") if isinstance(payload.get("timing_ms"), Mapping) else {}
    branch_c = scene.get("m38_4_branch_c") if isinstance(scene.get("m38_4_branch_c"), Mapping) else {}
    branch_d = scene.get("m38_5_branch_d") if isinstance(scene.get("m38_5_branch_d"), Mapping) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "capture_timestamp_ms": payload.get("capture_timestamp_ms"),
        "timestamp_delta_ms": (payload.get("rgbd_match") or {}).get("timestamp_delta_ms"),
        "rings_detected": scene.get("rings_detected"),
        "mouths_detected": scene.get("mouths_detected"),
        "matched_pairs": scene.get("matched_pairs"),
        "eligible_count": scene.get("eligible_count"),
        "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
        "selected_clock_hour": scene.get("selected_clock_hour"),
        "selected_clock_angle_deg_cw_from_12": scene.get("selected_clock_angle_deg_cw_from_12"),
        "selected_clock_search_batch": scene.get("selected_clock_search_batch"),
        "selected_grasp_branch": (scene.get("hybrid_grasp") or {}).get("selected_branch") or scene.get("selected_grasp_branch"),
        "terminal_reject": branch_c.get("fast_terminated"),
        "terminal_reject_reason": branch_c.get("decision"),
        "terminal_reject_message": branch_c.get("reason"),
        "terminal_reject_display": branch_c.get("display_reason_short"),
        "collision_check_performed": branch_c.get("collision_check_performed"),
        "operator_action": branch_c.get("operator_action"),
        "fallback_triggered": (scene.get("hybrid_grasp") or {}).get("fallback_triggered"),
        "m38_5_candidate_found": branch_d.get("candidate_found"),
        "m38_5_selected_ring_instance_id": branch_d.get("selected_ring_instance_id"),
        "m38_5_robot_ready": branch_d.get("robot_ready"),
        "m38_5_scope": branch_d.get("scope"),
        "m37_evaluated_count": (scene.get("side_ring_branch") or {}).get("evaluated_count"),
        "m37_selected_ring_instance_id": (scene.get("side_ring_branch") or {}).get("selected_ring_instance_id"),
        "selected_depth_layer_index": (scene.get("depth_layering") or {}).get("selected_layer_index"),
        "selected_surface_depth_mm": (scene.get("depth_layering") or {}).get("selected_surface_depth_mm"),
        "selected_depth_rank": (scene.get("depth_layering") or {}).get("selected_depth_rank"),
        "m37_fast_attempt_count": (scene.get("side_ring_branch") or {}).get("fast_attempt_count"),
        "m37_accurate_refinement_count": (scene.get("side_ring_branch") or {}).get("accurate_refinement_count"),
        "runtime_total_ms": ((payload.get("runtime") or {}).get("timing") or {}).get("total_ms"),
        "polygon_to_mask_ms": timing.get("polygon_to_mask_ms"),
        "geometry_ms": timing.get("geometry_ms"),
        "geometry_mode": (scene.get("geometry_optimization") or {}).get("mode"),
        "light_candidate_count": (scene.get("geometry_optimization") or {}).get("light_candidate_count"),
        "full_candidate_evaluated_count": (scene.get("geometry_optimization") or {}).get("full_candidate_evaluated_count"),
        "full_candidate_valid_count": (scene.get("geometry_optimization") or {}).get("full_candidate_valid_count"),
        "fully_analyzed_pair_count": (scene.get("geometry_optimization") or {}).get("fully_analyzed_pair_count"),
        "deferred_pair_count": (scene.get("geometry_optimization") or {}).get("deferred_pair_count"),
        "adaptive_fallback_used": (scene.get("geometry_optimization") or {}).get("adaptive_fallback_used"),
        "early_exit_triggered": (scene.get("geometry_optimization") or {}).get("early_exit_triggered"),
        "geometry_breakdown_ms": scene.get("timing_ms"),
        "hybrid_timing_ms": (scene.get("hybrid_grasp") or {}).get("timing_ms"),
        "full_candidate_timing": (scene.get("timing_detail") or {}).get("full_candidates"),
        "total_ms": timing.get("total_ms"),
        "robot_ready": payload.get("robot_ready"),
        "files": payload.get("files"),
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = run_once(
            config_path=args.config,
            runtime_url=args.runtime_url,
            output_root=args.output,
            exact_match_timeout_ms=args.exact_match_timeout_ms,
            geometry_mode=args.geometry_mode,
        )
    except (OnlineGeometryError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[FAIL] M38.6 hybrid online geometry failed: {error}", file=sys.stderr)
        return 2
    document = payload if args.print_full_json else _summary(payload)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print("[PASS] M38.6 direction/collision/contact-fix online geometry completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
