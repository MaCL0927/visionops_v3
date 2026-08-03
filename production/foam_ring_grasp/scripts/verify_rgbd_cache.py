#!/usr/bin/env python3
"""M36.3 acceptance: exact Runtime RGB timestamp -> cached D2C depth match."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.rgbd_cache import (
    RgbdFrame,
    RgbdFrameCache,
    load_rgbd_cache_settings,
)


def _request_json(base_url: str, path: str, *, method: str = "GET", timeout: float = 8.0) -> dict[str, Any]:
    request = Request(base_url.rstrip("/") + path, method=method, data=b"" if method != "GET" else None)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - localhost device API
            body = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{method} {path} did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return value


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def _depth_metrics(frame: RgbdFrame) -> dict[str, Any]:
    depth = frame.depth_mm
    valid = depth > 0
    valid_count = int(np.count_nonzero(valid))
    total = int(depth.size)
    if valid_count:
        values = depth[valid]
        minimum = int(values.min())
        maximum = int(values.max())
        median = float(np.median(values))
    else:
        minimum = maximum = 0
        median = 0.0
    return {
        "valid_count": valid_count,
        "total_count": total,
        "valid_ratio": round(valid_count / total, 6) if total else 0.0,
        "minimum_mm": minimum,
        "maximum_mm": maximum,
        "median_mm": round(median, 3),
    }


def _save_sample(directory: Path, frame: RgbdFrame, result: dict[str, Any]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = str(frame.timestamp_epoch_ms)
    depth_path = directory / f"{stem}_depth.npy"
    np.save(depth_path, frame.depth_mm)
    files = {"depth_npy": str(depth_path)}
    if frame.rgb is not None:
        rgb_path = directory / f"{stem}_rgb.npy"
        np.save(rgb_path, frame.rgb)
        files["rgb_npy"] = str(rgb_path)
    metadata_path = directory / f"{stem}_meta.json"
    metadata = {
        "schema_version": "1.0",
        "stage": "M36.3",
        "frame": frame.metadata(),
        "runtime": {
            "capture_timestamp_ms": result.get("capture_timestamp_ms"),
            "frame_id": result.get("frame_id"),
            "result_id": result.get("result_id"),
        },
        "depth": _depth_metrics(frame),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["metadata_json"] = str(metadata_path)
    return files


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    bridge_health = _request_json(args.bridge_url, "/health")
    runtime_status = _request_json(args.runtime_url, "/api/runtime/status")
    loaded_model = runtime_status.get("loaded_model") if isinstance(runtime_status.get("loaded_model"), dict) else {}
    source = runtime_status.get("frame_source") if isinstance(runtime_status.get("frame_source"), dict) else {}

    if loaded_model.get("task_type") != "segmentation":
        failures.append(f"Runtime task_type is {loaded_model.get('task_type')!r}, expected 'segmentation'")
    if source.get("configured_transport") != "posix_shared_memory":
        failures.append("Runtime RGB configured_transport is not posix_shared_memory")
    if source.get("transport") not in (None, "posix_shared_memory"):
        failures.append(f"Runtime RGB active transport is {source.get('transport')!r}")
    if not bridge_health.get("shared_rgb_ready"):
        failures.append("Bridge shared RGB is not ready")
    if not bridge_health.get("shared_depth_ready"):
        failures.append("Bridge shared depth is not ready")
    if not bridge_health.get("shared_depth_calibration_ready"):
        failures.append("Bridge shared depth calibration is not ready")
    if int(bridge_health.get("color_width") or 0) != int(bridge_health.get("depth_width") or -1) or int(
        bridge_health.get("color_height") or 0
    ) != int(bridge_health.get("depth_height") or -1):
        failures.append("Bridge RGB and D2C depth resolutions are different")

    cache = RgbdFrameCache(
        rgb_name=args.rgb_shm,
        depth_name=args.depth_shm,
        max_frames=args.cache_frames,
        max_age_ms=args.max_age_ms,
        poll_interval_ms=args.poll_interval_ms,
        cache_rgb=not args.depth_only,
    )
    cache.start()
    try:
        if not cache.wait_until_ready(args.ready_timeout_s):
            failures.append(f"RGB-D cache did not become ready: {cache.status().get('last_error')}")
        _request_json(args.runtime_url, "/api/runtime/start_preview", method="POST")
        # Fill a short history before the first inference. This also avoids a
        # pre-existing Runtime preview frame being older than the first cached
        # pair when the validator attaches to an already-running process.
        if args.warmup_ms > 0:
            time.sleep(args.warmup_ms / 1000.0)

        matches: list[dict[str, Any]] = []
        match_wait_ms: list[float] = []
        runtime_total_ms: list[float] = []
        runtime_inference_ms: list[float] = []
        timestamp_deltas: list[int] = []
        saved_files: dict[str, str] | None = None

        for index in range(max(1, args.samples)):
            result = _request_json(args.runtime_url, "/api/runtime/infer_once", method="POST", timeout=15.0)
            timestamp = int(result.get("capture_timestamp_ms") or 0)
            started = time.monotonic()
            frame = cache.get_exact(timestamp, timeout=args.match_timeout_ms / 1000.0)
            wait_ms = (time.monotonic() - started) * 1000.0
            match_wait_ms.append(wait_ms)
            timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
            runtime_total_ms.append(float(timing.get("total_ms") or 0.0))
            runtime_inference_ms.append(float(timing.get("inference_ms") or 0.0))
            if timestamp <= 0:
                failures.append(f"sample {index}: Runtime capture_timestamp_ms is invalid")
                continue
            if frame is None:
                failures.append(f"sample {index}: no exact cached RGB-D frame for timestamp {timestamp}")
                continue
            delta = int(frame.timestamp_epoch_ms) - timestamp
            timestamp_deltas.append(delta)
            if delta != 0:
                failures.append(f"sample {index}: timestamp delta is {delta}ms, expected 0")
            if not frame.aligned_to_color or not frame.calibration_ready:
                failures.append(f"sample {index}: depth is not D2C aligned/calibrated")
            if frame.rgb is not None and frame.rgb.shape[:2] != frame.depth_mm.shape:
                failures.append(f"sample {index}: cached RGB/depth shapes differ")
            if frame.fx <= 0 or frame.fy <= 0:
                failures.append(f"sample {index}: invalid depth intrinsics")
            depth_metrics = _depth_metrics(frame)
            if depth_metrics["valid_ratio"] <= 0.0:
                failures.append(f"sample {index}: depth contains no valid pixels")
            matches.append(
                {
                    "sample": index,
                    "runtime_capture_timestamp_ms": timestamp,
                    "matched_depth_timestamp_ms": frame.timestamp_epoch_ms,
                    "timestamp_delta_ms": delta,
                    "rgb_sequence": frame.rgb_sequence,
                    "depth_sequence": frame.depth_sequence,
                    "width": frame.width,
                    "height": frame.height,
                    "intrinsics": {"fx": frame.fx, "fy": frame.fy, "cx": frame.cx, "cy": frame.cy},
                    "depth": depth_metrics,
                    "match_wait_ms": round(wait_ms, 3),
                }
            )
            if args.save_sample_dir and saved_files is None:
                saved_files = _save_sample(args.save_sample_dir, frame, result)
            if args.interval_ms > 0:
                time.sleep(args.interval_ms / 1000.0)

        cache_status = cache.status()
        if cache_status.get("pair_fps", 0.0) < args.minimum_pair_fps:
            failures.append(
                f"RGB-D cache pair_fps={cache_status.get('pair_fps')} below {args.minimum_pair_fps}"
            )
        if cache_status.get("latest_age_ms") is None or int(cache_status["latest_age_ms"]) > args.max_age_ms:
            failures.append("RGB-D cache latest frame is stale")
        if len(matches) != max(1, args.samples):
            failures.append(f"exact match count {len(matches)}/{max(1, args.samples)}")

        report = {
            "schema_version": "1.0",
            "stage": "M36.3",
            "status": "passed" if not failures else "failed",
            "runtime_url": args.runtime_url,
            "bridge_url": args.bridge_url,
            "model": {
                "model_id": loaded_model.get("model_id"),
                "task_type": loaded_model.get("task_type"),
                "mask_decode_mode": loaded_model.get("mask_decode_mode"),
            },
            "bridge": {
                "shared_rgb_ready": bridge_health.get("shared_rgb_ready"),
                "shared_depth_ready": bridge_health.get("shared_depth_ready"),
                "shared_depth_calibration_ready": bridge_health.get("shared_depth_calibration_ready"),
                "color_resolution": [bridge_health.get("color_width"), bridge_health.get("color_height")],
                "depth_resolution": [bridge_health.get("depth_width"), bridge_health.get("depth_height")],
                "capture_fps_measured": bridge_health.get("capture_fps_measured"),
            },
            "cache": cache_status,
            "samples": max(1, args.samples),
            "exact_matches": len(matches),
            "timestamp_delta_ms": _summary([float(value) for value in timestamp_deltas]),
            "match_wait_ms": _summary(match_wait_ms),
            "runtime_inference_ms": _summary(runtime_inference_ms),
            "runtime_total_ms": _summary(runtime_total_ms),
            "matches": matches,
            "saved_files": saved_files,
            "failures": failures,
        }
        return report, failures
    finally:
        cache.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate M36.3 exact Runtime RGB -> shared D2C depth matching")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:28081")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18182")
    default_config = Path(__file__).resolve().parents[1] / "config" / "line.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--rgb-shm")
    parser.add_argument("--depth-shm")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cache-frames", type=int)
    parser.add_argument("--max-age-ms", type=int)
    parser.add_argument("--poll-interval-ms", type=float)
    parser.add_argument("--ready-timeout-s", type=float, default=5.0)
    parser.add_argument("--warmup-ms", type=int, default=250)
    parser.add_argument("--match-timeout-ms", type=int)
    parser.add_argument("--minimum-pair-fps", type=float, default=20.0)
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--depth-only", action="store_true", help="cache depth only; M36.4 overlays will then be unavailable")
    parser.add_argument("--save-sample-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = load_rgbd_cache_settings(args.config)
        if not settings.enabled:
            raise ValueError(f"online_rgbd.enabled=false in {args.config}")
        args.rgb_shm = args.rgb_shm or settings.rgb_name
        args.depth_shm = args.depth_shm or settings.depth_name
        args.cache_frames = args.cache_frames or settings.cache_frames
        args.max_age_ms = args.max_age_ms or settings.max_age_ms
        args.poll_interval_ms = args.poll_interval_ms or settings.poll_interval_ms
        args.match_timeout_ms = (
            args.match_timeout_ms
            if args.match_timeout_ms is not None
            else settings.exact_match_timeout_ms
        )
        if settings.cache_rgb is False:
            args.depth_only = True
        report, failures = validate(args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if failures:
        print("[FAIL] M36.3 acceptance failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("[PASS] M36.3 exact RGB-D cache matching is healthy.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

