#!/usr/bin/env python3
"""M36.2 live RGB transport and RKNN segmentation acceptance check.

This stage intentionally validates RGB only. It does not read depth or execute
foam-ring geometry. A successful report proves that Runtime is consuming fresh
Orbbec RGB frames through POSIX shared memory and returning segmentation masks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(base_url: str, path: str, *, method: str = "GET", timeout: float = 5.0) -> dict[str, Any]:
    request = Request(base_url.rstrip("/") + path, method=method, data=b"" if method != "GET" else None)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local device endpoint
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


def _timing_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def validate(
    runtime_url: str,
    bridge_url: str,
    samples: int,
    interval_ms: int,
    allow_http_fallback: bool,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    bridge_health = _request_json(bridge_url, "/health")
    initial_status = _request_json(runtime_url, "/api/runtime/status")
    _request_json(runtime_url, "/api/runtime/start_preview", method="POST")

    status: dict[str, Any] = {}
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        status = _request_json(runtime_url, "/api/runtime/status")
        source = status.get("frame_source") if isinstance(status.get("frame_source"), dict) else {}
        if source.get("thread_alive") and source.get("shared_memory_sequence", 0) > 0 and not source.get("stale"):
            break
        time.sleep(0.1)

    source = status.get("frame_source") if isinstance(status.get("frame_source"), dict) else {}
    loaded_model = status.get("loaded_model") if isinstance(status.get("loaded_model"), dict) else {}
    if loaded_model.get("task_type") != "segmentation":
        failures.append(f"Runtime task_type is {loaded_model.get('task_type')!r}, expected 'segmentation'")
    if source.get("type") != "shared_memory":
        failures.append(f"frame_source.type is {source.get('type')!r}, expected 'shared_memory'")
    if source.get("configured_transport") != "posix_shared_memory":
        failures.append("configured_transport is not posix_shared_memory")
    if source.get("transport") != "posix_shared_memory" and not allow_http_fallback:
        failures.append(f"active transport is {source.get('transport')!r}; HTTP fallback is not allowed")
        shared_error = source.get("shared_memory_last_error")
        if shared_error:
            failures.append(f"shared-memory reader error: {shared_error}")
        shm_name = str(source.get("device") or "/visionops_orbbec336l_rgb")
        shm_path = Path("/dev/shm") / shm_name.lstrip("/")
        if shm_path.exists() and not os.access(shm_path, os.R_OK):
            failures.append(
                f"current user cannot read {shm_path}; fix Bridge shared-memory mode/ownership"
            )
    if source.get("stale"):
        failures.append("RGB frame source is stale")
    if not source.get("thread_alive"):
        failures.append("RGB capture thread is not alive")
    if not source.get("opened"):
        failures.append("RGB frame source is not opened")

    results: list[dict[str, Any]] = []
    totals: list[float] = []
    inference: list[float] = []
    postprocess: list[float] = []
    capture: list[float] = []
    decode: list[float] = []
    timestamps: list[int] = []
    proto_masks = 0
    fallback_masks = 0
    errors = 0

    for _ in range(max(1, samples)):
        result = _request_json(runtime_url, "/api/runtime/infer_once", method="POST", timeout=15.0)
        results.append(result)
        if result.get("status") != "ok":
            errors += 1
        timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
        totals.append(float(timing.get("total_ms") or 0.0))
        inference.append(float(timing.get("inference_ms") or 0.0))
        postprocess.append(float(timing.get("postprocess_ms") or 0.0))
        capture.append(float(timing.get("capture_ms") or 0.0))
        decode.append(float(timing.get("decode_ms") or 0.0))
        timestamps.append(int(result.get("capture_timestamp_ms") or 0))
        detections = result.get("detections") if isinstance(result.get("detections"), list) else []
        for detection in detections:
            mask = detection.get("mask") if isinstance(detection, dict) and isinstance(detection.get("mask"), dict) else {}
            if mask.get("source") == "proto":
                proto_masks += 1
            elif mask:
                fallback_masks += 1
        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)

    final_status = _request_json(runtime_url, "/api/runtime/status")
    final_source = final_status.get("frame_source") if isinstance(final_status.get("frame_source"), dict) else {}

    positive_timestamps = [value for value in timestamps if value > 0]
    if errors:
        failures.append(f"{errors}/{len(results)} inference calls returned non-ok status")
    if not positive_timestamps:
        failures.append("inference results have no valid capture_timestamp_ms")
    elif len(set(positive_timestamps)) < 2 and len(results) > 1:
        failures.append("all inference calls used the same RGB timestamp; live frame refresh is not proven")
    if fallback_masks:
        failures.append(f"{fallback_masks} masks were not generated from proto")
    if final_source.get("transport") != "posix_shared_memory" and not allow_http_fallback:
        failures.append(f"final active transport is {final_source.get('transport')!r}")
    if final_source.get("shared_memory_sequence", 0) <= source.get("shared_memory_sequence", 0):
        failures.append("shared_memory_sequence did not advance during validation")

    report = {
        "schema_version": "1.0",
        "stage": "M36.2",
        "status": "passed" if not failures else "failed",
        "runtime_url": runtime_url,
        "bridge_url": bridge_url,
        "bridge_health": bridge_health,
        "model": {
            "model_id": loaded_model.get("model_id"),
            "task_type": loaded_model.get("task_type"),
            "mask_decode_mode": loaded_model.get("mask_decode_mode"),
        },
        "frame_source_initial": source,
        "frame_source_final": final_source,
        "samples": len(results),
        "unique_capture_timestamps": len(set(positive_timestamps)),
        "proto_mask_count": proto_masks,
        "other_mask_count": fallback_masks,
        "timing_ms": {
            "capture": _timing_summary(capture),
            "decode": _timing_summary(decode),
            "inference": _timing_summary(inference),
            "postprocess": _timing_summary(postprocess),
            "total": _timing_summary(totals),
        },
        "failures": failures,
        "initial_runtime_status": initial_status,
    }
    return report, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate M36.2 Orbbec shared-RGB RKNN runtime")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:28081")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18182")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--allow-http-fallback", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report, failures = validate(
            args.runtime_url,
            args.bridge_url,
            args.samples,
            args.interval_ms,
            args.allow_http_fallback,
        )
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if failures:
        print("[FAIL] M36.2 acceptance failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("[PASS] M36.2 shared-memory RGB inference path is healthy.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
