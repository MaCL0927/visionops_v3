#!/usr/bin/env python3
"""VisionOps v3 Runtime infer_once 轻量基准脚本。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


FIELDS = [
    "total_ms",
    "capture_ms",
    "decode_ms",
    "preprocess_ms",
    "inference_ms",
    "rknn_set_input_ms",
    "rknn_run_ms",
    "rknn_get_output_ms",
    "postprocess_ms",
    "result_build_ms",
]


def _request_json(url: str, *, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _pick_value(payload: dict[str, Any], field: str) -> float | None:
    timing = payload.get("timing")
    if isinstance(timing, dict) and field in timing:
        try:
            return float(timing[field])
        except (TypeError, ValueError):
            return None
    timing_detail = payload.get("timing_detail")
    if isinstance(timing_detail, dict) and field in timing_detail:
        try:
            return float(timing_detail[field])
        except (TypeError, ValueError):
            return None
    return None


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "avg": round(statistics.fmean(ordered), 4),
        "p50": round(statistics.median(ordered), 4),
        "p90": round(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def run(runtime_url: str, count: int, warmup: int) -> dict[str, Any]:
    runtime_url = runtime_url.rstrip("/")
    results: list[dict[str, Any]] = []

    for index in range(warmup + count):
        started = time.perf_counter()
        payload = _request_json(f"{runtime_url}/api/runtime/infer_once", method="POST")
        request_ms = (time.perf_counter() - started) * 1000.0
        if index >= warmup:
            results.append({
                "request_index": index - warmup + 1,
                "http_roundtrip_ms": round(request_ms, 4),
                "result_id": payload.get("result_id"),
                "frame_id": payload.get("frame_id"),
                "timing": payload.get("timing", {}),
                "timing_detail": payload.get("timing_detail", {}),
            })

    aggregates: dict[str, Any] = {
        "http_roundtrip_ms": _summary([float(item["http_roundtrip_ms"]) for item in results]),
    }
    for field in FIELDS:
        values = []
        for item in results:
            value = _pick_value(item, field)
            if value is not None:
                values.append(value)
        summary = _summary(values)
        if summary is not None:
            aggregates[field] = summary

    return {
        "schema_version": "1.0",
        "message_type": "runtime_benchmark",
        "runtime_url": runtime_url,
        "warmup": warmup,
        "count": count,
        "aggregates": aggregates,
        "samples": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionOps Runtime infer_once benchmark")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:18080")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, help="可选输出 JSON 文件")
    args = parser.parse_args()

    payload = run(args.runtime_url, max(1, args.count), max(0, args.warmup))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
