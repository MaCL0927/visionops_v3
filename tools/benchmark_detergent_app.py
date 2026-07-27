#!/usr/bin/env python3
"""Poll detergent App status and print pipeline/timing summaries."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any, Dict


def get_json(url: str, timeout: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("status response must be a JSON object")
    return value


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark detergent App pipeline status")
    parser.add_argument("--app-url", default="http://127.0.0.1:19212")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    url = args.app_url.rstrip("/") + "/api/app/status"
    samples = []
    print(
        " out_fps prod_fps app_ms p50_ms p95_ms | "
        "transport rt_req hdr_ms json_ms post_ms q_ms ws_ms | q drop stale"
    )
    for _ in range(max(1, args.count)):
        try:
            status = get_json(url, args.timeout)
            last = status.get("last_app_timing") if isinstance(status.get("last_app_timing"), dict) else {}
            latency = status.get("latency_ms") if isinstance(status.get("latency_ms"), dict) else {}
            pipeline = status.get("pipeline") if isinstance(status.get("pipeline"), dict) else {}
            counters = status.get("counters") if isinstance(status.get("counters"), dict) else {}
            sample = {
                "timestamp": time.time(),
                "detection_fps": number(status.get("detection_fps")),
                "inference_stage_fps": number(status.get("inference_stage_fps")),
                "last_latency_ms": number(status.get("last_latency_ms")),
                "latency_p50_ms": number(latency.get("p50")),
                "latency_p95_ms": number(latency.get("p95")),
                "runtime_transport": str(last.get("runtime_transport") or "unknown"),
                "runtime_request_ms": number(last.get("runtime_request_ms")),
                "runtime_headers_wait_ms": number(last.get("runtime_headers_wait_ms")),
                "runtime_json_decode_ms": number(last.get("runtime_json_decode_ms")),
                "postprocess_stage_ms": number(last.get("postprocess_stage_ms")),
                "result_queue_wait_ms": number(last.get("result_queue_wait_ms")),
                "websocket_send_ms": number(last.get("websocket_send_ms")),
                "result_queue_size": int(pipeline.get("result_queue_size") or 0),
                "pipeline_results_dropped": int(counters.get("pipeline_results_dropped") or 0),
                "pipeline_stale_results_dropped": int(counters.get("pipeline_stale_results_dropped") or 0),
            }
            samples.append(sample)
            print(
                f"{sample['detection_fps']:8.2f} {sample['inference_stage_fps']:8.2f} "
                f"{sample['last_latency_ms']:6.1f} {sample['latency_p50_ms']:6.1f} "
                f"{sample['latency_p95_ms']:6.1f} | "
                f"{sample['runtime_transport'][:9]:>9s} "
                f"{sample['runtime_request_ms']:6.1f} "
                f"{sample['runtime_headers_wait_ms']:6.1f} "
                f"{sample['runtime_json_decode_ms']:7.2f} "
                f"{sample['postprocess_stage_ms']:7.1f} "
                f"{sample['result_queue_wait_ms']:4.1f} {sample['websocket_send_ms']:5.1f} | "
                f"{sample['result_queue_size']:1d} {sample['pipeline_results_dropped']:4d} "
                f"{sample['pipeline_stale_results_dropped']:5d}"
            )
        except Exception as error:
            print("ERROR", error)
        time.sleep(max(0.05, args.interval))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": "1.0", "samples": samples}, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
