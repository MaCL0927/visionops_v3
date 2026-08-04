#!/usr/bin/env python3
"""M36.5 acceptance test for the persistent foam-ring trigger service."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = int(index)
    high = min(len(ordered) - 1, low + 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    document: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes, Mapping[str, str]]:
    body = None
    headers = {"Accept": "application/json,image/jpeg"}
    if document is not None:
        body = json.dumps(document, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                int(getattr(response, "status", 200)),
                response.read(),
                {str(k).lower(): str(v) for k, v in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        return (
            int(error.code),
            error.read(),
            {str(k).lower(): str(v) for k, v in error.headers.items()},
        )


def _json_request(*args: Any, **kwargs: Any) -> tuple[int, Dict[str, Any]]:
    code, body, _headers = _request(*args, **kwargs)
    try:
        document = json.loads(body.decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"响应不是有效JSON: HTTP {code}, {body[:200]!r}") from error
    if not isinstance(document, dict):
        raise RuntimeError("响应JSON顶层不是对象")
    return code, document


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify M36.5 persistent trigger service")
    parser.add_argument("--service-url", default="http://127.0.0.1:19213")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--report", type=Path, default=Path("/tmp/m36_5_report.json"))
    parser.add_argument("--save-debug-once", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    code, health = _json_request(args.service_url, "/health", timeout=5.0)
    if code != 200 or health.get("health") != "ok":
        failures.append(f"service health is not ok: HTTP {code}, {health}")
    code, initial_status = _json_request(args.service_url, "/status", timeout=5.0)
    if code != 200:
        failures.append(f"status HTTP {code}")

    results: list[Dict[str, Any]] = []
    elapsed_values: list[float] = []
    prefix = f"m36-5-{int(time.time() * 1000)}"
    for index in range(max(1, int(args.samples))):
        request_id = f"{prefix}-{index:03d}"
        started = time.perf_counter()
        code, result = _json_request(
            args.service_url,
            "/api/foam_ring/infer_once",
            method="POST",
            document={
                "request_id": request_id,
                "wait": True,
                "timeout_ms": int(args.timeout_s * 1000),
                "save_debug": bool(args.save_debug_once and index == 0),
            },
            timeout=args.timeout_s + 5.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        elapsed_values.append(elapsed_ms)
        results.append(result)
        if code != 200 or result.get("status") != "ok":
            failures.append(f"trigger {request_id} failed: HTTP {code}, {result}")
            continue
        if result.get("request_id") != request_id:
            failures.append(f"request_id mismatch: expected {request_id}, got {result.get('request_id')}")
        if result.get("rgbd_timestamp_delta_ms") != 0:
            failures.append(f"{request_id} exact RGB-D delta is not zero")
        if result.get("robot_ready") is not False:
            failures.append(f"{request_id} robot_ready must remain false in M36.5")
        if not (args.save_debug_once and index == 0) and result.get("files"):
            failures.append(f"{request_id} production trigger unexpectedly wrote debug files")

    duplicate: Dict[str, Any] = {}
    if results and results[0].get("status") == "ok":
        first_id = str(results[0]["request_id"])
        code, duplicate = _json_request(
            args.service_url,
            "/api/foam_ring/infer_once",
            method="POST",
            document={"request_id": first_id, "wait": True},
            timeout=5.0,
        )
        if code != 200:
            failures.append(f"duplicate request HTTP {code}")
        if not duplicate.get("idempotent_replay"):
            failures.append("duplicate request_id was not marked as idempotent replay")
        if duplicate.get("capture_timestamp_ms") != results[0].get("capture_timestamp_ms"):
            failures.append("duplicate request_id re-ran inference instead of replaying result")

    snapshot_status, snapshot_body, snapshot_headers = _request(
        args.service_url,
        "/snapshot.jpg",
        timeout=5.0,
    )
    if snapshot_status != 200:
        failures.append(f"snapshot.jpg unavailable: HTTP {snapshot_status}")
    elif not snapshot_headers.get("content-type", "").startswith("image/jpeg"):
        failures.append("snapshot.jpg content-type is not image/jpeg")
    elif len(snapshot_body) < 100:
        failures.append("snapshot.jpg body is unexpectedly small")

    code, final_status = _json_request(args.service_url, "/status", timeout=5.0)
    if code != 200:
        failures.append(f"final status HTTP {code}")
    processor = final_status.get("processor") if isinstance(final_status.get("processor"), dict) else {}
    cache = processor.get("cache") if isinstance(processor.get("cache"), dict) else {}
    if cache.get("last_error"):
        failures.append(f"RGB-D cache error: {cache.get('last_error')}")
    if float(cache.get("pair_fps") or 0.0) < 20.0:
        failures.append(f"RGB-D cache FPS too low: {cache.get('pair_fps')}")

    report = {
        "schema_version": "1.0",
        "stage": "M36.5",
        "status": "passed" if not failures else "failed",
        "service_url": args.service_url,
        "samples": len(results),
        "target_found_count": sum(bool(row.get("target_found")) for row in results),
        "exact_match_count": sum(row.get("rgbd_timestamp_delta_ms") == 0 for row in results),
        "latency_ms": {
            "mean": round(statistics.fmean(elapsed_values), 3) if elapsed_values else 0.0,
            "p50": round(_percentile(elapsed_values, 0.50), 3),
            "p95": round(_percentile(elapsed_values, 0.95), 3),
            "max": round(max(elapsed_values), 3) if elapsed_values else 0.0,
        },
        "initial_health": health,
        "initial_status": initial_status,
        "results": results,
        "duplicate_replay": duplicate,
        "final_status": final_status,
        "snapshot": {
            "http_status": snapshot_status,
            "bytes": len(snapshot_body),
            "capture_timestamp_ms": snapshot_headers.get("x-visionops-capture-timestamp-ms"),
        },
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "stage": report["stage"],
        "status": report["status"],
        "samples": report["samples"],
        "target_found_count": report["target_found_count"],
        "exact_match_count": report["exact_match_count"],
        "latency_ms": report["latency_ms"],
        "report": str(args.report),
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    if failures:
        print("[FAIL] M36.5 acceptance failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    print("[PASS] M36.5 persistent trigger service is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
