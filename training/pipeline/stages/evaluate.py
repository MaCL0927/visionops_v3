"""Collect metrics produced by Ultralytics training."""

from __future__ import annotations

from typing import Any

from training.pipeline.common import PipelineContext, write_json


def run(ctx: PipelineContext, train_report: dict[str, Any]) -> dict[str, Any]:
    metrics = train_report.get("metrics") if isinstance(train_report.get("metrics"), dict) else {}
    report = {
        "status": "success",
        "source": "ultralytics_results_csv",
        "metrics": metrics,
        "best_pt": train_report.get("best_pt"),
        "run_dir": train_report.get("run_dir"),
    }
    write_json(ctx.output_dir / "evaluate_report.json", report)
    ctx.log(f"[evaluate] metrics_keys={list(metrics.keys())}")
    return report
