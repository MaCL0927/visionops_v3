#!/usr/bin/env python3
"""VisionOps v3 training pipeline.

This runner keeps the v3 service-side boundary clean:
accepted batch -> materialized YOLO dataset -> train -> evaluate -> ONNX export
-> RKNN conversion -> v3 model package (model.rknn + model.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, read_json, write_json
from training.pipeline.stages import convert_rknn, evaluate, export_onnx, package_v3_model, preprocess, train

STAGES = ["preprocess", "train", "evaluate", "export_onnx", "convert_rknn", "package_v3_model"]


def _status_path(job_dir: Path) -> Path:
    return job_dir / "pipeline_status.json"


def _write_status(job_dir: Path, **updates: Any) -> None:
    path = _status_path(job_dir)
    status = read_json(path, {}) or {}
    status.update(updates)
    status["updated_at_ms"] = int(time.time() * 1000)
    write_json(path, status)


def run(job_config_path: Path, output_dir: Path, project_root: Path | None = None) -> dict[str, Any]:
    job = read_json(job_config_path, None)
    if not isinstance(job, dict):
        raise ValueError(f"job config 不是合法 JSON 对象: {job_config_path}")
    dataset_path = Path(str(job.get("dataset_json") or ""))
    dataset = read_json(dataset_path, None)
    if not isinstance(dataset, dict):
        raise ValueError(f"dataset_json 不是合法 JSON 对象: {dataset_path}")
    batches_path = Path(str(job.get("dataset_batches_json") or dataset_path.parent / "batches.json"))
    batches = read_json(batches_path, [])
    if isinstance(batches, list):
        dataset["batches"] = batches

    root = project_root or Path(__file__).resolve().parents[2]
    job_dir = Path(str(job.get("job_path") or job_config_path.parent)).resolve()
    work_dir = Path(str(job.get("work_dir") or job_dir / "work")).resolve()
    output_dir = output_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(str(job.get("log_path") or job_dir / "job.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    with log_path.open("a", encoding="utf-8") as log_file:
        ctx = PipelineContext(project_root=root, job=job, dataset=dataset, job_dir=job_dir, work_dir=work_dir, output_dir=output_dir, log_file=log_file)
        ctx.log(f"[pipeline] start job_id={job.get('job_id')} dataset={dataset.get('dataset_id')}")
        _write_status(job_dir, status="running", current_stage="preprocess", stages=STAGES)
        reports["preprocess"] = preprocess.run(ctx)
        _write_status(job_dir, current_stage="train")
        reports["train"] = train.run(ctx, reports["preprocess"])
        _write_status(job_dir, current_stage="evaluate")
        reports["evaluate"] = evaluate.run(ctx, reports["train"])
        _write_status(job_dir, current_stage="export_onnx")
        reports["export_onnx"] = export_onnx.run(ctx, reports["train"])
        _write_status(job_dir, current_stage="convert_rknn")
        reports["convert_rknn"] = convert_rknn.run(ctx, reports["export_onnx"], reports["preprocess"])
        _write_status(job_dir, current_stage="package_v3_model")
        reports["package_v3_model"] = package_v3_model.run(
            ctx,
            reports["preprocess"],
            reports["train"],
            reports["evaluate"],
            reports["export_onnx"],
            reports["convert_rknn"],
        )
        _write_status(job_dir, status="success", current_stage="done", output_model_package=reports["package_v3_model"].get("model_id"))
        ctx.log("[pipeline] success")

    final_report = {"status": "success", "job_id": job.get("job_id"), "reports": reports}
    write_json(output_dir / "pipeline_report.json", final_report)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionOps v3 training pipeline")
    parser.add_argument("--job-config", required=True, help="训练任务 JSON 配置路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--project-root", default="", help="仓库根目录，默认自动解析")
    args = parser.parse_args()
    try:
        report = run(Path(args.job_config), Path(args.output_dir), Path(args.project_root).resolve() if args.project_root else None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        # The parent service reads pipeline_status.json and logs; still print a
        # clear error for manual runs.
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
