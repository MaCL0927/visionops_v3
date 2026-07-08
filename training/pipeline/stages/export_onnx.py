"""Export trained Ultralytics weights to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, conda_run_prefix, run_command, write_json, yolo_task, normalize_task


def run(ctx: PipelineContext, train_report: dict[str, Any]) -> dict[str, Any]:
    best_pt = Path(str(train_report.get("best_pt") or ""))
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt 不存在，无法导出 ONNX: {best_pt}")
    imgsz = int(ctx.job.get("imgsz", 640))
    opset = int(ctx.job.get("onnx_opset", 12))
    simplify = bool(ctx.job.get("onnx_simplify", True))
    yolo_cmd = str(ctx.job.get("yolo_cmd") or "yolo")
    task_type = normalize_task(str(ctx.job.get("task_type") or train_report.get("task_type") or "detection"))
    env_name = str(ctx.job.get("onnx_conda_env") or "")
    command = conda_run_prefix(ctx, env_name) + [
        yolo_cmd,
        yolo_task(task_type),
        "export",
        f"model={best_pt}",
        "format=onnx",
        f"imgsz={imgsz}",
        f"opset={opset}",
        f"simplify={str(simplify)}",
    ]
    ctx.log(f"[export_onnx] best_pt={best_pt}")
    ctx.log(f"[export_onnx] env={env_name or 'current'} note=ONNX 导出使用 pt2onnx 环境时会调用该环境中的瑞芯微修改版 Ultralytics")
    run_command(command, cwd=ctx.project_root, log_file=ctx.log_file)
    onnx_path = best_pt.with_suffix(".onnx")
    if not onnx_path.exists():
        # Ultralytics normally exports beside best.pt; search nearby as fallback.
        candidates = sorted(best_pt.parent.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            onnx_path = candidates[0]
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 导出完成但没有找到 .onnx 文件: {best_pt.parent}")
    report = {
        "status": "success",
        "onnx_path": str(onnx_path),
        "best_pt": str(best_pt),
        "imgsz": imgsz,
        "opset": opset,
        "simplify": simplify,
        "onnx_conda_env": env_name or "current",
    }
    write_json(ctx.output_dir / "export_onnx_report.json", report)
    ctx.log(f"[export_onnx] onnx_path={onnx_path}")
    return report
