"""Export trained Ultralytics weights to Rockchip-compatible ONNX.

The parent service normally runs in the `visionops` environment, while this
stage runs `yolo ... export` in the `pt2onnx` conda environment by default.
For Rockchip modified Ultralytics, `format=rknn` is required even though the
artifact we consume next is still an ONNX file; this produces split-head outputs
for detection/OBB/segmentation that match the v3 edge Runtime postprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, conda_run_prefix, normalize_task, run_command, write_json, yolo_task


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
    export_format = str(ctx.job.get("onnx_export_format") or ctx.job.get("export_format") or "rknn").strip() or "rknn"

    command = conda_run_prefix(ctx, env_name) + [
        yolo_cmd,
        yolo_task(task_type),
        "export",
        f"model={best_pt}",
        f"format={export_format}",
        f"imgsz={imgsz}",
        f"opset={opset}",
        f"simplify={str(simplify)}",
    ]

    ctx.log(f"[export_onnx] task_type={task_type} yolo_task={yolo_task(task_type)} best_pt={best_pt}")
    ctx.log(f"[export_onnx] env={env_name or 'current'} format={export_format} imgsz={imgsz}")
    ctx.log("[export_onnx] Rockchip 多头导出必须使用 format=rknn；产物仍由后续 RKNN stage 读取 .onnx")
    run_command(command, cwd=ctx.project_root, log_file=ctx.log_file)

    onnx_path = _find_exported_onnx(best_pt, ctx)
    if onnx_path is None:
        raise FileNotFoundError(f"ONNX 导出完成但没有找到 .onnx 文件: {best_pt.parent}")

    report = {
        "status": "success",
        "task_type": task_type,
        "yolo_task": yolo_task(task_type),
        "onnx_path": str(onnx_path),
        "best_pt": str(best_pt),
        "imgsz": imgsz,
        "opset": opset,
        "simplify": simplify,
        "export_format": export_format,
        "onnx_conda_env": env_name or "current",
    }
    write_json(ctx.output_dir / "export_onnx_report.json", report)
    ctx.log(f"[export_onnx] onnx_path={onnx_path}")
    return report


def _find_exported_onnx(best_pt: Path, ctx: PipelineContext) -> Path | None:
    direct = best_pt.with_suffix(".onnx")
    if direct.exists():
        return direct

    roots = [best_pt.parent, best_pt.parent.parent, ctx.work_dir, ctx.output_dir]
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
