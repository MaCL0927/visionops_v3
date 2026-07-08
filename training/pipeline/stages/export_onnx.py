"""Export trained Ultralytics weights to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, conda_run_prefix, run_command, write_json, yolo_task, normalize_task




def _find_exported_onnx(best_pt: Path, work_dir: Path, output_dir: Path) -> Path:
    """Locate ONNX produced by Ultralytics/Rockchip export.

    普通 Ultralytics 和瑞芯微修改版 Ultralytics 的导出位置都可能在
    best.pt 同级目录；少数版本会在 run/work 目录下再创建子目录。
    优先选择 best.pt 同名 ONNX，其次选择最近修改的 ONNX。
    """
    expected = best_pt.with_suffix(".onnx")
    if expected.exists():
        return expected

    roots = [best_pt.parent, best_pt.parent.parent, work_dir, output_dir]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)
        candidates.extend(root.glob("*.onnx"))
        candidates.extend(root.rglob("*.onnx"))

    unique = []
    seen_files: set[Path] = set()
    for item in candidates:
        try:
            key = item.resolve()
        except Exception:
            key = item
        if key in seen_files or not item.is_file():
            continue
        seen_files.add(key)
        unique.append(item)

    if not unique:
        return expected
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique[0]

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
    # Rockchip 修改版 Ultralytics 的多头 ONNX 导出不是普通 format=onnx，
    # 而是通过 format=rknn 触发。这里仍然把本 stage 命名为 export_onnx，
    # 因为它的产物仍是供 rknn-toolkit2 转换使用的 .onnx 文件。
    export_format = str(ctx.job.get("onnx_export_format") or "rknn").strip() or "rknn"

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
    ctx.log(f"[export_onnx] best_pt={best_pt}")
    ctx.log(f"[export_onnx] env={env_name or 'current'}")
    ctx.log(f"[export_onnx] export_format={export_format} note=Rockchip 多头 ONNX 需要使用 format=rknn 触发")
    run_command(command, cwd=ctx.project_root, log_file=ctx.log_file)

    onnx_path = _find_exported_onnx(best_pt, ctx.work_dir, ctx.output_dir)
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
        "export_format": export_format,
    }
    write_json(ctx.output_dir / "export_onnx_report.json", report)
    ctx.log(f"[export_onnx] onnx_path={onnx_path}")
    return report
