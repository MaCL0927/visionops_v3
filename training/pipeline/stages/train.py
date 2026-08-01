"""Run Ultralytics training for a prepared dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, normalize_task, run_command, write_json, yolo_task

AMP_CHECK_MODEL_NAME = "yolo26n.pt"
AMP_CHECK_MODEL_PATH = Path("models/pretrained") / AMP_CHECK_MODEL_NAME


def run(ctx: PipelineContext, preprocess_report: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize_task(str(ctx.job.get("task_type") or preprocess_report.get("task_type") or "detection"))
    yolo_subcommand = yolo_task(task_type)
    data_arg = str(preprocess_report.get("data_path") or preprocess_report.get("data_yaml") or "")
    if not data_arg:
        raise RuntimeError("preprocess_report 缺少 data_path/data_yaml，无法启动训练")

    model = _resolve_pretrained_model(ctx, str(ctx.job.get("pretrained_model") or _default_pretrained_model(task_type)))
    epochs = int(ctx.job.get("epochs", 50))
    batch_size = int(ctx.job.get("batch_size", 16))
    imgsz = int(ctx.job.get("imgsz", 640))
    device = str(ctx.job.get("device") or "")
    amp = bool(ctx.job.get("amp", False))
    workers = int(ctx.job.get("workers", 4))
    yolo_cmd = str(ctx.job.get("yolo_cmd") or "yolo")

    runs_dir = ctx.work_dir / "runs"
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    if amp:
        _prepare_amp_check_model(ctx)
    run_name = f"{yolo_subcommand}_train"
    if task_type == "classification":
        command = _classification_command(
            yolo_cmd=yolo_cmd,
            yolo_subcommand=yolo_subcommand,
            model=model,
            data_arg=data_arg,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            amp=amp,
            workers=workers,
            runs_dir=runs_dir,
            run_name=run_name,
        )
    else:
        command = _yolo_label_command(
            yolo_cmd=yolo_cmd,
            yolo_subcommand=yolo_subcommand,
            model=model,
            data_arg=data_arg,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            amp=amp,
            workers=workers,
            runs_dir=runs_dir,
            run_name=run_name,
        )

    ctx.log(f"[train] start task_type={task_type} yolo_task={yolo_subcommand} model={model} data={data_arg}")
    run_command(command, cwd=ctx.work_dir, log_file=ctx.log_file)

    run_dir = runs_dir / run_name
    best_pt = _find_best_pt(run_dir) or _find_best_pt(runs_dir)
    if best_pt is None:
        raise RuntimeError(f"训练完成但没有找到 weights/best.pt: {runs_dir}")
    last_pt = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    report = {
        "status": "success",
        "task_type": task_type,
        "yolo_task": yolo_subcommand,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt) if last_pt.exists() else "",
        "results_csv": str(results_csv) if results_csv.exists() else "",
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "metrics": _parse_latest_metrics(results_csv),
    }
    write_json(ctx.output_dir / "train_report.json", report)
    ctx.log(f"[train] best_pt={best_pt}")
    return report


def _resolve_pretrained_model(ctx: PipelineContext, value: str) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ctx.project_root / path
    return str(path.resolve())


def _prepare_amp_check_model(ctx: PipelineContext) -> Path:
    """Expose the Ultralytics AMP check model from the canonical repo path."""
    source = (ctx.project_root / AMP_CHECK_MODEL_PATH).resolve()
    if not source.exists():
        raise RuntimeError(
            "AMP 已开启，但缺少 AMP 检查模型: "
            f"{AMP_CHECK_MODEL_PATH}。请把 {AMP_CHECK_MODEL_NAME} 放到该目录，"
            "不要放在仓库根目录。"
        )
    target = ctx.work_dir / AMP_CHECK_MODEL_NAME
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source:
                return target
        except OSError:
            pass
        target.unlink()
    try:
        relative_source = os.path.relpath(source, start=ctx.work_dir)
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)
    return target


def _yolo_label_command(
    *,
    yolo_cmd: str,
    yolo_subcommand: str,
    model: str,
    data_arg: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    device: str,
    amp: bool,
    workers: int,
    runs_dir: Path,
    run_name: str,
) -> list[str]:
    command = [
        yolo_cmd,
        yolo_subcommand,
        "train",
        f"model={model}",
        f"data={data_arg}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch_size}",
        "mosaic=0.0",
        "mixup=0.0",
        "copy_paste=0.0",
        "degrees=0.0",
        "perspective=0.0",
        "translate=0.02",
        "scale=0.5",
        "patience=30",
        f"amp={str(amp)}",
        f"workers={workers}",
        f"project={runs_dir}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    if device:
        command.insert(3, f"device={device}")
    return command


def _classification_command(
    *,
    yolo_cmd: str,
    yolo_subcommand: str,
    model: str,
    data_arg: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    device: str,
    amp: bool,
    workers: int,
    runs_dir: Path,
    run_name: str,
) -> list[str]:
    command = [
        yolo_cmd,
        yolo_subcommand,
        "train",
        f"model={model}",
        f"data={data_arg}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch_size}",
        "patience=30",
        f"amp={str(amp)}",
        f"workers={workers}",
        f"project={runs_dir}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    if device:
        command.insert(3, f"device={device}")
    return command


def _default_pretrained_model(task_type: str) -> str:
    if task_type == "obb":
        return "models/pretrained/yolov8n-obb.pt"
    if task_type == "segmentation":
        return "models/pretrained/yolov8n-seg.pt"
    if task_type == "classification":
        return "models/pretrained/yolov8n-cls.pt"
    return "models/pretrained/yolov8n.pt"


def _find_best_pt(root: Path) -> Path | None:
    candidates = sorted(root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_latest_metrics(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {}
    try:
        lines = [line.strip() for line in results_csv.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        if len(lines) < 2:
            return {}
        headers = [x.strip() for x in lines[0].split(",")]
        values = [x.strip() for x in lines[-1].split(",")]
        out: dict[str, Any] = {}
        for key, value in zip(headers, values):
            try:
                out[key] = float(value)
            except Exception:
                out[key] = value
        return out
    except Exception:
        return {}
