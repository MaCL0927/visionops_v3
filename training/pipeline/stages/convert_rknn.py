"""Convert ONNX model to RKNN for Rockchip devices.

The main training pipeline normally runs in the `visionops` environment.  RKNN
conversion, however, must run in the `rknn311` environment where
rknn-toolkit2 is installed.  Therefore this stage delegates the actual
conversion to a small worker process via `conda run -n rknn311 ...` by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, conda_run_prefix, read_json, run_command, write_json


def run(ctx: PipelineContext, export_report: dict[str, Any], preprocess_report: dict[str, Any]) -> dict[str, Any]:
    onnx_path = Path(str(export_report.get("onnx_path") or ""))
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在，无法转换 RKNN: {onnx_path}")

    target_platform = str(ctx.job.get("target_platform") or "rk3576")
    do_quantization = bool(ctx.job.get("do_quantization", False))
    rknn_path = ctx.output_dir / "model.rknn"
    report_path = ctx.output_dir / "convert_rknn_report.json"
    dataset_txt = _make_quant_dataset(ctx, preprocess_report) if do_quantization else ""
    env_name = str(ctx.job.get("rknn_conda_env") or "")

    payload = {
        "onnx_path": str(onnx_path),
        "rknn_path": str(rknn_path),
        "target_platform": target_platform,
        "do_quantization": do_quantization,
        "dataset_txt": dataset_txt or "",
        "report_path": str(report_path),
    }
    worker_config = ctx.output_dir / "convert_rknn_worker_config.json"
    write_json(worker_config, payload)

    ctx.log(f"[convert_rknn] onnx={onnx_path} target={target_platform} quant={do_quantization}")
    ctx.log(f"[convert_rknn] env={env_name or 'current'}")

    if env_name and env_name.lower() not in {"current", "none", "base", "false", "0"}:
        command = conda_run_prefix(ctx, env_name) + [
            "python",
            "-m",
            "training.pipeline.stages.convert_rknn_worker",
            "--config",
            str(worker_config),
        ]
        run_command(command, cwd=ctx.project_root, log_file=ctx.log_file)
        report = read_json(report_path, {}) or {}
    else:
        # Useful for unit tests or machines where the parent process itself is
        # already running inside rknn311.
        from training.pipeline.stages.convert_rknn_worker import convert_with_rknn

        report = convert_with_rknn(payload)
        write_json(report_path, report)

    if report.get("status") != "success":
        raise RuntimeError(f"RKNN 转换失败: {report}")
    if not Path(str(report.get("rknn_path") or rknn_path)).exists():
        raise FileNotFoundError(f"RKNN 转换报告成功但未找到文件: {report.get('rknn_path') or rknn_path}")
    ctx.log(f"[convert_rknn] rknn_path={report.get('rknn_path')}")
    return report


def _make_quant_dataset(ctx: PipelineContext, preprocess_report: dict[str, Any]) -> str:
    dataset_dir = Path(str(preprocess_report.get("dataset_dir") or ""))
    task_type = str(preprocess_report.get("task_type") or ctx.job.get("task_type") or "detection").lower()
    images: list[Path] = []
    if task_type in {"classification", "cls", "classify"}:
        train_root = dataset_dir / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(f"classification 量化数据目录不存在: {train_root}")
        images = sorted(p for p in train_root.rglob("*") if p.is_file())
    else:
        images_dir = dataset_dir / "images" / "train"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"量化数据目录不存在: {images_dir}")
        images = sorted(p for p in images_dir.iterdir() if p.is_file())
    images = images[: int(ctx.job.get("quant_image_count", 128))]
    if not images:
        raise RuntimeError("启用量化但没有可用图片")
    dataset_txt = ctx.output_dir / "rknn_quant_dataset.txt"
    dataset_txt.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    return str(dataset_txt)
