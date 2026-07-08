"""Create v3 standard model package from training artifacts."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, write_json


def run(
    ctx: PipelineContext,
    preprocess_report: dict[str, Any],
    train_report: dict[str, Any],
    evaluate_report: dict[str, Any],
    export_report: dict[str, Any],
    rknn_report: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(ctx.job.get("model_id") or _default_model_id(ctx))[:96]
    package_root = Path(str(ctx.job.get("model_packages_root") or ctx.project_root / "server_data" / "model_packages"))
    package_dir = package_root / _safe_id(model_id)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "logs").mkdir(parents=True, exist_ok=True)

    rknn_src = Path(str(rknn_report.get("rknn_path") or ""))
    if not rknn_src.exists():
        raise FileNotFoundError(f"RKNN 文件不存在，无法生成模型包: {rknn_src}")
    shutil.copy2(rknn_src, package_dir / "model.rknn")

    classes = preprocess_report.get("classes") if isinstance(preprocess_report.get("classes"), list) else ["object"]
    version = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(ctx.job.get("model_name") or f"{ctx.job.get('task_type', 'detection')}-{ctx.dataset.get('dataset_id', 'dataset')}")
    model_yaml = _make_model_yaml(
        model_id=model_id,
        model_name=model_name,
        version=version,
        task_type=str(ctx.job.get("task_type") or ctx.dataset.get("task_type") or "detection"),
        classes=[str(x) for x in classes],
        input_size=[int(ctx.job.get("imgsz", 640)), int(ctx.job.get("imgsz", 640))],
        target_platform=str(ctx.job.get("target_platform") or "rk3576"),
        conf_threshold=float(ctx.job.get("conf_threshold", 0.25)),
        iou_threshold=float(ctx.job.get("iou_threshold", 0.45)),
        max_det=int(ctx.job.get("max_det", 100)),
    )
    _write_runtime_model_yaml(package_dir / "model.yaml", model_yaml)

    now = int(time.time() * 1000)
    package_meta = {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": model_name,
        "version": version,
        "task_type": str(ctx.job.get("task_type") or ctx.dataset.get("task_type") or "detection"),
        "target_platform": str(ctx.job.get("target_platform") or "rk3576"),
        "dataset_id": ctx.dataset.get("dataset_id"),
        "job_id": ctx.job.get("job_id"),
        "created_at_ms": now,
        "updated_at_ms": now,
        "artifacts": {
            "best_pt": train_report.get("best_pt"),
            "onnx_path": export_report.get("onnx_path"),
            "source_rknn_path": rknn_report.get("rknn_path"),
        },
    }
    write_json(package_dir / "package.json", package_meta)
    write_json(package_dir / "metrics.json", evaluate_report.get("metrics") or {})
    write_json(package_dir / "train_config.yaml.json", ctx.job)
    write_json(package_dir / "export_report.json", {"onnx": export_report, "rknn": rknn_report})

    # Copy stage reports and job log for traceability.
    for report in ctx.output_dir.glob("*_report.json"):
        shutil.copy2(report, package_dir / "logs" / report.name)
    job_log = ctx.job_dir / "job.log"
    if job_log.exists():
        shutil.copy2(job_log, package_dir / "logs" / "job.log")

    report = {
        "status": "success",
        "model_id": model_id,
        "package_dir": str(package_dir),
        "model_yaml": str(package_dir / "model.yaml"),
        "model_rknn": str(package_dir / "model.rknn"),
    }
    write_json(ctx.output_dir / "package_v3_model_report.json", report)
    ctx.log(f"[package] model_id={model_id} package_dir={package_dir}")
    return report


def _yaml_quote(value: Any) -> str:
    text = str(value)
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return _yaml_quote(value)


def _write_runtime_model_yaml(path: Path, doc: dict[str, Any]) -> None:
    """Write the model.yaml contract consumed by edge/runtime_cpp.

    PyYAML emits anchors when the same list object is reused at top level and
    under model.input_size, e.g. ``input_size: &id001``.  The C++ runtime parser
    intentionally supports only a small, deployment-oriented YAML subset, so
    anchors make input_size fail to parse.  This writer keeps the file stable and
    anchor-free, with ``input_size: [640, 640]`` at both required locations.
    """
    input_size = doc.get("input_size") or [640, 640]
    width = int(input_size[0])
    height = int(input_size[1])
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    post = doc.get("postprocess") if isinstance(doc.get("postprocess"), dict) else {}
    runtime = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
    classes = doc.get("classes") if isinstance(doc.get("classes"), list) else []
    class_names = doc.get("class_names") if isinstance(doc.get("class_names"), list) else [
        str(item.get("name")) for item in classes if isinstance(item, dict) and item.get("name") is not None
    ]
    if not classes:
        classes = [{"id": i, "name": name} for i, name in enumerate(class_names or ["object"])]
    if not class_names:
        class_names = [str(item.get("name", f"class_{i}")) for i, item in enumerate(classes) if isinstance(item, dict)]

    lines: list[str] = []
    lines.append(f"schema_version: {_yaml_scalar(doc.get('schema_version', '1.0'))}")
    lines.append(f"model_id: {_yaml_scalar(doc.get('model_id', ''))}")
    lines.append(f"model_name: {_yaml_scalar(doc.get('model_name', ''))}")
    lines.append(f"model_version: {_yaml_scalar(doc.get('model_version', ''))}")
    lines.append(f"task_type: {_yaml_scalar(doc.get('task_type', 'detection'))}")
    lines.append(f"target_platform: {_yaml_scalar(doc.get('target_platform', 'rk3576'))}")
    lines.append(f"input_size: [{width}, {height}]")
    lines.append("model:")
    lines.append(f"  name: {_yaml_scalar(model.get('name', doc.get('model_name', '')))}")
    lines.append(f"  version: {_yaml_scalar(model.get('version', doc.get('model_version', '')))}")
    lines.append(f"  task: {_yaml_scalar(model.get('task', doc.get('task_type', 'detection')))}")
    lines.append(f"  format: {_yaml_scalar(model.get('format', 'rknn'))}")
    lines.append(f"  target_platform: {_yaml_scalar(model.get('target_platform', doc.get('target_platform', 'rk3576')))}")
    lines.append(f"  input_size: [{width}, {height}]")
    lines.append("classes:")
    for index, item in enumerate(classes):
        if isinstance(item, dict):
            class_id = int(item.get("id", index))
            name = str(item.get("name", f"class_{index}"))
        else:
            class_id = index
            name = str(item)
        lines.append(f"- id: {class_id}")
        lines.append(f"  name: {_yaml_scalar(name)}")
    lines.append("class_names:")
    for name in class_names:
        lines.append(f"- {_yaml_scalar(name)}")
    lines.append("postprocess:")
    lines.append(f"  conf_threshold: {float(post.get('conf_threshold', 0.25))}")
    lines.append(f"  iou_threshold: {float(post.get('iou_threshold', 0.45))}")
    lines.append(f"  max_det: {int(post.get('max_det', 100))}")
    lines.append("runtime:")
    lines.append(f"  preprocess: {_yaml_scalar(runtime.get('preprocess', 'letterbox'))}")
    lines.append(f"  color: {_yaml_scalar(runtime.get('color', 'rgb'))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_model_id(ctx: PipelineContext) -> str:
    task = _task_for_name(str(ctx.job.get("task_type") or ctx.dataset.get("task_type") or "detection"))
    device_id = str(ctx.dataset.get("source_device_id") or ctx.job.get("source_device_id") or "multi-device")
    customer_id = str(ctx.dataset.get("source_customer_id") or ctx.job.get("source_customer_id") or "multi-customer")
    return f"{device_id}_{customer_id}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"


def _task_for_name(task_type: str) -> str:
    task = str(task_type or "detection").lower()
    if task in {"obb", "obb_detection"}:
        return "obb"
    if task in {"seg", "segment", "segmentation"}:
        return "seg"
    if task in {"cls", "classification"}:
        return "cls"
    return "det"


def _safe_id(value: str) -> str:
    text = str(value or "").strip().replace(" ", "_")
    safe = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法模型 ID")
    return safe


def _make_model_yaml(
    *,
    model_id: str,
    model_name: str,
    version: str,
    task_type: str,
    classes: list[str],
    input_size: list[int],
    target_platform: str,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> dict[str, Any]:
    class_docs = [{"id": i, "name": name} for i, name in enumerate(classes or ["object"])]
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": model_name,
        "model_version": version,
        "task_type": task_type,
        "target_platform": target_platform,
        "input_size": input_size,
        "model": {
            "name": model_name,
            "version": version,
            "task": task_type,
            "format": "rknn",
            "target_platform": target_platform,
            "input_size": input_size,
        },
        "classes": class_docs,
        "class_names": [item["name"] for item in class_docs],
        "postprocess": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
        },
        "runtime": {
            "preprocess": "letterbox",
            "color": "rgb",
        },
    }
