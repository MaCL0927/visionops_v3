"""Create v3 standard model package from training artifacts."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, normalize_task, write_json


def run(
    ctx: PipelineContext,
    preprocess_report: dict[str, Any],
    train_report: dict[str, Any],
    evaluate_report: dict[str, Any],
    export_report: dict[str, Any],
    rknn_report: dict[str, Any],
) -> dict[str, Any]:
    task_type = _runtime_task(str(ctx.job.get("task_type") or ctx.dataset.get("task_type") or "detection"))
    model_id = str(ctx.job.get("model_id") or _default_model_id(ctx, task_type))[:96]
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
    # For YOLOv8 classification, Ultralytics assigns class ids from the trained
    # model metadata (usually sorted class-folder names).  The annotation UI may
    # keep the user-created order, so using preprocess_report/classes here can
    # produce a wrong class_id -> class_name mapping on the edge runtime.
    # Prefer best.pt names when available, especially for classification.
    trained_classes = _classes_from_trained_pt(train_report.get("best_pt"))
    if task_type == "classification" and trained_classes:
        classes = trained_classes
    version = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(ctx.job.get("model_name") or f"{task_type}-{ctx.dataset.get('dataset_id', 'dataset')}")
    imgsz = int(ctx.job.get("imgsz", export_report.get("imgsz", 640) or 640))
    input_size = [imgsz, imgsz]
    target_platform = str(ctx.job.get("target_platform") or "rk3576")

    model_yaml = _make_model_yaml(
        model_id=model_id,
        model_name=model_name,
        version=version,
        task_type=task_type,
        classes=[str(x) for x in classes],
        input_size=input_size,
        target_platform=target_platform,
        conf_threshold=float(ctx.job.get("conf_threshold", 0.25)),
        iou_threshold=float(ctx.job.get("iou_threshold", 0.45)),
        max_det=int(ctx.job.get("max_det", 100)),
    )
    _write_model_yaml(package_dir / "model.yaml", model_yaml)

    now = int(time.time() * 1000)
    package_meta = {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": model_name,
        "version": version,
        "task_type": task_type,
        "target_platform": target_platform,
        "input_size": input_size,
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

    for report in ctx.output_dir.glob("*_report.json"):
        shutil.copy2(report, package_dir / "logs" / report.name)
    job_log = Path(str(ctx.job.get("log_path") or ctx.job_dir / f"{ctx.job.get('job_id', 'job')}.log"))
    if job_log.exists():
        shutil.copy2(job_log, package_dir / "logs" / "job.log")

    report = {
        "status": "success",
        "model_id": model_id,
        "package_dir": str(package_dir),
        "model_yaml": str(package_dir / "model.yaml"),
        "model_rknn": str(package_dir / "model.rknn"),
        "task_type": task_type,
        "input_size": input_size,
    }
    write_json(ctx.output_dir / "package_v3_model_report.json", report)
    ctx.log(f"[package] model_id={model_id} task={task_type} package_dir={package_dir}")
    return report


def _runtime_task(task_type: str | None) -> str:
    task = normalize_task(task_type)
    if task == "obb":
        return "obb"
    if task == "segmentation":
        return "segmentation"
    if task == "classification":
        return "classification"
    return "detection"


def _default_model_id(ctx: PipelineContext, task_type: str) -> str:
    task = _task_for_name(task_type)
    device_id = str(ctx.dataset.get("source_device_id") or ctx.job.get("source_device_id") or "multi-device")
    customer_id = str(ctx.dataset.get("source_customer_id") or ctx.job.get("source_customer_id") or "multi-customer")
    return f"{device_id}_{customer_id}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"


def _task_for_name(task_type: str) -> str:
    task = _runtime_task(task_type)
    if task == "obb":
        return "obb"
    if task == "segmentation":
        return "seg"
    if task == "classification":
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
    runtime_task = _runtime_task(task_type)
    class_docs = [{"id": i, "name": name} for i, name in enumerate(classes or ["object"])]
    width, height = int(input_size[0]), int(input_size[1])
    # Use two independent lists so PyYAML never emits anchors/aliases.
    top_input_size = [width, height]
    nested_input_size = [width, height]
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": model_name,
        "model_version": version,
        "task_type": runtime_task,
        "target_platform": target_platform,
        "input_size": top_input_size,
        "model": {
            "name": model_name,
            "version": version,
            "task": runtime_task,
            "format": "rknn",
            "target_platform": target_platform,
            "input_size": nested_input_size,
        },
        "classes": class_docs,
        "class_names": [item["name"] for item in class_docs],
        "postprocess": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
        },
        "runtime": {
            "preprocess": "resize" if runtime_task == "classification" else "letterbox",
            "color": "rgb",
        },
    }



def _classes_from_trained_pt(best_pt: Any) -> list[str]:
    """Read class names from an Ultralytics trained checkpoint if possible.

    This is critical for classification because YOLOv8-cls class ids are tied to
    the class-folder order stored in best.pt.  Dataset metadata may preserve UI
    creation order, which can differ from the trained model's actual class ids.
    """
    path = Path(str(best_pt or ""))
    if not path.is_file():
        return []

    # Prefer Ultralytics because it normalizes both old and new checkpoint layouts.
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(path))
        names = getattr(model, "names", None)
        ordered = _ordered_names_from_mapping(names)
        if ordered:
            return ordered
    except Exception:
        pass

    # Lightweight fallback for environments where ultralytics import is not
    # available during packaging but torch can still read the checkpoint.
    try:
        import torch  # type: ignore

        ckpt = torch.load(str(path), map_location="cpu")
        candidates: list[Any] = []
        if isinstance(ckpt, dict):
            candidates.extend([ckpt.get("names"), ckpt.get("model")])
        else:
            candidates.append(ckpt)
        for item in candidates:
            names = item
            if hasattr(item, "names"):
                names = getattr(item, "names")
            ordered = _ordered_names_from_mapping(names)
            if ordered:
                return ordered
    except Exception:
        pass
    return []


def _ordered_names_from_mapping(names: Any) -> list[str]:
    if isinstance(names, dict):
        ordered: list[str] = []
        for key in sorted(names.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x)):
            value = str(names[key]).strip()
            if value:
                ordered.append(value)
        return ordered
    if isinstance(names, (list, tuple)):
        return [str(x).strip() for x in names if str(x).strip()]
    return []


def _yaml_scalar(value: Any) -> str:
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "[", "]", "{", "}", ",", "&", "*", "!", "|", ">", "'", '"']) or text.lower() in {"true", "false", "null", "none"}:
        return "'" + text.replace("'", "''") + "'"
    return text


def _write_model_yaml(path: Path, doc: dict[str, Any]) -> None:
    """Write the deployment model.yaml in the simple contract parsed by C++ Runtime."""
    size = doc.get("input_size") or [640, 640]
    width, height = int(size[0]), int(size[1])
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    post = doc.get("postprocess") if isinstance(doc.get("postprocess"), dict) else {}
    runtime = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
    classes = doc.get("classes") if isinstance(doc.get("classes"), list) else []
    class_names = doc.get("class_names") if isinstance(doc.get("class_names"), list) else []

    lines: list[str] = [
        "schema_version: '1.0'",
        f"model_id: {_yaml_scalar(doc.get('model_id', ''))}",
        f"model_name: {_yaml_scalar(doc.get('model_name', ''))}",
        f"model_version: {_yaml_scalar(doc.get('model_version', ''))}",
        f"task_type: {_yaml_scalar(doc.get('task_type', 'detection'))}",
        f"target_platform: {_yaml_scalar(doc.get('target_platform', 'rk3576'))}",
        f"input_size: [{width}, {height}]",
        "model:",
        f"  name: {_yaml_scalar(model.get('name', doc.get('model_name', '')))}",
        f"  version: {_yaml_scalar(model.get('version', doc.get('model_version', '')))}",
        f"  task: {_yaml_scalar(model.get('task', doc.get('task_type', 'detection')))}",
        f"  format: {_yaml_scalar(model.get('format', 'rknn'))}",
        f"  target_platform: {_yaml_scalar(model.get('target_platform', doc.get('target_platform', 'rk3576')))}",
        f"  input_size: [{width}, {height}]",
        "classes:",
    ]
    for idx, item in enumerate(classes):
        if isinstance(item, dict):
            cid = int(item.get("id", idx))
            name = str(item.get("name", f"class_{idx}"))
        else:
            cid = idx
            name = str(item)
        lines.append(f"- id: {cid}")
        lines.append(f"  name: {_yaml_scalar(name)}")
    lines.append("class_names:")
    for name in class_names:
        lines.append(f"- {_yaml_scalar(name)}")
    lines.extend([
        "postprocess:",
        f"  conf_threshold: {float(post.get('conf_threshold', 0.25))}",
        f"  iou_threshold: {float(post.get('iou_threshold', 0.45))}",
        f"  max_det: {int(post.get('max_det', 100))}",
        "runtime:",
        f"  preprocess: {_yaml_scalar(runtime.get('preprocess', 'letterbox'))}",
        f"  color: {_yaml_scalar(runtime.get('color', 'rgb'))}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
