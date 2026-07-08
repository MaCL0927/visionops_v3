"""Build a materialized YOLO dataset from accepted v3 batches."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, list_images, normalize_task, read_json, write_json, write_yaml


def run(ctx: PipelineContext) -> dict[str, Any]:
    dataset = ctx.dataset
    task_type = normalize_task(str(ctx.job.get("task_type") or dataset.get("task_type") or "detection"))
    classes = _classes_from_dataset(dataset)
    if not classes:
        classes = ["object"]

    data_root = ctx.work_dir / "yolo_dataset"
    if data_root.exists():
        shutil.rmtree(data_root)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    items = _collect_labeled_items(dataset)
    if not items:
        raise RuntimeError("数据集中没有找到带非空标签的图片，请先在第二步完成标注和审核。")

    rng = random.Random(int(ctx.job.get("split_seed", 42)))
    rng.shuffle(items)
    val_ratio = float(ctx.job.get("val_ratio", 0.2))
    val_count = max(1, int(round(len(items) * val_ratio))) if len(items) >= 2 else 1
    train_items = items[val_count:] if len(items) > 1 else items
    val_items = items[:val_count]
    if not train_items:
        train_items = val_items

    for split, split_items in [("train", train_items), ("val", val_items)]:
        for image_path, label_path in split_items:
            unique_name = _unique_name(image_path, label_path)
            shutil.copy2(image_path, data_root / "images" / split / unique_name)
            shutil.copy2(label_path, data_root / "labels" / split / f"{Path(unique_name).stem}.txt")

    names = {idx: name for idx, name in enumerate(classes)}
    data_yaml = {
        "path": str(data_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": names,
    }
    if task_type == "segmentation":
        data_yaml["task"] = "segment"
    elif task_type == "obb_detection":
        data_yaml["task"] = "obb"
    data_yaml_path = data_root / "data.yaml"
    write_yaml(data_yaml_path, data_yaml)

    report = {
        "status": "success",
        "task_type": task_type,
        "dataset_id": dataset.get("dataset_id"),
        "data_yaml": str(data_yaml_path),
        "dataset_dir": str(data_root),
        "classes": classes,
        "total_labeled_images": len(items),
        "train_images": len(train_items),
        "val_images": len(val_items),
    }
    write_json(ctx.output_dir / "preprocess_report.json", report)
    ctx.log(f"[preprocess] data_yaml={data_yaml_path} train={len(train_items)} val={len(val_items)} classes={classes}")
    return report


def _classes_from_dataset(dataset: dict[str, Any]) -> list[str]:
    classes = dataset.get("classes")
    if isinstance(classes, list) and classes:
        out: list[str] = []
        for i, item in enumerate(classes):
            if isinstance(item, dict):
                out.append(str(item.get("name") or f"class_{i}"))
            else:
                out.append(str(item))
        return out
    for batch in dataset.get("batches", []) if isinstance(dataset.get("batches"), list) else []:
        raw = Path(str(batch.get("raw_path") or ""))
        data = read_json(raw / "annotation_classes.json", {}) or {}
        names = data.get("names")
        if isinstance(names, list) and names:
            return [str(x) for x in names]
    return []


def _collect_labeled_items(dataset: dict[str, Any]) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []
    batches = dataset.get("batches") if isinstance(dataset.get("batches"), list) else []
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        images_dir = _find_images_dir(raw, batch)
        labels_dir = raw / "labels"
        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue
        for image in list_images(images_dir):
            label = labels_dir / f"{image.stem}.txt"
            if label.exists() and label.read_text(encoding="utf-8", errors="ignore").strip():
                items.append((image, label))
    return items


def _find_images_dir(raw: Path, batch: dict[str, Any]) -> Path:
    candidates = []
    if batch.get("images_path"):
        candidates.append(Path(str(batch["images_path"])))
    candidates.extend([raw / "all_images", raw / "images", raw / "positive", raw / "negative", raw])
    for path in candidates:
        if path.is_dir() and list_images(path):
            return path
    return raw / "images"


def _unique_name(image_path: Path, label_path: Path) -> str:
    try:
        raw_dir = label_path.parents[1]
        batch_id = raw_dir.parent.name
    except Exception:
        batch_id = image_path.parent.name
    return f"{batch_id}__{image_path.name}"
