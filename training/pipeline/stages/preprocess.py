"""Build a materialized Ultralytics dataset from accepted v3 batches."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from training.pipeline.common import PipelineContext, IMAGE_EXTS, list_images, normalize_task, read_json, write_json, write_yaml


CLASSIFICATION_ROOT_NAMES = {"cls", "classification", "raw_classification", "classes"}
RESERVED_DIR_NAMES = {
    "labels", "labels_auto", "all_images", "images", "positive", "negative",
    "quick_train", "roi_classification_sessions", "previews", "candidates",
}


def run(ctx: PipelineContext) -> dict[str, Any]:
    dataset = ctx.dataset
    task_type = normalize_task(str(ctx.job.get("task_type") or dataset.get("task_type") or "detection"))
    classes = _classes_from_dataset(dataset)
    if not classes:
        classes = ["object"]

    shared = _reuse_materialized_dataset(ctx, dataset, task_type, classes)
    if shared is not None:
        return shared

    # Backward-compatible fallback for legacy dataset.json files that predate
    # dataset materialization metadata. New datasets never enter this path.
    ctx.log("[preprocess] legacy dataset metadata detected; rebuilding a job-local dataset copy")
    if task_type == "classification":
        return _run_classification(ctx, dataset, classes)
    return _run_yolo_labels(ctx, dataset, task_type, classes)


def _reuse_materialized_dataset(
    ctx: PipelineContext,
    dataset: dict[str, Any],
    task_type: str,
    classes: list[str],
) -> dict[str, Any] | None:
    """Use the immutable dataset under server_data/datasets directly.

    The old pipeline copied every image into ``jobs/<job_id>/work``. The
    dataset service already created the Ultralytics layout, so repeating that
    work consumed a full extra dataset per training job. This path validates
    the materialized dataset and writes only a small preprocess report.
    """

    dataset_dir_raw = str(dataset.get("yolo_dataset_path") or "").strip()
    if not dataset_dir_raw:
        return None
    dataset_dir = Path(dataset_dir_raw).expanduser().resolve()
    if not dataset_dir.is_dir():
        return None

    if task_type == "classification":
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            return None
        train_images = _count_images_recursive(train_dir)
        val_images = _count_images_recursive(val_dir)
        if train_images <= 0 and val_images <= 0:
            return None
        data_path = dataset_dir
        data_yaml = dataset_dir / "data.yaml"
    else:
        train_dir = dataset_dir / "images" / "train"
        val_dir = dataset_dir / "images" / "val"
        data_yaml_raw = str(dataset.get("data_yaml") or dataset.get("training_data_path") or "").strip()
        data_yaml = Path(data_yaml_raw).expanduser().resolve() if data_yaml_raw else dataset_dir / "data.yaml"
        if not train_dir.is_dir() or not val_dir.is_dir() or not data_yaml.is_file():
            return None
        train_images = _count_images_recursive(train_dir)
        val_images = _count_images_recursive(val_dir)
        if train_images <= 0 and val_images <= 0:
            return None
        data_path = data_yaml

    # A stale job-local copy may exist if an older job is manually resumed
    # with the new code. It is safe to remove because this report now points to
    # the canonical dataset directory.
    for stale_name in ("yolo_dataset", "cls_dataset"):
        stale = ctx.work_dir / stale_name
        if stale.is_dir() and stale.resolve() != dataset_dir:
            shutil.rmtree(stale)

    report = {
        "status": "success",
        "task_type": task_type,
        "dataset_id": dataset.get("dataset_id"),
        "data_yaml": str(data_yaml),
        "data_path": str(data_path),
        "dataset_dir": str(dataset_dir),
        "classes": classes,
        "total_labeled_images": train_images + val_images,
        "train_images": train_images,
        "val_images": val_images,
        "storage_mode": "shared_dataset_reference",
        "source_dataset_path": str(dataset.get("dataset_path") or dataset_dir.parent),
        "job_dataset_copy_created": False,
    }
    if task_type == "classification":
        report["classification_layout"] = "ultralytics_folder"
    write_json(ctx.output_dir / "preprocess_report.json", report)
    ctx.log(
        f"[preprocess] reuse dataset={dataset_dir} train={train_images} "
        f"val={val_images} storage=shared_reference"
    )
    return report


def _count_images_recursive(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _run_yolo_labels(ctx: PipelineContext, dataset: dict[str, Any], task_type: str, classes: list[str]) -> dict[str, Any]:
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
    elif task_type == "obb":
        data_yaml["task"] = "obb"
    data_yaml_path = data_root / "data.yaml"
    write_yaml(data_yaml_path, data_yaml)

    report = {
        "status": "success",
        "task_type": task_type,
        "dataset_id": dataset.get("dataset_id"),
        "data_yaml": str(data_yaml_path),
        "data_path": str(data_yaml_path),
        "dataset_dir": str(data_root),
        "classes": classes,
        "total_labeled_images": len(items),
        "train_images": len(train_items),
        "val_images": len(val_items),
    }
    write_json(ctx.output_dir / "preprocess_report.json", report)
    ctx.log(f"[preprocess] data_yaml={data_yaml_path} train={len(train_items)} val={len(val_items)} classes={classes}")
    return report


def _run_classification(ctx: PipelineContext, dataset: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    """Materialize an Ultralytics classification folder dataset.

    Expected layout for classification source data is one of:
      raw/<class_name>/*.jpg
      raw/images/<class_name>/*.jpg
      raw/all_images/<class_name>/*.jpg
      raw/classification/<class_name>/*.jpg
      raw/raw_classification/<class_name>/*.jpg

    For compatibility, if class folders are absent but YOLO txt labels exist,
    the first class id in each non-empty label file is used as the image class.
    """
    data_root = ctx.work_dir / "cls_dataset"
    if data_root.exists():
        shutil.rmtree(data_root)
    (data_root / "train").mkdir(parents=True, exist_ok=True)
    (data_root / "val").mkdir(parents=True, exist_ok=True)

    items = _collect_classification_items(dataset, classes)
    if not items:
        raise RuntimeError(
            "classification 数据集中没有找到分类图片。请使用 class folders 格式，例如 raw/ok/*.jpg、raw/ng/*.jpg，"
            "或提供可从 labels 推断类别的图片。"
        )

    class_names = _ordered_class_names(items, classes)
    rng = random.Random(int(ctx.job.get("split_seed", 42)))
    rng.shuffle(items)
    val_ratio = float(ctx.job.get("val_ratio", 0.2))
    val_count = max(1, int(round(len(items) * val_ratio))) if len(items) >= 2 else 1
    val_items = items[:val_count]
    train_items = items[val_count:] if len(items) > 1 else items
    if not train_items:
        train_items = val_items

    for class_name in class_names:
        (data_root / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (data_root / "val" / class_name).mkdir(parents=True, exist_ok=True)

    for split, split_items in [("train", train_items), ("val", val_items)]:
        for image_path, class_name in split_items:
            dst_name = _unique_classification_name(image_path)
            shutil.copy2(image_path, data_root / split / class_name / dst_name)

    # data.yaml is not used by `yolo classify train`, but writing it keeps the
    # v3 dataset preview and downstream reports consistent with other tasks.
    names = {idx: name for idx, name in enumerate(class_names)}
    data_yaml_path = data_root / "data.yaml"
    write_yaml(data_yaml_path, {"path": str(data_root), "train": "train", "val": "val", "nc": len(class_names), "names": names, "task": "classify"})

    report = {
        "status": "success",
        "task_type": "classification",
        "dataset_id": dataset.get("dataset_id"),
        "data_yaml": str(data_yaml_path),
        "data_path": str(data_root),
        "dataset_dir": str(data_root),
        "classes": class_names,
        "total_labeled_images": len(items),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "classification_layout": "ultralytics_folder",
    }
    write_json(ctx.output_dir / "preprocess_report.json", report)
    ctx.log(f"[preprocess] classification data={data_root} train={len(train_items)} val={len(val_items)} classes={class_names}")
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


def _collect_classification_items(dataset: dict[str, Any], classes: list[str]) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    batches = dataset.get("batches") if isinstance(dataset.get("batches"), list) else []
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        items.extend(_classification_items_from_folders(raw))
        if items:
            continue
        items.extend(_classification_items_from_labels(raw, batch, classes))
    return sorted(items, key=lambda x: (x[1], str(x[0])))


def _classification_items_from_folders(raw: Path) -> list[tuple[Path, str]]:
    roots: list[Path] = []
    roots.extend([raw / name for name in CLASSIFICATION_ROOT_NAMES])
    roots.extend([raw / "images", raw / "all_images", raw])
    seen: set[str] = set()
    items: list[tuple[Path, str]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name in RESERVED_DIR_NAMES or class_dir.name.startswith("."):
                continue
            images = _list_images_recursive_one_level(class_dir)
            if not images:
                continue
            class_name = _safe_class_name(class_dir.name)
            for image in images:
                key = str(image.resolve())
                if key not in seen:
                    seen.add(key)
                    items.append((image, class_name))
        if items:
            return items
    return items


def _classification_items_from_labels(raw: Path, batch: dict[str, Any], classes: list[str]) -> list[tuple[Path, str]]:
    images_dir = _find_images_dir(raw, batch)
    labels_dir = raw / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for image in list_images(images_dir):
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            continue
        class_id = _first_class_id(label)
        if class_id is None:
            continue
        class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
        out.append((image, _safe_class_name(class_name)))
    return out


def _first_class_id(label_path: Path) -> int | None:
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            return int(float(parts[0]))
        except Exception:
            continue
    return None


def _ordered_class_names(items: list[tuple[Path, str]], preferred: list[str]) -> list[str]:
    present = {name for _, name in items}
    ordered = [_safe_class_name(name) for name in preferred if _safe_class_name(name) in present]
    for name in sorted(present):
        if name not in ordered:
            ordered.append(name)
    return ordered or ["class_0"]


def _find_images_dir(raw: Path, batch: dict[str, Any]) -> Path:
    candidates = []
    if batch.get("images_path"):
        candidates.append(Path(str(batch["images_path"])))
    candidates.extend([raw / "all_images", raw / "images", raw / "positive", raw / "negative", raw])
    for path in candidates:
        if path.is_dir() and list_images(path):
            return path
    return raw / "images"


def _list_images_recursive_one_level(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _unique_name(image_path: Path, label_path: Path) -> str:
    try:
        raw_dir = label_path.parents[1]
        batch_id = raw_dir.parent.name
    except Exception:
        batch_id = image_path.parent.name
    return f"{batch_id}__{image_path.name}"


def _unique_classification_name(image_path: Path) -> str:
    try:
        batch_id = image_path.parents[2].name
    except Exception:
        batch_id = image_path.parent.name
    return f"{batch_id}__{image_path.name}"


def _safe_class_name(value: str) -> str:
    text = str(value or "").strip().replace("\\", "_").replace("/", "_").replace("..", "_")
    text = "_".join(text.split())
    return text or "class_0"
