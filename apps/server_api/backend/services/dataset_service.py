"""数据集版本管理。"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from .ingest_service import BatchService, IMAGE_EXTENSIONS
from .storage_utils import link_or_copy_immutable

CLASSIFICATION_ROOT_NAMES = {"cls", "classification", "raw_classification", "classes"}
RESERVED_DIR_NAMES = {
    "labels", "labels_auto", "all_images", "images", "positive", "negative",
    "quick_train", "roi_classification_sessions", "previews", "candidates",
}


class DatasetService:
    def __init__(self, datasets_root: Path, batch_service: BatchService) -> None:
        self.datasets_root = Path(datasets_root)
        self.batch_service = batch_service
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        result = []
        for dataset_dir in [entry for entry in self.datasets_root.iterdir() if entry.is_dir()]:
            meta = _read_json(dataset_dir / "dataset.json", {})
            if meta.get("dataset_id"):
                result.append(meta)
        result.sort(key=lambda item: int(item.get("created_at_ms") or item.get("updated_at_ms") or 0), reverse=True)
        return result

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset_id = _safe_id(dataset_id)
        meta = _read_json(self.datasets_root / dataset_id / "dataset.json", {})
        if not meta.get("dataset_id"):
            raise FileNotFoundError(f"数据集不存在: {dataset_id}")
        batches = _read_json(self.datasets_root / dataset_id / "batches.json", [])
        if isinstance(batches, list):
            meta["batches"] = batches
        return meta

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset_id = _safe_id(dataset_id)
        meta = self.get_dataset(dataset_id)
        active_refs = self.active_training_references(dataset_id)
        if active_refs:
            job_ids = ", ".join(str(item.get("job_id") or "unknown") for item in active_refs)
            raise ValueError(f"数据集正在被训练任务引用，不能删除: {job_ids}")
        dataset_dir = self.datasets_root / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        meta["status"] = "deleted"
        meta["deleted_at_ms"] = int(time.time() * 1000)
        return meta

    def acquire_training_reference(self, dataset_id: str, job_id: str, job_path: Path) -> None:
        """Protect a dataset while a training process is actively using it."""

        dataset_id = _safe_id(dataset_id)
        job_id = _safe_id(job_id)
        dataset_dir = self.datasets_root / dataset_id
        if not (dataset_dir / "dataset.json").is_file():
            raise FileNotFoundError(f"数据集不存在: {dataset_id}")
        ref = {
            "dataset_id": dataset_id,
            "job_id": job_id,
            "job_path": str(Path(job_path)),
            "job_json": str(Path(job_path) / "job.json"),
            "created_at_ms": int(time.time() * 1000),
        }
        _write_json(dataset_dir / ".active_job_refs" / f"{job_id}.json", ref)

    def release_training_reference(self, dataset_id: str, job_id: str) -> None:
        dataset_id = _safe_id(dataset_id)
        job_id = _safe_id(job_id)
        ref_path = self.datasets_root / dataset_id / ".active_job_refs" / f"{job_id}.json"
        try:
            ref_path.unlink()
        except FileNotFoundError:
            pass
        try:
            ref_path.parent.rmdir()
        except OSError:
            pass

    def active_training_references(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return live pending/running references and prune stale markers."""

        dataset_id = _safe_id(dataset_id)
        refs_root = self.datasets_root / dataset_id / ".active_job_refs"
        if not refs_root.is_dir():
            return []
        active: list[dict[str, Any]] = []
        for ref_path in sorted(refs_root.glob("*.json")):
            ref = _read_json(ref_path, {})
            job_json = Path(str(ref.get("job_json") or ""))
            # The marker itself is authoritative. A canceled job may still be
            # shutting down and reading the dataset; its worker removes the
            # marker only after the child process has exited. Prune only truly
            # orphaned markers whose job directory no longer exists.
            if job_json.is_file():
                active.append(ref)
                continue
            try:
                ref_path.unlink()
            except OSError:
                pass
        try:
            refs_root.rmdir()
        except OSError:
            pass
        return active

    def build_dataset(self, *, task_type: str, batch_ids: list[str] | None = None, name: str | None = None) -> dict[str, Any]:
        """Create a dataset version from explicitly selected reviewed batches."""
        task_type = normalize_task(task_type)
        if task_type not in self.batch_service.allowed_task_types:
            raise ValueError(f"不支持的 task_type: {task_type}")
        if not batch_ids:
            raise ValueError("构建数据集必须明确指定 batch_ids；多包合并应在第一步处理上传包时完成")

        selected = [self.batch_service.get_batch(item) for item in batch_ids]
        selected = [item for item in selected if item.get("status") in {"extracted", "accepted"}]
        if not selected:
            raise ValueError("没有可用于构建数据集的 extracted/accepted batch")

        normalized_selected: list[dict[str, Any]] = []
        for item in selected:
            batch = self.batch_service.set_status(item["batch_id"], "accepted", item.get("review_note", ""), task_type=task_type)
            normalized_selected.append(batch)

        source_device_id = _common_or_multi([str(item.get("device_id") or "") for item in normalized_selected], "multi-device")
        source_customer_id = _common_or_multi([str(item.get("customer_id") or "") for item in normalized_selected], "multi-customer")
        created_time = time.strftime("%Y%m%d_%H%M%S")
        dataset_id = _unique_dataset_id(self.datasets_root, _make_dataset_id(source_device_id, source_customer_id, task_type, created_time))
        dataset_dir = self.datasets_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=False)
        now = int(time.time() * 1000)

        if task_type == "classification":
            classes = _collect_classification_classes(normalized_selected)
            image_count, label_count = _count_classification_images(normalized_selected)
            dataset_subdir = "cls_dataset"
            data_path = str(dataset_dir / dataset_subdir)
        else:
            classes = _collect_yolo_classes(normalized_selected)
            image_count, label_count = _count_labeled_pairs(normalized_selected)
            dataset_subdir = "yolo_dataset"
            data_path = str(dataset_dir / dataset_subdir / "data.yaml")

        if image_count <= 0:
            if task_type == "classification":
                raise ValueError("选中的 batch 中没有分类图片。请使用 raw/<class_name>/*.jpg 或 raw/images/<class_name>/*.jpg 格式。")
            raise ValueError("选中的 batch 中没有带非空 labels 的图片，请先完成标注审核")

        source_manifests = {item["batch_id"]: item.get("manifest", {}) for item in normalized_selected if isinstance(item.get("manifest"), dict)}
        meta = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "name": name or dataset_id,
            "source_device_id": source_device_id,
            "source_customer_id": source_customer_id,
            "source_batch_count": len(normalized_selected),
            "task_type": task_type,
            "status": "ready",
            "batch_ids": [item["batch_id"] for item in normalized_selected],
            "image_count": image_count,
            "label_count": label_count,
            "class_count": len(classes),
            "classes": [{"id": i, "name": cls_name} for i, cls_name in enumerate(classes or ["object"])],
            "dataset_path": str(dataset_dir),
            "dataset_json": str(dataset_dir / "dataset.json"),
            "batches_json": str(dataset_dir / "batches.json"),
            "yolo_dataset_path": str(dataset_dir / dataset_subdir),
            "data_yaml": data_path,
            "training_data_path": data_path,
            "source_manifests": source_manifests,
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "Reviewed batch materialized once under datasets; image bytes are hard-linked from batches when possible, and training jobs reference this dataset directly.",
        }
        _write_json(dataset_dir / "dataset.json", meta)
        _write_json(dataset_dir / "batches.json", normalized_selected)
        if task_type == "classification":
            storage = _materialize_classification_dataset(dataset_dir / dataset_subdir, normalized_selected, classes)
        else:
            storage = _materialize_yolo_dataset(dataset_dir / dataset_subdir, normalized_selected, classes, task_type)
        meta["storage"] = storage
        meta["storage_mode"] = storage.get("storage_mode")
        meta["materialized_at_ms"] = int(time.time() * 1000)
        _write_json(dataset_dir / "dataset.json", meta)
        return meta


def normalize_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg"}:
        return "segmentation"
    if task in {"classification", "cls", "classify"}:
        return "classification"
    return "detection"


def _common_or_multi(values: list[str], fallback: str) -> str:
    cleaned = [v.strip() for v in values if v and v.strip()]
    if not cleaned:
        return fallback
    unique = sorted(set(cleaned))
    return unique[0] if len(unique) == 1 else fallback


def _task_for_name(task_type: str) -> str:
    if task_type == "obb":
        return "obb"
    if task_type == "segmentation":
        return "seg"
    if task_type == "classification":
        return "cls"
    return "det"


def _make_dataset_id(device_id: str, customer_id: str, task_type: str, created_time: str) -> str:
    return _safe_id(f"{device_id}_{customer_id}_{_task_for_name(task_type)}_{created_time}")


def _unique_dataset_id(root: Path, base: str) -> str:
    candidate = base
    index = 2
    while (root / candidate).exists():
        candidate = f"{base}_{index:02d}"
        index += 1
    return candidate


def _collect_yolo_classes(batches: list[dict[str, Any]]) -> list[str]:
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        data = _read_json(raw / "annotation_classes.json", {}) or {}
        names = data.get("names")
        if isinstance(names, list) and names:
            return [str(x) for x in names]
    max_class_id = -1
    for batch in batches:
        labels_dir = Path(str(batch.get("raw_path") or "")) / "labels"
        if not labels_dir.is_dir():
            continue
        for label in labels_dir.glob("*.txt"):
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    max_class_id = max(max_class_id, int(float(parts[0])))
                except Exception:
                    pass
    return [f"class_{i}" for i in range(max_class_id + 1)] if max_class_id >= 0 else ["object"]


def _collect_classification_classes(batches: list[dict[str, Any]]) -> list[str]:
    # Classification 类别顺序优先使用标注器保存的 annotation_classes.json，
    # 这样 UI 中 0/1/2 的类别顺序能继续传递到 YOLOv8 cls data.yaml。
    classes: list[str] = []
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        data = _read_json(raw / "annotation_classes.json", {}) or {}
        names = data.get("names")
        if isinstance(names, list):
            for name in names:
                safe = _safe_class_name(str(name))
                if safe and safe not in classes:
                    classes.append(safe)
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        for _, class_name in _classification_items_from_folders(raw):
            if class_name not in classes:
                classes.append(class_name)
    if classes:
        return classes
    return _collect_yolo_classes(batches)


def _count_labeled_pairs(batches: list[dict[str, Any]]) -> tuple[int, int]:
    count = 0
    labels = 0
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        images_dir = _find_images_dir(raw, batch)
        labels_dir = raw / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image in _list_images(images_dir):
            label = labels_dir / f"{image.stem}.txt"
            if label.exists() and label.read_text(encoding="utf-8", errors="ignore").strip():
                count += 1
                labels += 1
    return count, labels


def _count_classification_images(batches: list[dict[str, Any]]) -> tuple[int, int]:
    count = 0
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        items = _classification_items_from_folders(raw)
        if not items:
            classes = _collect_yolo_classes([batch])
            items = _classification_items_from_labels(raw, batch, classes)
        count += len(items)
    return count, count


def _materialize_yolo_dataset(yolo_dir: Path, batches: list[dict[str, Any]], classes: list[str], task_type: str) -> dict[str, Any]:
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        images_dir = _find_images_dir(raw, batch)
        labels_dir = raw / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image in _list_images(images_dir):
            label = labels_dir / f"{image.stem}.txt"
            if label.exists() and label.read_text(encoding="utf-8", errors="ignore").strip():
                pairs.append((image, label))
    pairs = sorted(pairs, key=lambda x: str(x[0]))
    if not pairs:
        raise ValueError("无法物化数据集：没有 image+label 配对")
    val, train = _split_items(pairs)

    storage = _new_storage_report()
    for split, items in [("train", train), ("val", val)]:
        for image, label in items:
            name = _unique_name(image, label)
            _materialize_image(image, yolo_dir / "images" / split / name, storage)
            label_dst = yolo_dir / "labels" / split / f"{Path(name).stem}.txt"
            shutil.copy2(label, label_dst)
            storage["label_files"] += 1
            storage["label_bytes"] += int(label.stat().st_size)

    _write_data_yaml(yolo_dir / "data.yaml", yolo_dir, classes or ["object"], task_type)
    return _finalize_storage_report(storage)


def _materialize_classification_dataset(cls_dir: Path, batches: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    if cls_dir.exists():
        shutil.rmtree(cls_dir)
    (cls_dir / "train").mkdir(parents=True, exist_ok=True)
    (cls_dir / "val").mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, str]] = []
    for batch in batches:
        raw = Path(str(batch.get("raw_path") or ""))
        batch_items = _classification_items_from_folders(raw)
        if not batch_items:
            batch_items = _classification_items_from_labels(raw, batch, classes)
        items.extend(batch_items)
    items = sorted(items, key=lambda x: (x[1], str(x[0])))
    if not items:
        raise ValueError("无法物化 classification 数据集：没有分类图片")

    class_names = _ordered_class_names(items, classes)
    for class_name in class_names:
        (cls_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (cls_dir / "val" / class_name).mkdir(parents=True, exist_ok=True)

    storage = _new_storage_report()
    val, train = _split_items(items)
    for split, split_items in [("train", train), ("val", val)]:
        for image, class_name in split_items:
            _materialize_image(image, cls_dir / split / class_name / _unique_classification_name(image), storage)

    _write_data_yaml(cls_dir / "data.yaml", cls_dir, class_names, "classification")
    return _finalize_storage_report(storage)


def _new_storage_report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "image_files": 0,
        "image_bytes": 0,
        "hardlinked_images": 0,
        "copied_images": 0,
        "label_files": 0,
        "label_bytes": 0,
        "estimated_saved_bytes": 0,
        "fallback_errors": [],
    }


def _materialize_image(src: Path, dst: Path, storage: dict[str, Any]) -> None:
    result = link_or_copy_immutable(src, dst)
    size = int(result.get("size_bytes") or 0)
    storage["image_files"] += 1
    storage["image_bytes"] += size
    if result.get("mode") == "hardlink":
        storage["hardlinked_images"] += 1
        storage["estimated_saved_bytes"] += size
    else:
        storage["copied_images"] += 1
        error = str(result.get("fallback_error") or "")
        if error and error not in storage["fallback_errors"] and len(storage["fallback_errors"]) < 5:
            storage["fallback_errors"].append(error)


def _finalize_storage_report(storage: dict[str, Any]) -> dict[str, Any]:
    image_files = int(storage.get("image_files") or 0)
    hardlinks = int(storage.get("hardlinked_images") or 0)
    copies = int(storage.get("copied_images") or 0)
    if image_files and hardlinks == image_files:
        mode = "hardlink_images_copy_labels"
    elif image_files and copies == image_files:
        mode = "copy_fallback"
    else:
        mode = "mixed_hardlink_copy"
    storage["storage_mode"] = mode
    storage["physical_image_bytes_added"] = int(storage.get("image_bytes") or 0) - int(storage.get("estimated_saved_bytes") or 0)
    storage["physical_bytes_added_estimate"] = int(storage["physical_image_bytes_added"]) + int(storage.get("label_bytes") or 0)
    return storage


def _split_items(items: list[Any]) -> tuple[list[Any], list[Any]]:
    items = list(items)
    random.Random(42).shuffle(items)
    val_count = max(1, int(round(len(items) * 0.2))) if len(items) >= 2 else 1
    val = items[:val_count]
    train = items[val_count:] if len(items) > 1 else items
    if not train:
        train = val
    return val, train


def _write_data_yaml(path: Path, yolo_dir: Path, classes: list[str], task_type: str) -> None:
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    if task_type == "segmentation":
        task_line = "task: segment\n"
        train_line = "train: images/train\nval: images/val\n"
    elif task_type == "obb":
        task_line = "task: obb\n"
        train_line = "train: images/train\nval: images/val\n"
    elif task_type == "classification":
        task_line = "task: classify\n"
        train_line = "train: train\nval: val\n"
    else:
        task_line = ""
        train_line = "train: images/train\nval: images/val\n"
    path.write_text(
        f"path: {yolo_dir.as_posix()}\n"
        f"{train_line}"
        f"{task_line}"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names_lines}\n",
        encoding="utf-8",
    )


def _find_images_dir(raw: Path, batch: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if batch.get("images_path"):
        candidates.append(Path(str(batch["images_path"])))
    candidates.extend([raw / "all_images", raw / "images", raw / "positive", raw / "negative", raw])
    for path in candidates:
        if path.is_dir() and _list_images(path):
            return path
    return raw / "images"


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
            images = _list_images_recursive(class_dir)
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
    for image in _list_images(images_dir):
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


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _list_images_recursive(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _unique_name(image_path: Path, label_path: Path) -> str:
    try:
        batch_id = label_path.parents[1].parent.name
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


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in str(value or "") if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法 ID")
    return safe


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
