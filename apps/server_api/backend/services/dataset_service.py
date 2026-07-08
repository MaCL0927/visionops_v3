"""数据集版本管理。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .ingest_service import BatchService, IMAGE_EXTENSIONS


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
        dataset_dir = self.datasets_root / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        meta["status"] = "deleted"
        meta["deleted_at_ms"] = int(time.time() * 1000)
        return meta

    def build_dataset(self, *, task_type: str, batch_ids: list[str] | None = None, name: str | None = None) -> dict[str, Any]:
        """Create a dataset version from explicitly selected reviewed batches.

        v3 服务端现在遵循“审核完成即生成 dataset”的流程：
        - 第一步如果选中多个 tar.gz，BatchService 已经把它们合并成一个 batch；
        - 第二步标注器点击“确认审核完成”后，只针对当前 batch 生成一个 dataset；
        - 不再在第三步把所有 accepted 且任务类型一致的 batch 自动合并。

        因此本方法要求调用方明确传入 ``batch_ids``。
        """
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
        dataset_id = _unique_dataset_id(
            self.datasets_root,
            _make_dataset_id(source_device_id, source_customer_id, task_type, created_time),
        )
        dataset_dir = self.datasets_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=False)
        now = int(time.time() * 1000)
        classes = _collect_classes(normalized_selected)
        image_count, label_count = _count_labeled_pairs(normalized_selected)
        if image_count <= 0:
            raise ValueError("选中的 batch 中没有带非空 labels 的图片，请先完成标注审核")

        source_manifests = {
            item["batch_id"]: item.get("manifest", {})
            for item in normalized_selected
            if isinstance(item.get("manifest"), dict)
        }
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
            "yolo_dataset_path": str(dataset_dir / "yolo_dataset"),
            "data_yaml": str(dataset_dir / "yolo_dataset" / "data.yaml"),
            "source_manifests": source_manifests,
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "v3 dataset version: reviewed batch materialized to YOLO dataset automatically after annotator review.",
        }
        _write_json(dataset_dir / "dataset.json", meta)
        _write_json(dataset_dir / "batches.json", normalized_selected)
        _materialize_yolo_dataset(dataset_dir / "yolo_dataset", normalized_selected, classes, task_type)
        meta["materialized_at_ms"] = int(time.time() * 1000)
        _write_json(dataset_dir / "dataset.json", meta)
        return meta


def normalize_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb_detection"
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg"}:
        return "segmentation"
    if task in {"classification", "cls"}:
        return "classification"
    return "detection"


def _common_or_multi(values: list[str], fallback: str) -> str:
    cleaned = [v.strip() for v in values if v and v.strip()]
    if not cleaned:
        return fallback
    unique = sorted(set(cleaned))
    return unique[0] if len(unique) == 1 else fallback


def _task_for_name(task_type: str) -> str:
    if task_type == "obb_detection":
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


def _collect_classes(batches: list[dict[str, Any]]) -> list[str]:
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


def _materialize_yolo_dataset(yolo_dir: Path, batches: list[dict[str, Any]], classes: list[str], task_type: str) -> None:
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
    val_count = max(1, int(round(len(pairs) * 0.2))) if len(pairs) >= 2 else 1
    val = pairs[:val_count]
    train = pairs[val_count:] if len(pairs) > 1 else pairs
    if not train:
        train = val

    for split, items in [("train", train), ("val", val)]:
        for image, label in items:
            name = _unique_name(image, label)
            shutil.copy2(image, yolo_dir / "images" / split / name)
            shutil.copy2(label, yolo_dir / "labels" / split / f"{Path(name).stem}.txt")

    _write_data_yaml(yolo_dir / "data.yaml", yolo_dir, classes or ["object"], task_type)


def _write_data_yaml(path: Path, yolo_dir: Path, classes: list[str], task_type: str) -> None:
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    task_line = "task: segment\n" if task_type == "segmentation" else ("task: obb\n" if task_type == "obb_detection" else "")
    path.write_text(
        f"path: {yolo_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
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


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _unique_name(image_path: Path, label_path: Path) -> str:
    try:
        batch_id = label_path.parents[1].parent.name
    except Exception:
        batch_id = image_path.parent.name
    return f"{batch_id}__{image_path.name}"


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
