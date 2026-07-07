"""数据集版本管理。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .ingest_service import BatchService


class DatasetService:
    def __init__(self, datasets_root: Path, batch_service: BatchService) -> None:
        self.datasets_root = Path(datasets_root)
        self.batch_service = batch_service
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        result = []
        for dataset_dir in sorted([entry for entry in self.datasets_root.iterdir() if entry.is_dir()], key=lambda x: x.name):
            meta = _read_json(dataset_dir / "dataset.json", {})
            if meta.get("dataset_id"):
                result.append(meta)
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

    def build_dataset(self, *, task_type: str, batch_ids: list[str] | None = None, name: str | None = None) -> dict[str, Any]:
        task_type = str(task_type or "").strip()
        if task_type not in self.batch_service.allowed_task_types:
            raise ValueError(f"不支持的 task_type: {task_type}")

        batches = self.batch_service.list_batches()
        if batch_ids:
            selected = [self.batch_service.get_batch(item) for item in batch_ids]
        else:
            selected = [item for item in batches if item.get("status") == "accepted" and item.get("task_type") == task_type]

        allowed_status = {"extracted", "accepted"}
        selected = [item for item in selected if item.get("status") in allowed_status]
        if not selected:
            raise ValueError("没有可用于构建数据集的 extracted/accepted batch")

        # 第二步才确认任务类型：构建数据集时把选中的 batch 标记为 accepted，并写入 task_type。
        normalized_selected: list[dict[str, Any]] = []
        for item in selected:
            batch = self.batch_service.set_status(item["batch_id"], "accepted", item.get("review_note", ""), task_type=task_type)
            normalized_selected.append(batch)

        dataset_id = f"dataset-{task_type}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        dataset_dir = self.datasets_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=False)
        now = int(time.time() * 1000)
        source_manifests = {
            item["batch_id"]: item.get("manifest", {})
            for item in normalized_selected
            if isinstance(item.get("manifest"), dict)
        }
        meta = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "name": name or dataset_id,
            "task_type": task_type,
            "status": "ready",
            "batch_ids": [item["batch_id"] for item in normalized_selected],
            "image_count": sum(int(item.get("image_count", 0)) for item in normalized_selected),
            "label_count": sum(int(item.get("label_count", 0)) for item in normalized_selected),
            "dataset_path": str(dataset_dir),
            "source_manifests": source_manifests,
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "MVP 阶段仅建立数据集清单，不复制大文件；训练接入时可按 batch raw_path/images_path 构建 YOLO 数据目录。",
        }
        _write_json(dataset_dir / "dataset.json", meta)
        _write_json(dataset_dir / "batches.json", normalized_selected)
        return meta


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
