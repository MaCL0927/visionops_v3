"""上传包扫描、解压与批次管理。"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..storage.json_store import JsonStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt", ".xml"}
MANIFEST_NAMES = {"manifest.json", "collector_manifest.json"}
IMAGE_DIR_NAMES = {"images", "all_images", "positive", "negative"}
LABEL_DIR_NAMES = {"labels", "labels_auto"}


def _now_ms() -> int:
    return int(time.time() * 1000)


class BatchService:
    """管理服务端数据批次。

    v3 服务端第一步不再要求用户手动输入 device_id / task_type。
    边缘端 Web 打好的 ``*.tar.gz`` 先放入 ``incoming_root``，服务端扫描后再解压为
    ``batches/<batch_id>/raw``。device/customer/time 优先从包名和 manifest 推导；任务类型
    留到第二步“标注与审核/构建数据集”时再确认。
    """

    def __init__(
        self,
        batches_root: Path,
        allowed_task_types: tuple[str, ...],
        incoming_root: Path | None = None,
    ) -> None:
        self.batches_root = Path(batches_root)
        self.allowed_task_types = set(allowed_task_types)
        self.incoming_root = Path(incoming_root) if incoming_root else self.batches_root.parent / "incoming"
        self.processed_root = self.incoming_root / "processed"
        self.failed_root = self.incoming_root / "failed"
        self.batches_root.mkdir(parents=True, exist_ok=True)
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.failed_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # v3/v2-compatible incoming package flow
    # ------------------------------------------------------------------
    def list_incoming_packages(self) -> list[dict[str, Any]]:
        """列出 incoming_root 下尚未处理的 tar.gz 包，按修改时间倒序。"""
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        packages: list[dict[str, Any]] = []
        for path in sorted(self.incoming_root.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            batch_id, device_id, customer_id, captured_at = parse_package_name(path.name)
            stat = path.stat()
            packages.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "relative_path": path.name,
                    "size_mb": round(stat.st_size / 1024 / 1024, 3),
                    "mtime_ms": int(stat.st_mtime * 1000),
                    "batch_id": batch_id,
                    "device_id": device_id,
                    "customer_id": customer_id,
                    "captured_at": captured_at,
                    "status": "pending",
                }
            )
        return packages

    def process_incoming_packages(self, packages: list[str]) -> dict[str, Any]:
        """处理选中的 incoming tar.gz。

        - 单包：解压为一个 batch。
        - 多包：合并为一个 batch，适合 v2 的“多包自动合并”场景。
        """
        package_paths = [self._resolve_incoming_package(item) for item in packages]
        if not package_paths:
            raise ValueError("请先选择要处理的上传压缩包")
        if len(package_paths) == 1:
            return self._process_one_tar_package(package_paths[0])
        return self._process_multiple_tar_packages(package_paths)

    def _process_one_tar_package(self, package_path: Path) -> dict[str, Any]:
        batch_id, name_device_id, name_customer_id, captured_at = parse_package_name(package_path.name)
        batch_dir = self.batches_root / batch_id
        raw_dir = batch_dir / "raw"
        if batch_dir.exists():
            raise FileExistsError(f"批次目录已存在: {batch_dir}")
        try:
            with tempfile.TemporaryDirectory(prefix="visionops-ingest-") as tmp:
                tmp_dir = Path(tmp)
                _safe_extract_tar_gz(package_path, tmp_dir)
                dataset_root = _find_package_root(tmp_dir)
                shutil.copytree(dataset_root, raw_dir)

            manifest = _read_manifest(raw_dir)
            device_id = _pick_text(manifest, ["device_id", "equipment_id", "edge_device_id"], name_device_id)
            customer_id = _pick_text(manifest, ["customer_id", "cust_id", "user_id"], name_customer_id)
            image_count, label_count, total_files = _count_dataset_files(raw_dir)
            now = _now_ms()
            meta = {
                "schema_version": "1.0",
                "batch_id": batch_id,
                "device_id": device_id,
                "customer_id": customer_id,
                "captured_at": captured_at,
                "task_type": "unassigned",
                "source": "incoming_tar_gz",
                "source_package": package_path.name,
                "source_package_path": str(package_path),
                "status": "extracted",
                "image_count": image_count,
                "label_count": label_count,
                "total_files": total_files,
                "batch_path": str(batch_dir),
                "raw_path": str(raw_dir),
                "images_path": _best_existing_dir(raw_dir, ["images", "all_images", "positive"]),
                "labels_path": _best_existing_dir(raw_dir, ["labels", "labels_auto"]),
                "manifest_path": str(raw_dir / "manifest.json") if (raw_dir / "manifest.json").exists() else "",
                "manifest": manifest,
                "created_at_ms": now,
                "updated_at_ms": now,
                "review_note": "",
            }
            self._store(batch_id).write(meta)
            moved = _move_package(package_path, self.processed_root)
            meta["processed_package_path"] = str(moved)
            self._store(batch_id).write(meta)
            return meta
        except Exception:
            if package_path.exists():
                _move_package(package_path, self.failed_root)
            raise

    def _process_multiple_tar_packages(self, package_paths: list[Path]) -> dict[str, Any]:
        first_batch_id, first_device_id, first_customer_id, first_captured_at = parse_package_name(package_paths[0].name)
        batch_id = safe_name(f"{first_device_id}_{first_customer_id}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        batch_dir = self.batches_root / batch_id
        raw_dir = batch_dir / "raw"
        if batch_dir.exists():
            raise FileExistsError(f"批次目录已存在: {batch_dir}")
        raw_dir.mkdir(parents=True, exist_ok=False)
        sources: list[dict[str, Any]] = []
        try:
            for package_path in package_paths:
                sub_batch_id, device_id, customer_id, captured_at = parse_package_name(package_path.name)
                with tempfile.TemporaryDirectory(prefix="visionops-merge-") as tmp:
                    tmp_dir = Path(tmp)
                    _safe_extract_tar_gz(package_path, tmp_dir)
                    dataset_root = _find_package_root(tmp_dir)
                    manifest = _read_manifest(dataset_root)
                    sources.append(
                        {
                            "package_name": package_path.name,
                            "batch_id": sub_batch_id,
                            "device_id": _pick_text(manifest, ["device_id", "equipment_id", "edge_device_id"], device_id),
                            "customer_id": _pick_text(manifest, ["customer_id", "cust_id", "user_id"], customer_id),
                            "captured_at": captured_at,
                            "manifest": manifest,
                        }
                    )
                    for folder in IMAGE_DIR_NAMES | LABEL_DIR_NAMES:
                        if (dataset_root / folder).is_dir():
                            _copy_dir_contents(dataset_root / folder, raw_dir / folder)
                    for filename in MANIFEST_NAMES | {"collector_meta.jsonl"}:
                        if (dataset_root / filename).is_file() and not (raw_dir / filename).exists():
                            shutil.copy2(dataset_root / filename, raw_dir / filename)

            merged_manifest = {
                "schema_version": "visionops_server_merged_manifest_v1",
                "is_merged": True,
                "batch_id": batch_id,
                "device_id": first_device_id,
                "customer_id": first_customer_id,
                "captured_at": first_captured_at,
                "merged_at": datetime.now().isoformat(timespec="seconds"),
                "source_package_count": len(package_paths),
                "source_packages": [path.name for path in package_paths],
                "sources": sources,
            }
            _write_json(raw_dir / "manifest.json", merged_manifest)
            image_count, label_count, total_files = _count_dataset_files(raw_dir)
            now = _now_ms()
            meta = {
                "schema_version": "1.0",
                "batch_id": batch_id,
                "device_id": first_device_id,
                "customer_id": first_customer_id,
                "captured_at": first_captured_at,
                "task_type": "unassigned",
                "source": "incoming_tar_gz_merged",
                "source_packages": [path.name for path in package_paths],
                "status": "extracted",
                "image_count": image_count,
                "label_count": label_count,
                "total_files": total_files,
                "batch_path": str(batch_dir),
                "raw_path": str(raw_dir),
                "images_path": _best_existing_dir(raw_dir, ["images", "all_images", "positive"]),
                "labels_path": _best_existing_dir(raw_dir, ["labels", "labels_auto"]),
                "manifest_path": str(raw_dir / "manifest.json"),
                "manifest": merged_manifest,
                "created_at_ms": now,
                "updated_at_ms": now,
                "review_note": "",
            }
            self._store(batch_id).write(meta)
            processed_paths = [_move_package(path, self.processed_root) for path in package_paths if path.exists()]
            meta["processed_package_paths"] = [str(path) for path in processed_paths]
            self._store(batch_id).write(meta)
            return meta
        except Exception:
            for package_path in package_paths:
                if package_path.exists():
                    _move_package(package_path, self.failed_root)
            raise

    # ------------------------------------------------------------------
    # Backward-compatible direct upload flow used by tests and simple API clients
    # ------------------------------------------------------------------
    def create_from_zip(
        self,
        zip_path: Path,
        *,
        device_id: str = "unknown-device",
        task_type: str = "unassigned",
        source: str = "upload_zip",
    ) -> dict[str, Any]:
        if task_type != "unassigned" and task_type not in self.allowed_task_types:
            raise ValueError(f"不支持的 task_type: {task_type}")
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("上传文件不是有效 zip")
        batch_id = f"batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        batch_dir = self.batches_root / batch_id
        raw_dir = batch_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=False)
        _safe_extract_zip(zip_path, raw_dir)
        image_count, label_count, total_files = _count_dataset_files(raw_dir)
        manifest = _read_manifest(raw_dir)
        now = _now_ms()
        meta = {
            "schema_version": "1.0",
            "batch_id": batch_id,
            "device_id": device_id or "unknown-device",
            "customer_id": _pick_text(manifest, ["customer_id", "cust_id", "user_id"], "unknown-customer"),
            "captured_at": _pick_text(manifest, ["created_at", "captured_at"], "unknown_time"),
            "task_type": task_type,
            "source": source,
            "status": "extracted",
            "image_count": image_count,
            "label_count": label_count,
            "total_files": total_files,
            "batch_path": str(batch_dir),
            "raw_path": str(raw_dir),
            "images_path": _best_existing_dir(raw_dir, ["images", "all_images", "positive"]),
            "labels_path": _best_existing_dir(raw_dir, ["labels", "labels_auto"]),
            "manifest_path": str(raw_dir / "manifest.json") if (raw_dir / "manifest.json").exists() else "",
            "manifest": manifest,
            "created_at_ms": now,
            "updated_at_ms": now,
            "review_note": "",
        }
        self._store(batch_id).write(meta)
        return meta

    def create_from_upload_archive(self, archive_path: Path, *, filename: str = "") -> dict[str, Any]:
        """兼容 API 直传 zip/tar.gz。Web 主流程仍建议先放 incoming 再处理。"""
        filename = filename or archive_path.name
        if filename.endswith(".tar.gz") or tarfile.is_tarfile(archive_path):
            incoming_path = self.incoming_root / _unique_filename(self.incoming_root, filename if filename.endswith(".tar.gz") else f"{filename}.tar.gz")
            shutil.copy2(archive_path, incoming_path)
            return self._process_one_tar_package(incoming_path)
        return self.create_from_zip(archive_path)

    def list_batches(self) -> list[dict[str, Any]]:
        result = []
        for batch_dir in sorted([entry for entry in self.batches_root.iterdir() if entry.is_dir()], key=lambda x: x.name):
            meta = self._store(batch_dir.name).read()
            if isinstance(meta, dict) and meta.get("batch_id"):
                result.append(meta)
        return result

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        meta = self._store(_safe_id(batch_id)).read()
        if not isinstance(meta, dict) or not meta.get("batch_id"):
            raise FileNotFoundError(f"批次不存在: {batch_id}")
        return meta

    def set_status(self, batch_id: str, status: str, note: str = "", task_type: str | None = None) -> dict[str, Any]:
        if status not in {"extracted", "accepted", "rejected", "failed"}:
            raise ValueError(f"非法批次状态: {status}")
        batch_id = _safe_id(batch_id)
        meta = self.get_batch(batch_id)
        if task_type:
            task_type = str(task_type).strip()
            if task_type not in self.allowed_task_types:
                raise ValueError(f"不支持的 task_type: {task_type}")
            meta["task_type"] = task_type
        meta["status"] = status
        meta["review_note"] = note
        meta["updated_at_ms"] = _now_ms()
        self._store(batch_id).write(meta)
        return meta

    def _resolve_incoming_package(self, value: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("空压缩包路径")
        candidate = Path(raw)
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (self.incoming_root / raw).resolve()
        try:
            path.relative_to(self.incoming_root.resolve())
        except ValueError as exc:
            raise ValueError(f"非法压缩包路径: {value}") from exc
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"压缩包不存在: {value}")
        if not path.name.endswith(".tar.gz"):
            raise ValueError(f"只支持 .tar.gz 上传包: {path.name}")
        return path

    def _store(self, batch_id: str) -> JsonStore:
        return JsonStore(self.batches_root / batch_id / "batch.json", default={})


def parse_package_name(package_name: str) -> tuple[str, str, str, str]:
    name = package_name[:-7] if package_name.endswith(".tar.gz") else Path(package_name).stem
    parts = name.split("_")
    if len(parts) >= 4:
        device_id = safe_name(parts[0])
        customer_id = safe_name(parts[1])
        captured_at = f"{parts[2]}_{parts[3]}"
        return safe_name(name), device_id, customer_id, captured_at
    batch_id = safe_name(name)
    return batch_id, "unknown-device", "unknown-customer", "unknown_time"


def safe_name(name: str) -> str:
    value = str(name or "").strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.\-]+", "_", value)
    return value.strip("._") or "unknown"


def _safe_extract_tar_gz(package_path: Path, target_dir: Path) -> None:
    if not tarfile.is_tarfile(package_path):
        raise ValueError(f"不是合法 tar.gz: {package_path.name}")
    with tarfile.open(package_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (target_dir / member.name).resolve()
            try:
                target.relative_to(target_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"tar.gz 内存在非法路径: {member.name}") from exc
        try:
            tar.extractall(target_dir, filter="data")
        except TypeError:
            tar.extractall(target_dir)


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue
            target = (target_dir / name).resolve()
            try:
                target.relative_to(target_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"zip 内存在非法路径: {name}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_package_root(extracted_dir: Path) -> Path:
    candidates = [extracted_dir]
    candidates.extend([p for p in extracted_dir.rglob("*") if p.is_dir()])
    candidates = sorted(candidates, key=lambda p: len(p.relative_to(extracted_dir).parts))
    for path in candidates:
        has_manifest = any((path / name).is_file() for name in MANIFEST_NAMES)
        has_images = any((path / name).is_dir() for name in IMAGE_DIR_NAMES)
        if has_manifest or has_images:
            return path
    raise FileNotFoundError("未在上传包中找到 manifest.json 或 images/all_images 目录")


def _count_dataset_files(root: Path) -> tuple[int, int, int]:
    image_count = 0
    label_count = 0
    total_files = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_count += 1
        elif suffix in LABEL_EXTENSIONS or any(part in LABEL_DIR_NAMES for part in path.parts):
            label_count += 1
    return image_count, label_count, total_files


def _read_manifest(root: Path) -> dict[str, Any]:
    for name in MANIFEST_NAMES:
        path = root / name
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
            except Exception:
                return {}
    return {}


def _pick_text(document: dict[str, Any], keys: list[str], default: str) -> str:
    for key in keys:
        value = document.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _best_existing_dir(root: Path, names: list[str]) -> str:
    for name in names:
        if (root / name).is_dir():
            return str(root / name)
    return ""


def _copy_dir_contents(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _move_package(package_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / package_path.name
    if target.exists():
        target = target_dir / _unique_filename(target_dir, package_path.name)
    shutil.move(str(package_path), str(target))
    return target


def _unique_filename(root: Path, filename: str) -> str:
    candidate = filename
    stem = filename[:-7] if filename.endswith(".tar.gz") else Path(filename).stem
    suffix = ".tar.gz" if filename.endswith(".tar.gz") else Path(filename).suffix
    counter = 1
    while (root / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _safe_id(value: str) -> str:
    value = str(value or "").strip()
    safe = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法 ID")
    return safe
