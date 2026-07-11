#!/usr/bin/env python3
"""Deduplicate existing VisionOps v3 server datasets.

New dataset builds already hard-link immutable images from ``batches`` and new
training jobs directly reference ``datasets``. This migration tool applies the
same policy to existing ``server_data`` contents.

The default mode is dry-run. Pass ``--apply`` to replace duplicate dataset
images with hard links and remove completed job-local dataset copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ACTIVE_JOB_STATES = {"pending", "running"}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def source_images_from_batches(dataset_dir: Path) -> list[Path]:
    batches = read_json(dataset_dir / "batches.json", [])
    if not isinstance(batches, list):
        return []
    result: list[Path] = []
    seen: set[str] = set()
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        roots: list[Path] = []
        images_path = str(batch.get("images_path") or "").strip()
        raw_path = str(batch.get("raw_path") or "").strip()
        if images_path:
            roots.append(Path(images_path))
        if raw_path:
            roots.append(Path(raw_path))
        for root in roots:
            for path in image_files(root):
                # Exclude temporary annotation/training derivatives. The
                # reviewed source image itself remains under raw/images,
                # raw/all_images or class folders.
                if any(part in {"quick_train", "previews", "candidates", "roi_classification_sessions"} for part in path.parts):
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    result.append(path)
    return result


def choose_source(dst: Path, candidates: list[Path], hash_cache: dict[str, str]) -> Path | None:
    original_name = dst.name.split("__", 1)[-1]
    size = dst.stat().st_size
    matches = [path for path in candidates if path.name == original_name and path.stat().st_size == size]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        matches = [path for path in candidates if path.stat().st_size == size]
    if not matches:
        return None
    dst_key = str(dst)
    dst_hash = hash_cache.setdefault(dst_key, sha256(dst))
    equal: list[Path] = []
    for path in matches:
        key = str(path)
        if hash_cache.setdefault(key, sha256(path)) == dst_hash:
            equal.append(path)
            if len(equal) > 1:
                break
    return equal[0] if len(equal) == 1 else None


def replace_with_hardlink(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.hardlink-tmp-{os.getpid()}")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.link(str(src.resolve()), str(tmp))
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def optimize_datasets(data_root: Path, apply: bool) -> dict[str, int]:
    stats = {
        "datasets_seen": 0,
        "dataset_images_seen": 0,
        "already_hardlinked": 0,
        "images_convertible": 0,
        "images_converted": 0,
        "images_unmatched": 0,
        "estimated_saved_bytes": 0,
    }
    datasets_root = data_root / "datasets"
    if not datasets_root.is_dir():
        return stats

    hash_cache: dict[str, str] = {}
    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        meta_path = dataset_dir / "dataset.json"
        meta = read_json(meta_path, {})
        if not isinstance(meta, dict) or not meta.get("dataset_id"):
            continue
        sources = source_images_from_batches(dataset_dir)
        materialized_root = Path(str(meta.get("yolo_dataset_path") or ""))
        if not materialized_root.is_dir():
            continue
        destinations = image_files(materialized_root)
        stats["datasets_seen"] += 1
        dataset_hardlinks = 0
        dataset_copies = 0
        dataset_saved = 0
        for dst in destinations:
            stats["dataset_images_seen"] += 1
            src = choose_source(dst, sources, hash_cache)
            if src is None:
                stats["images_unmatched"] += 1
                dataset_copies += 1
                continue
            if same_inode(src, dst):
                stats["already_hardlinked"] += 1
                dataset_hardlinks += 1
                dataset_saved += dst.stat().st_size
                continue
            stats["images_convertible"] += 1
            size = dst.stat().st_size
            if apply:
                try:
                    replace_with_hardlink(src, dst)
                except OSError:
                    stats["images_unmatched"] += 1
                    dataset_copies += 1
                    continue
                stats["images_converted"] += 1
            dataset_hardlinks += 1
            dataset_saved += size
            stats["estimated_saved_bytes"] += size

        if apply and destinations:
            storage = dict(meta.get("storage") or {})
            storage.update(
                {
                    "schema_version": "1.0",
                    "image_files": len(destinations),
                    "image_bytes": sum(path.stat().st_size for path in destinations),
                    "hardlinked_images": dataset_hardlinks,
                    "copied_images": dataset_copies,
                    "estimated_saved_bytes": dataset_saved,
                    "storage_mode": "hardlink_images_copy_labels" if dataset_hardlinks == len(destinations) else "mixed_hardlink_copy",
                    "migrated_by": "tools.storage.optimize_server_data",
                }
            )
            meta["storage"] = storage
            meta["storage_mode"] = storage["storage_mode"]
            write_json(meta_path, meta)
    return stats


def directory_logical_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0


def canonical_dataset_paths(data_root: Path, dataset_id: str, task_type: str) -> tuple[Path, Path]:
    dataset_dir = data_root / "datasets" / dataset_id
    meta = read_json(dataset_dir / "dataset.json", {})
    materialized = Path(str(meta.get("yolo_dataset_path") or ""))
    if not materialized.is_dir():
        return Path(), Path()
    if str(task_type).lower() in {"classification", "cls", "classify"}:
        return materialized, materialized / "data.yaml"
    data_yaml = Path(str(meta.get("data_yaml") or materialized / "data.yaml"))
    return materialized, data_yaml


def optimize_completed_jobs(data_root: Path, apply: bool) -> dict[str, int]:
    stats = {
        "jobs_seen": 0,
        "active_jobs_skipped": 0,
        "job_dataset_dirs_found": 0,
        "job_dataset_dirs_removed": 0,
        "estimated_reclaimed_bytes": 0,
    }
    jobs_root = data_root / "jobs"
    if not jobs_root.is_dir():
        return stats
    for job_dir in sorted(path for path in jobs_root.iterdir() if path.is_dir()):
        job = read_json(job_dir / "job.json", {})
        if not isinstance(job, dict) or not job.get("job_id"):
            continue
        stats["jobs_seen"] += 1
        if str(job.get("status") or "") in ACTIVE_JOB_STATES:
            stats["active_jobs_skipped"] += 1
            continue
        dataset_id = str(job.get("dataset_id") or "")
        canonical, data_yaml = canonical_dataset_paths(data_root, dataset_id, str(job.get("task_type") or ""))
        if not canonical.is_dir():
            continue
        for name in ("yolo_dataset", "cls_dataset"):
            duplicate = job_dir / "work" / name
            if not duplicate.is_dir() or duplicate.resolve() == canonical.resolve():
                continue
            stats["job_dataset_dirs_found"] += 1
            bytes_used = directory_logical_bytes(duplicate)
            stats["estimated_reclaimed_bytes"] += bytes_used
            if not apply:
                continue
            report_path = job_dir / "outputs" / "preprocess_report.json"
            report = read_json(report_path, {})
            if isinstance(report, dict):
                report.update(
                    {
                        "dataset_dir": str(canonical),
                        "data_path": str(canonical if name == "cls_dataset" else data_yaml),
                        "data_yaml": str(data_yaml),
                        "storage_mode": "shared_dataset_reference",
                        "source_dataset_path": str(canonical.parent),
                        "job_dataset_copy_created": False,
                    }
                )
                write_json(report_path, report)
            shutil.rmtree(duplicate)
            stats["job_dataset_dirs_removed"] += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate VisionOps v3 server_data datasets and completed jobs")
    parser.add_argument("--data-root", default="server_data", help="VisionOps server_data path")
    parser.add_argument("--apply", action="store_true", help="Apply changes; default is dry-run")
    parser.add_argument("--skip-jobs", action="store_true", help="Do not remove completed job-local dataset copies")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    result = {
        "mode": "apply" if args.apply else "dry-run",
        "data_root": str(data_root),
        "datasets": optimize_datasets(data_root, args.apply),
        "jobs": {} if args.skip_jobs else optimize_completed_jobs(data_root, args.apply),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
