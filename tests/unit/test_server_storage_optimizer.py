from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.storage.optimize_server_data import optimize_completed_jobs, optimize_datasets, same_inode


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_optimizer_hardlinks_existing_dataset_and_removes_completed_job_copy(tmp_path: Path) -> None:
    data_root = tmp_path / "server_data"
    batch_id = "batch-1"
    dataset_id = "dataset-1"
    source = data_root / "batches" / batch_id / "raw" / "images" / "a.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")

    dataset_dir = data_root / "datasets" / dataset_id
    dataset_image = dataset_dir / "yolo_dataset" / "images" / "train" / f"{batch_id}__a.jpg"
    dataset_image.parent.mkdir(parents=True)
    shutil.copy2(source, dataset_image)
    data_yaml = dataset_dir / "yolo_dataset" / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")
    _write_json(
        dataset_dir / "dataset.json",
        {
            "dataset_id": dataset_id,
            "task_type": "detection",
            "dataset_path": str(dataset_dir),
            "yolo_dataset_path": str(dataset_dir / "yolo_dataset"),
            "data_yaml": str(data_yaml),
        },
    )
    _write_json(
        dataset_dir / "batches.json",
        [{"batch_id": batch_id, "raw_path": str(source.parents[1]), "images_path": str(source.parent)}],
    )

    job_dir = data_root / "jobs" / "job-1"
    duplicate = job_dir / "work" / "yolo_dataset"
    (duplicate / "images" / "train").mkdir(parents=True)
    shutil.copy2(source, duplicate / "images" / "train" / "a.jpg")
    _write_json(job_dir / "job.json", {"job_id": "job-1", "dataset_id": dataset_id, "task_type": "detection", "status": "success"})
    _write_json(job_dir / "outputs" / "preprocess_report.json", {"dataset_dir": str(duplicate)})

    dry = optimize_datasets(data_root, apply=False)
    assert dry["images_convertible"] == 1
    assert not same_inode(source, dataset_image)

    applied = optimize_datasets(data_root, apply=True)
    assert applied["images_converted"] == 1
    assert same_inode(source, dataset_image)

    jobs = optimize_completed_jobs(data_root, apply=True)
    assert jobs["job_dataset_dirs_removed"] == 1
    assert not duplicate.exists()
    report = json.loads((job_dir / "outputs" / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["storage_mode"] == "shared_dataset_reference"
    assert Path(report["dataset_dir"]) == dataset_dir / "yolo_dataset"
