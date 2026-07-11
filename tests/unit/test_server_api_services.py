from __future__ import annotations

import json
import tarfile
import time
import zipfile
from pathlib import Path

from apps.server_api.backend.services.dataset_service import DatasetService
from apps.server_api.backend.services.device_service import DeviceService
from apps.server_api.backend.services.ingest_service import BatchService
from apps.server_api.backend.services.model_package_service import ModelPackageService, make_model_yaml
from apps.server_api.backend.services.training_job_service import TrainingJobService


def _sample_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images/a.jpg", b"fake-image")
        archive.writestr("labels/a.txt", "0 0.5 0.5 0.1 0.1\n")
    return path


def _sample_tar_gz(path: Path) -> Path:
    root = path.parent / "package_root"
    (root / "images").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"device_id": "rk3576-001", "customer_id": "package-test", "counts": {"all": 1}}), encoding="utf-8")
    (root / "images" / "a.jpg").write_bytes(b"fake-image")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(root / "manifest.json", arcname="manifest.json")
        archive.add(root / "images" / "a.jpg", arcname="images/a.jpg")
    return path


def test_incoming_tar_gz_process_parses_name_and_manifest(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    package = _sample_tar_gz(incoming / "rk3576-001_package-test_20260707_085333.tar.gz")
    batch_service = BatchService(tmp_path / "batches", ("detection",), incoming_root=incoming)
    packages = batch_service.list_incoming_packages()
    assert [item["name"] for item in packages] == [package.name]
    batch = batch_service.process_incoming_packages([package.name])
    assert batch["batch_id"] == "rk3576-001_package-test_20260707_085333"
    assert batch["device_id"] == "rk3576-001"
    assert batch["customer_id"] == "package-test"
    assert batch["task_type"] == "unassigned"
    assert batch["status"] == "extracted"
    assert batch["image_count"] == 1
    assert batch["manifest"]["counts"]["all"] == 1
    assert not package.exists()
    assert (incoming / "processed" / package.name).exists()


def test_batch_upload_accept_and_dataset_build(tmp_path: Path) -> None:
    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    assert batch["status"] == "extracted"
    assert batch["image_count"] == 1
    assert batch["label_count"] == 1

    accepted = batch_service.set_status(batch["batch_id"], "accepted")
    assert accepted["status"] == "accepted"

    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])
    assert dataset["image_count"] == 1
    assert dataset["batch_ids"] == [batch["batch_id"]]


def test_model_yaml_matches_v3_package_contract() -> None:
    document = make_model_yaml(
        model_id="demo-det",
        model_name="demo",
        version="0.1.0",
        task_type="detection",
        classes=[{"id": 0, "name": "tube"}],
        target_platform="rk3576",
    )
    assert document["model_id"] == "demo-det"
    assert document["model"]["format"] == "rknn"
    assert document["classes"] == [{"id": 0, "name": "tube"}]
    assert document["class_names"] == ["tube"]
    assert document["runtime"]["color"] == "rgb"


def test_model_package_publish_copies_only_runtime_files(tmp_path: Path) -> None:
    service = ModelPackageService(tmp_path / "packages")
    package = service.create_mock_package(model_id="demo-det", model_name="demo", task_type="detection")
    assert package["status"] == "ready"
    published = service.publish_package("demo-det", publish_root=tmp_path / "publish")
    target = Path(published["publish_path"])
    assert sorted(p.name for p in target.iterdir()) == ["model.rknn", "model.yaml"]


def test_training_job_mock_runner_generates_model_package(tmp_path: Path) -> None:
    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    batch_service.set_status(batch["batch_id"], "accepted")
    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])
    package_service = ModelPackageService(tmp_path / "packages")
    job_service = TrainingJobService(tmp_path / "jobs", dataset_service, package_service)

    job = job_service.create_job({"dataset_id": dataset["dataset_id"], "task_type": "detection", "run_inline": True, "runner": "mock"})
    assert job["status"] == "success"
    assert job["output_model_package"]
    assert package_service.get_package(job["output_model_package"])["status"] == "ready"
    assert "模型包已生成" in job_service.get_logs(job["job_id"])


def test_device_registry_assign_model(tmp_path: Path) -> None:
    service = DeviceService(tmp_path / "devices.json")
    device = service.upsert_device({"device_id": "lb3576-dev", "device_type": "lb3576"})
    assert device["device_id"] == "lb3576-dev"
    assigned = service.assign_model("lb3576-dev", "demo-det")
    assert assigned["target_model"] == "demo-det"
    assert service.get_device("lb3576-dev")["sync_status"] == "assigned"


def test_model_package_delete_removes_package_dir(tmp_path: Path) -> None:
    service = ModelPackageService(tmp_path / "packages")
    package = service.create_mock_package(model_id="demo-delete", model_name="demo", task_type="detection")
    package_path = Path(package["package_path"])
    assert package_path.exists()

    deleted = service.delete_package("demo-delete")
    assert deleted["deleted"] is True
    assert not package_path.exists()


def test_training_job_delete_removes_job_dir(tmp_path: Path) -> None:
    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    batch_service.set_status(batch["batch_id"], "accepted")
    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])
    package_service = ModelPackageService(tmp_path / "packages")
    job_service = TrainingJobService(tmp_path / "jobs", dataset_service, package_service)

    job = job_service.create_job({"dataset_id": dataset["dataset_id"], "task_type": "detection", "run_inline": True, "runner": "mock"})
    job_path = Path(job["job_path"])
    assert job_path.exists()

    deleted = job_service.delete_job(job["job_id"])
    assert deleted["deleted"] is True
    assert not job_path.exists()


def test_classification_dataset_builds_ultralytics_folder_layout(tmp_path: Path) -> None:
    batch_service = BatchService(tmp_path / "batches", ("detection", "classification"))
    raw_zip = tmp_path / "cls_upload.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("ok/a.jpg", b"fake-ok")
        archive.writestr("ng/b.jpg", b"fake-ng")
        archive.writestr("ng/c.jpg", b"fake-ng-2")
    batch = batch_service.create_from_zip(raw_zip, device_id="edge-dev", task_type="classification")
    batch_service.set_status(batch["batch_id"], "accepted", task_type="classification")

    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="classification", batch_ids=[batch["batch_id"]])

    assert dataset["task_type"] == "classification"
    assert dataset["image_count"] == 3
    assert dataset["class_count"] == 2
    cls_root = Path(dataset["yolo_dataset_path"])
    assert (cls_root / "train" / "ok").is_dir()
    assert (cls_root / "train" / "ng").is_dir()
    assert (cls_root / "val").is_dir()
    assert (cls_root / "data.yaml").exists()


def test_dataset_materialization_hardlinks_images_but_copies_labels(tmp_path: Path) -> None:
    from apps.server_api.backend.services.storage_utils import same_inode

    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    batch_service.set_status(batch["batch_id"], "accepted")
    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])

    source_image = Path(batch["images_path"]) / "a.jpg"
    source_label = Path(batch["raw_path"]) / "labels" / "a.txt"
    dataset_root = Path(dataset["yolo_dataset_path"])
    dataset_image = next((dataset_root / "images").rglob("*.jpg"))
    dataset_label = next((dataset_root / "labels").rglob("*.txt"))

    assert same_inode(source_image, dataset_image)
    assert not same_inode(source_label, dataset_label)
    assert dataset["storage_mode"] == "hardlink_images_copy_labels"
    assert dataset["storage"]["estimated_saved_bytes"] == dataset["storage"]["image_bytes"]
    assert dataset["storage"]["physical_image_bytes_added"] == 0


def test_pipeline_preprocess_reuses_dataset_without_job_copy(tmp_path: Path) -> None:
    from training.pipeline.common import PipelineContext
    from training.pipeline.stages import preprocess

    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    batch_service.set_status(batch["batch_id"], "accepted")
    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])
    dataset["batches"] = json.loads(Path(dataset["batches_json"]).read_text(encoding="utf-8"))

    job_dir = tmp_path / "jobs" / "job-1"
    work_dir = job_dir / "work"
    output_dir = job_dir / "outputs"
    work_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    ctx = PipelineContext(
        project_root=tmp_path,
        job={"job_id": "job-1", "task_type": "detection"},
        dataset=dataset,
        job_dir=job_dir,
        work_dir=work_dir,
        output_dir=output_dir,
    )

    report = preprocess.run(ctx)
    assert report["storage_mode"] == "shared_dataset_reference"
    assert report["job_dataset_copy_created"] is False
    assert Path(report["dataset_dir"]).resolve() == Path(dataset["yolo_dataset_path"]).resolve()
    assert not (work_dir / "yolo_dataset").exists()


def test_active_training_reference_blocks_dataset_delete(tmp_path: Path) -> None:
    batch_service = BatchService(tmp_path / "batches", ("detection",))
    batch = batch_service.create_from_zip(_sample_zip(tmp_path / "upload.zip"), device_id="edge-dev", task_type="detection")
    batch_service.set_status(batch["batch_id"], "accepted")
    dataset_service = DatasetService(tmp_path / "datasets", batch_service)
    dataset = dataset_service.build_dataset(task_type="detection", batch_ids=[batch["batch_id"]])

    job_dir = tmp_path / "jobs" / "active-job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps({"job_id": "active-job", "dataset_id": dataset["dataset_id"], "status": "running"}),
        encoding="utf-8",
    )
    dataset_service.acquire_training_reference(dataset["dataset_id"], "active-job", job_dir)

    try:
        dataset_service.delete_dataset(dataset["dataset_id"])
    except ValueError as error:
        assert "正在被训练任务引用" in str(error)
    else:
        raise AssertionError("active dataset reference should block deletion")

    dataset_service.release_training_reference(dataset["dataset_id"], "active-job")
    deleted = dataset_service.delete_dataset(dataset["dataset_id"])
    assert deleted["status"] == "deleted"
