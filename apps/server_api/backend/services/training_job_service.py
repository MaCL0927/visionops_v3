"""训练任务管理：v3 stage 化训练流水线。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .dataset_service import DatasetService
from .model_package_service import ModelPackageService

STAGES = ["preprocess", "train", "evaluate", "export_onnx", "convert_rknn", "package_v3_model"]
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TrainingJobService:
    def __init__(self, jobs_root: Path, dataset_service: DatasetService, model_package_service: ModelPackageService, *, target_platform: str = "rk3576") -> None:
        self.jobs_root = Path(jobs_root)
        self.dataset_service = dataset_service
        self.model_package_service = model_package_service
        self.target_platform = target_platform
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError("缺少 dataset_id")
        dataset = self.dataset_service.get_dataset(dataset_id)
        task_type = normalize_task(str(payload.get("task_type") or dataset.get("task_type") or "detection"))
        source_device_id = str(dataset.get("source_device_id") or _dataset_name_part(dataset_id, 0, "multi-device"))
        source_customer_id = str(dataset.get("source_customer_id") or _dataset_name_part(dataset_id, 1, "multi-customer"))
        job_time = time.strftime("%Y%m%d_%H%M%S")
        job_id = _unique_job_id(
            self.jobs_root,
            _safe_id(f"{source_device_id}_{source_customer_id}_{_task_for_name(task_type)}_job_{job_time}"),
        )
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        now = int(time.time() * 1000)
        runner = str(payload.get("runner") or payload.get("mode") or "pipeline").strip().lower()
        job = {
            "schema_version": "1.0",
            "job_id": job_id,
            "runner": runner,
            "task_type": task_type,
            "dataset_id": dataset_id,
            "preset_name": payload.get("preset_name") or "default",
            "status": "pending",
            "current_stage": "pending",
            "stages": STAGES,
            "epochs": int(payload.get("epochs", 100)),
            "batch_size": int(payload.get("batch_size", 4)),
            "imgsz": int(payload.get("imgsz", 640)),
            "device": payload.get("device") if payload.get("device") not in {None, ""} else "0",
            "target_platform": payload.get("target_platform") or self.target_platform,
            "source_device_id": source_device_id,
            "source_customer_id": source_customer_id,
            "pretrained_model": payload.get("pretrained_model") or _default_pretrained(task_type),
            "yolo_cmd": payload.get("yolo_cmd") or "yolo",
            "conda_executable": payload.get("conda_executable") or os.environ.get("VISIONOPS_CONDA_EXE") or os.environ.get("CONDA_EXE") or "conda",
            "onnx_conda_env": payload.get("onnx_conda_env") if payload.get("onnx_conda_env") is not None else os.environ.get("VISIONOPS_ONNX_CONDA_ENV", "pt2onnx"),
            "rknn_conda_env": payload.get("rknn_conda_env") if payload.get("rknn_conda_env") is not None else os.environ.get("VISIONOPS_RKNN_CONDA_ENV", "rknn311"),
            "amp": _bool_value(payload.get("amp", True)),
            "workers": int(payload.get("workers", 4)),
            "do_quantization": _bool_value(payload.get("do_quantization", True)),
            "onnx_opset": int(payload.get("onnx_opset", 12)),
            "onnx_simplify": bool(payload.get("onnx_simplify", True)),
            "conf_threshold": float(payload.get("conf_threshold", 0.25)),
            "iou_threshold": float(payload.get("iou_threshold", 0.45)),
            "max_det": int(payload.get("max_det", 100)),
            "mlflow_run_id": None,
            "output_model_package": None,
            "job_path": str(job_dir),
            "work_dir": str(job_dir / "work"),
            "output_dir": str(job_dir / "outputs"),
            "log_path": str(job_dir / f"{job_id}.log"),
            "job_config_path": str(job_dir / "job_config.json"),
            "pipeline_status_path": str(job_dir / "pipeline_status.json"),
            "dataset_json": str(Path(str(dataset.get("dataset_json") or dataset.get("dataset_path"))) / "dataset.json")
            if dataset.get("dataset_json") is None else str(dataset.get("dataset_json")),
            "dataset_batches_json": str(Path(str(dataset.get("dataset_path"))) / "batches.json"),
            "model_packages_root": str(self.model_package_service.model_packages_root),
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "真实训练流水线：preprocess -> train -> evaluate -> export_onnx -> convert_rknn -> package_v3_model。",
        }
        if payload.get("model_id"):
            job["model_id"] = str(payload["model_id"])
        if payload.get("model_name"):
            job["model_name"] = str(payload["model_name"])
        self._write_job(job_id, job)
        _write_json(job_dir / "job_config.json", job)
        run_inline = bool(payload.get("run_inline", False))
        if runner == "mock":
            if run_inline:
                self._run_mock_pipeline(job_id)
            else:
                thread = threading.Thread(target=self._run_mock_pipeline, args=(job_id,), daemon=True)
                self._threads[job_id] = thread
                thread.start()
        else:
            if run_inline:
                self._run_real_pipeline(job_id)
            else:
                thread = threading.Thread(target=self._run_real_pipeline, args=(job_id,), daemon=True)
                self._threads[job_id] = thread
                thread.start()
        return self.get_job(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        result = []
        for job_dir in sorted([entry for entry in self.jobs_root.iterdir() if entry.is_dir()], key=lambda x: x.name):
            meta = _read_json(job_dir / "job.json", {})
            if meta.get("job_id"):
                # Refresh from pipeline status if this job is still running.
                if meta.get("status") == "running":
                    meta = self._sync_pipeline_status(meta)
                result.append(meta)
        return result

    def get_job(self, job_id: str) -> dict[str, Any]:
        job_id = _safe_id(job_id)
        job = _read_json(self.jobs_root / job_id / "job.json", {})
        if not job.get("job_id"):
            raise FileNotFoundError(f"训练任务不存在: {job_id}")
        if job.get("status") == "running":
            job = self._sync_pipeline_status(job)
        return job

    def get_logs(self, job_id: str, tail_bytes: int = 40000) -> str:
        job = self.get_job(job_id)
        log_path = Path(job["log_path"])
        if not log_path.exists():
            return ""
        data = log_path.read_bytes()
        return data[-tail_bytes:].decode("utf-8", errors="replace")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.get("status") in {"success", "failed", "canceled"}:
            return job
        with self._lock:
            proc = self._processes.get(job["job_id"])
        if proc is not None and proc.poll() is None:
            proc.terminate()
        job["status"] = "canceled"
        job["current_stage"] = "canceled"
        job["updated_at_ms"] = int(time.time() * 1000)
        self._write_job(job["job_id"], job)
        self._append_log(job["job_id"], "任务已标记为 canceled。")
        return job

    def delete_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._lock:
            proc = self._processes.get(job["job_id"])
        if proc is not None and proc.poll() is None:
            raise ValueError("训练任务仍在运行，请先取消后再删除")
        if job.get("status") == "running":
            raise ValueError("训练任务状态仍为 running，请先取消或等待结束后再删除")
        job_dir = Path(job.get("job_path") or (self.jobs_root / job["job_id"]))
        if not job_dir.is_dir():
            raise FileNotFoundError(f"训练任务目录不存在: {job_dir}")
        shutil.rmtree(job_dir)
        job["deleted"] = True
        job["deleted_at_ms"] = int(time.time() * 1000)
        return job

    def _run_real_pipeline(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            job["status"] = "running"
            job["current_stage"] = "preprocess"
            job["started_at_ms"] = int(time.time() * 1000)
            job["updated_at_ms"] = job["started_at_ms"]
            self._write_job(job_id, job)
            self._append_log(job_id, "训练流水线启动。")
            command = [
                sys.executable,
                "-m",
                "training.pipeline.run_pipeline",
                "--job-config",
                job["job_config_path"],
                "--output-dir",
                job["output_dir"],
                "--project-root",
                str(PROJECT_ROOT),
            ]
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("[PIPELINE_CMD] " + " ".join(command) + "\n")
                log_file.flush()
                proc = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )
                with self._lock:
                    self._processes[job_id] = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self._sync_pipeline_status_by_id(job_id)
                return_code = proc.wait()
            with self._lock:
                self._processes.pop(job_id, None)
            self._finish_from_pipeline_result(job_id, return_code)
        except Exception as error:  # pragma: no cover - background thread safety
            self._mark_failed(job_id, error)

    def _finish_from_pipeline_result(self, job_id: str, return_code: int) -> None:
        job = self.get_job(job_id)
        status = _read_json(Path(job["pipeline_status_path"]), {}) or {}
        package_report = _read_json(Path(job["output_dir"]) / "package_v3_model_report.json", {}) or {}
        if return_code == 0 and status.get("status") == "success":
            job["status"] = "success"
            job["current_stage"] = "done"
            job["output_model_package"] = status.get("output_model_package") or package_report.get("model_id")
            job["mlflow_run_id"] = status.get("mlflow_run_id")
            job.pop("error", None)
            self._append_log(job_id, f"模型包已生成: {job.get('output_model_package')}")
        else:
            job["status"] = "failed"
            job["current_stage"] = status.get("current_stage") or "failed"
            job["error"] = f"训练流水线失败，returncode={return_code}"
            self._append_log(job_id, f"[ERROR] {job['error']}")
        job["finished_at_ms"] = int(time.time() * 1000)
        job["updated_at_ms"] = job["finished_at_ms"]
        self._write_job(job_id, job)

    def _sync_pipeline_status_by_id(self, job_id: str) -> None:
        try:
            job = _read_json(self.jobs_root / _safe_id(job_id) / "job.json", {})
            if job:
                self._sync_pipeline_status(job)
        except Exception:
            pass

    def _sync_pipeline_status(self, job: dict[str, Any]) -> dict[str, Any]:
        status = _read_json(Path(str(job.get("pipeline_status_path") or "")), {}) or {}
        changed = False
        if status.get("current_stage") and job.get("current_stage") != status.get("current_stage"):
            job["current_stage"] = status.get("current_stage")
            changed = True
        if status.get("output_model_package") and job.get("output_model_package") != status.get("output_model_package"):
            job["output_model_package"] = status.get("output_model_package")
            changed = True
        if status.get("status") == "success" and job.get("status") == "running":
            job["status"] = "success"
            changed = True
        if changed:
            job["updated_at_ms"] = int(time.time() * 1000)
            self._write_job(job["job_id"], job)
        return job

    def _run_mock_pipeline(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            job["status"] = "running"
            job["updated_at_ms"] = int(time.time() * 1000)
            self._write_job(job_id, job)
            for stage in STAGES:
                job = self.get_job(job_id)
                if job.get("status") == "canceled":
                    return
                job["current_stage"] = stage
                job["updated_at_ms"] = int(time.time() * 1000)
                self._write_job(job_id, job)
                self._append_log(job_id, f"[STAGE] {stage} started")
                time.sleep(0.02)
                self._append_log(job_id, f"[STAGE] {stage} finished")
            model_id = f"{job['task_type']}_{job['dataset_id']}_{job_id}"[:96]
            package = self.model_package_service.create_mock_package(
                model_id=model_id,
                model_name=f"{job['task_type']}-{job['dataset_id']}",
                task_type=job["task_type"],
                dataset_id=job["dataset_id"],
                job_id=job_id,
                target_platform=job.get("target_platform") or self.target_platform,
                classes=["object"],
                metrics={"mAP50": None, "source": "mock_training_job"},
                train_config=job,
            )
            job = self.get_job(job_id)
            job["status"] = "success"
            job["current_stage"] = "done"
            job["output_model_package"] = package.get("model_id")
            job["mlflow_run_id"] = f"mock-{job_id}"
            job["updated_at_ms"] = int(time.time() * 1000)
            self._write_job(job_id, job)
            self._append_log(job_id, f"模型包已生成: {package.get('model_id')}")
        except Exception as error:  # pragma: no cover
            self._mark_failed(job_id, error)

    def _mark_failed(self, job_id: str, error: Exception) -> None:
        try:
            job = self.get_job(job_id)
        except Exception:
            return
        job["status"] = "failed"
        job["current_stage"] = job.get("current_stage") or "failed"
        job["error"] = str(error)
        job["finished_at_ms"] = int(time.time() * 1000)
        job["updated_at_ms"] = job["finished_at_ms"]
        self._write_job(job_id, job)
        self._append_log(job_id, f"[ERROR] {error}")

    def _write_job(self, job_id: str, value: dict[str, Any]) -> None:
        job_id = _safe_id(job_id)
        _write_json(self.jobs_root / job_id / "job.json", value)

    def _append_log(self, job_id: str, line: str) -> None:
        try:
            job = self.get_job(job_id)
            log_path = Path(job["log_path"])
        except Exception:
            safe_job_id = _safe_id(job_id)
            log_path = self.jobs_root / safe_job_id / f"{safe_job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def normalize_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg"}:
        return "segmentation"
    if task in {"classification", "cls", "classify"}:
        return "classification"
    return "detection"


def _task_for_name(task_type: str) -> str:
    task = normalize_task(task_type)
    if task == "obb":
        return "obb"
    if task == "segmentation":
        return "seg"
    if task == "classification":
        return "cls"
    return "det"


def _dataset_name_part(dataset_id: str, index: int, fallback: str) -> str:
    parts = str(dataset_id or "").split("_")
    if len(parts) > index and parts[index]:
        return parts[index]
    return fallback


def _unique_job_id(root: Path, base: str) -> str:
    candidate = base
    index = 2
    while (root / candidate).exists():
        candidate = f"{base}_{index:02d}"
        index += 1
    return candidate


def _default_pretrained(task_type: str) -> str:
    if task_type == "obb":
        return "models/pretrained/yolov8n-obb.pt"
    if task_type == "segmentation":
        return "models/pretrained/yolov8n-seg.pt"
    if task_type == "classification":
        return "models/pretrained/yolov8n-cls.pt"
    return "models/pretrained/yolov8n.pt"



def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

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
