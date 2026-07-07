"""训练任务管理。当前为可测试的 mock runner，真实训练/RKNN 转换后续接入。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .dataset_service import DatasetService
from .model_package_service import ModelPackageService

STAGES = ["preprocess", "train", "evaluate", "export_onnx", "convert_rknn", "package_v3_model"]


class TrainingJobService:
    def __init__(self, jobs_root: Path, dataset_service: DatasetService, model_package_service: ModelPackageService, *, target_platform: str = "rk3576") -> None:
        self.jobs_root = Path(jobs_root)
        self.dataset_service = dataset_service
        self.model_package_service = model_package_service
        self.target_platform = target_platform
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError("缺少 dataset_id")
        dataset = self.dataset_service.get_dataset(dataset_id)
        task_type = str(payload.get("task_type") or dataset.get("task_type") or "detection")
        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        now = int(time.time() * 1000)
        job = {
            "schema_version": "1.0",
            "job_id": job_id,
            "task_type": task_type,
            "dataset_id": dataset_id,
            "preset_name": payload.get("preset_name") or "default",
            "status": "pending",
            "current_stage": "pending",
            "epochs": int(payload.get("epochs", 50)),
            "batch_size": int(payload.get("batch_size", 16)),
            "imgsz": int(payload.get("imgsz", 640)),
            "device": payload.get("device") or "cuda:0",
            "target_platform": payload.get("target_platform") or self.target_platform,
            "mlflow_run_id": None,
            "output_model_package": None,
            "job_path": str(job_dir),
            "log_path": str(job_dir / "job.log"),
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "MVP mock runner：不执行真实训练，只验证任务编排和 v3 模型包生成契约。",
        }
        self._write_job(job_id, job)
        run_inline = bool(payload.get("run_inline", False))
        if run_inline:
            self._run_mock_pipeline(job_id)
        else:
            thread = threading.Thread(target=self._run_mock_pipeline, args=(job_id,), daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self.get_job(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        result = []
        for job_dir in sorted([entry for entry in self.jobs_root.iterdir() if entry.is_dir()], key=lambda x: x.name):
            meta = _read_json(job_dir / "job.json", {})
            if meta.get("job_id"):
                result.append(meta)
        return result

    def get_job(self, job_id: str) -> dict[str, Any]:
        job_id = _safe_id(job_id)
        job = _read_json(self.jobs_root / job_id / "job.json", {})
        if not job.get("job_id"):
            raise FileNotFoundError(f"训练任务不存在: {job_id}")
        return job

    def get_logs(self, job_id: str, tail_bytes: int = 20000) -> str:
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
        job["status"] = "canceled"
        job["current_stage"] = "canceled"
        job["updated_at_ms"] = int(time.time() * 1000)
        self._write_job(job_id, job)
        self._append_log(job_id, "任务已标记为 canceled。MVP mock runner 不强杀真实训练进程。")
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
        except Exception as error:  # pragma: no cover - 守护线程兜底
            try:
                job = self.get_job(job_id)
                job["status"] = "failed"
                job["current_stage"] = "failed"
                job["error"] = str(error)
                job["updated_at_ms"] = int(time.time() * 1000)
                self._write_job(job_id, job)
                self._append_log(job_id, f"[ERROR] {error}")
            except Exception:
                pass

    def _write_job(self, job_id: str, value: dict[str, Any]) -> None:
        job_id = _safe_id(job_id)
        _write_json(self.jobs_root / job_id / "job.json", value)

    def _append_log(self, job_id: str, line: str) -> None:
        job = self.get_job(job_id)
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


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
