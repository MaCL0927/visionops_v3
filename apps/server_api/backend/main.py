#!/usr/bin/env python3
"""VisionOps v3 服务端 MVP：数据批次、训练任务、模型包与设备注册表。"""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config import ServerConfig, parse_args
from .services.dataset_service import DatasetService
from .services.device_service import DeviceService
from .services.ingest_service import BatchService
from .services.model_package_service import ModelPackageService
from .services.training_job_service import TrainingJobService
from .services.annotation_service import AnnotationService

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "static"


class VisionOpsServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: ServerConfig) -> None:
        config.ensure_dirs()
        super().__init__((config.host, config.port), ServerRequestHandler)
        self.config = config
        self.started_at = time.monotonic()
        self.batch_service = BatchService(config.batches_root, config.allowed_task_types, incoming_root=config.incoming_packages_root)
        self.dataset_service = DatasetService(config.datasets_root, self.batch_service)
        self.model_package_service = ModelPackageService(config.model_packages_root, config.publish_root)
        self.training_job_service = TrainingJobService(
            config.jobs_root,
            self.dataset_service,
            self.model_package_service,
            target_platform=config.default_target_platform,
        )
        self.device_service = DeviceService(config.devices_path)
        self.annotation_service = AnnotationService(self.batch_service, config.data_root)

    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at


class ServerRequestHandler(BaseHTTPRequestHandler):
    server: VisionOpsServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_empty(204)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
                return
            if path == "/annotate":
                self._serve_file(FRONTEND_DIR / "annotator.html", "text/html; charset=utf-8")
                return
            if path.startswith("/static/"):
                self._serve_static(path)
                return
            if path in {"/health", "/api/server/health"}:
                self._send_health()
                return
            if path == "/api/server/incoming-packages":
                self._send_json(200, self._ok("server_incoming_package_list", {
                    "incoming_root": str(self.server.config.incoming_packages_root),
                    "packages": self.server.batch_service.list_incoming_packages(),
                }))
                return
            if path == "/api/server/batches":
                self._send_json(200, self._ok("server_batch_list", {"batches": self.server.batch_service.list_batches()}))
                return
            if path.startswith("/api/server/batches/"):
                self._send_batch_detail(path)
                return
            if path == "/api/server/datasets":
                self._send_json(200, self._ok("server_dataset_list", {"datasets": self.server.dataset_service.list_datasets()}))
                return
            if path.startswith("/api/server/datasets/"):
                dataset_id = path.rsplit("/", 1)[-1]
                self._send_json(200, self._ok("server_dataset_detail", {"dataset": self.server.dataset_service.get_dataset(dataset_id)}))
                return
            if path == "/api/server/training/jobs":
                self._send_json(200, self._ok("server_training_job_list", {"jobs": self.server.training_job_service.list_jobs()}))
                return
            if path.startswith("/api/server/training/jobs/"):
                self._send_training_job_get(path)
                return
            if path == "/api/server/model-packages":
                self._send_json(200, self._ok("server_model_package_list", {"model_packages": self.server.model_package_service.list_packages()}))
                return
            if path.startswith("/api/server/model-packages/"):
                self._send_model_package_detail(path)
                return
            if path == "/api/server/devices":
                self._send_json(200, self._ok("server_device_list", {"devices": self.server.device_service.list_devices()}))
                return
            if path.startswith("/api/server/devices/"):
                device_id = path.rsplit("/", 1)[-1]
                self._send_json(200, self._ok("server_device_detail", {"device": self.server.device_service.get_device(device_id)}))
                return
            if self._handle_annotator_get(path):
                return
            self._send_error(404, "ROUTE_NOT_FOUND", "接口不存在")
        except Exception as error:
            self._handle_exception(error)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/server/batches/process-incoming":
                body = self._read_json_body(default={})
                packages = body.get("packages") if isinstance(body.get("packages"), list) else []
                batch = self.server.batch_service.process_incoming_packages([str(item) for item in packages])
                self._send_json(200, self._ok("server_incoming_package_processed", {"batch": batch}))
                return
            if path == "/api/server/batches/upload":
                self._upload_batch()
                return
            if path.startswith("/api/server/batches/") and path.endswith("/delete"):
                batch_id = path.split("/")[-2]
                deleted = self.server.batch_service.delete_batch(batch_id)
                self._send_json(200, self._ok("server_batch_deleted", {"batch": deleted}))
                return
            if path.startswith("/api/server/batches/") and path.endswith("/accept"):
                batch_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                batch = self.server.batch_service.set_status(batch_id, "accepted", str(body.get("note", "")), task_type=body.get("task_type"))
                self._send_json(200, self._ok("server_batch_accepted", {"batch": batch}))
                return
            if path.startswith("/api/server/batches/") and path.endswith("/reject"):
                batch_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                batch = self.server.batch_service.set_status(batch_id, "rejected", str(body.get("note", "")), task_type=body.get("task_type"))
                self._send_json(200, self._ok("server_batch_rejected", {"batch": batch}))
                return
            if path == "/api/server/datasets/build":
                body = self._read_json_body(default={})
                dataset = self.server.dataset_service.build_dataset(
                    task_type=str(body.get("task_type") or "detection"),
                    batch_ids=body.get("batch_ids") if isinstance(body.get("batch_ids"), list) else None,
                    name=body.get("name"),
                )
                self._send_json(200, self._ok("server_dataset_built", {"dataset": dataset}))
                return
            if path.startswith("/api/server/datasets/") and path.endswith("/delete"):
                dataset_id = path.split("/")[-2]
                deleted = self.server.dataset_service.delete_dataset(dataset_id)
                self._send_json(200, self._ok("server_dataset_deleted", {"dataset": deleted}))
                return
            if path == "/api/server/open-path":
                body = self._read_json_body(default={})
                result = self._open_local_path(str(body.get("path") or ""))
                self._send_json(200, self._ok("server_path_opened", {"result": result}))
                return
            if path == "/api/server/training/jobs":
                body = self._read_json_body(default={})
                job = self.server.training_job_service.create_job(body)
                self._send_json(200, self._ok("server_training_job_created", {"job": job}))
                return
            if path.startswith("/api/server/training/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                job = self.server.training_job_service.cancel_job(job_id)
                self._send_json(200, self._ok("server_training_job_canceled", {"job": job}))
                return
            if path.startswith("/api/server/training/jobs/") and path.endswith("/delete"):
                job_id = path.split("/")[-2]
                job = self.server.training_job_service.delete_job(job_id)
                self._send_json(200, self._ok("server_training_job_deleted", {"job": job}))
                return
            if path.startswith("/api/server/model-packages/") and path.endswith("/publish"):
                model_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                publish_root = Path(body["publish_root"]) if body.get("publish_root") else None
                result = self.server.model_package_service.publish_package(model_id, publish_root=publish_root)
                self._send_json(200, self._ok("server_model_package_published", {"publish": result}))
                return
            if path.startswith("/api/server/model-packages/") and path.endswith("/delete"):
                model_id = path.split("/")[-2]
                package = self.server.model_package_service.delete_package(model_id)
                self._send_json(200, self._ok("server_model_package_deleted", {"model_package": package}))
                return
            if path == "/api/server/devices":
                body = self._read_json_body(default={})
                device = self.server.device_service.upsert_device(body)
                self._send_json(200, self._ok("server_device_upserted", {"device": device}))
                return
            if path.startswith("/api/server/devices/") and path.endswith("/delete"):
                device_id = path.split("/")[-2]
                device = self.server.device_service.delete_device(device_id)
                self._send_json(200, self._ok("server_device_deleted", {"device": device}))
                return
            if path.startswith("/api/server/devices/") and path.endswith("/assign-model"):
                device_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                model_id = str(body.get("model_id") or "")
                package = self.server.model_package_service.get_package(model_id)
                if not package:
                    raise FileNotFoundError(f"模型包不存在: {model_id}")
                result = self.server.device_service.sync_model_to_device(
                    device_id,
                    model_id,
                    Path(str(package.get("package_path") or "")),
                )
                self._send_json(200, self._ok("server_device_model_assigned", {"device": result.get("device"), "sync": result}))
                return
            if self._handle_annotator_post(path):
                return
            self._send_error(404, "ROUTE_NOT_FOUND", "接口不存在")
        except Exception as error:
            self._handle_exception(error)

    def _send_batch_detail(self, path: str) -> None:
        batch_id = path.rsplit("/", 1)[-1]
        self._send_json(200, self._ok("server_batch_detail", {"batch": self.server.batch_service.get_batch(batch_id)}))

    def _send_training_job_get(self, path: str) -> None:
        parts = path.strip("/").split("/")
        job_id = parts[4] if len(parts) >= 5 else ""
        if path.endswith("/logs"):
            logs = self.server.training_job_service.get_logs(job_id)
            self._send_json(200, self._ok("server_training_job_logs", {"job_id": job_id, "logs": logs}))
        else:
            self._send_json(200, self._ok("server_training_job_detail", {"job": self.server.training_job_service.get_job(job_id)}))

    def _send_model_package_detail(self, path: str) -> None:
        model_id = path.rsplit("/", 1)[-1]
        package = self.server.model_package_service.get_package(model_id)
        self._send_json(200, self._ok("server_model_package_detail", {"model_package": package}))


    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.path).query)

    def _query_text(self, name: str, default: str = "") -> str:
        values = self._query().get(name)
        if not values:
            return default
        return str(values[0])

    def _require_batch_id(self) -> str:
        batch_id = self._query_text("batch_id")
        if not batch_id:
            raise ValueError("缺少 batch_id")
        return batch_id

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _handle_annotator_get(self, path: str) -> bool:
        service = self.server.annotation_service
        if path == "/api/annotator/session":
            batch_id = self._require_batch_id()
            self._send_json(200, service.session_info(batch_id))
            return True
        if path.startswith("/api/annotator/image/"):
            batch_id = self._require_batch_id()
            index = int(path.rsplit("/", 1)[-1])
            self._send_json(200, service.image_meta(batch_id, index))
            return True
        if path.startswith("/api/annotator/file/"):
            batch_id = self._require_batch_id()
            index = int(path.rsplit("/", 1)[-1])
            file_path = service.image_file(batch_id, index)
            self._serve_file(file_path, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
            return True
        if path.startswith("/api/annotator/jobs/"):
            batch_id = self._require_batch_id()
            job_id = path.rsplit("/", 1)[-1]
            self._send_json(200, service.jobs.get(service.quick_root(batch_id), job_id))
            return True
        if path == "/api/annotator/classification/session":
            batch_id = self._require_batch_id()
            self._send_json(200, service.classification_info(batch_id))
            return True
        if path == "/api/annotator/roi-cls/session":
            batch_id = self._require_batch_id()
            self._send_json(200, service.roi_session_info(batch_id, self._project_root()))
            return True
        if path.startswith("/api/annotator/roi-cls/sessions/"):
            batch_id = self._require_batch_id()
            session_id = path.rsplit("/", 1)[-1]
            self._send_json(200, {"manifest": service.get_roi_session(batch_id, session_id), "classes": service.roi_classes()})
            return True
        if path.startswith("/api/annotator/roi-cls/file/"):
            batch_id = self._require_batch_id()
            parts = path.strip("/").split("/")
            # api/annotator/roi-cls/file/{session_id}/{kind}/{filename}
            if len(parts) < 7:
                raise ValueError("ROI 文件路径不完整")
            session_id, kind, filename = parts[4], parts[5], parts[6]
            file_path = service.roi_file(batch_id, session_id, kind, filename)
            self._serve_file(file_path, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
            return True
        if path.startswith("/api/jobs/") and path.endswith("/logs"):
            # v2 标注器遗留的审核任务轮询接口；v3 当前确认审核后会直接返回控制台。
            self._send_json(200, {"status": {"status": "success", "message": "v3 annotator returns to console directly"}, "logs": ""})
            return True
        return False

    def _handle_annotator_post(self, path: str) -> bool:
        service = self.server.annotation_service
        if path == "/api/annotator/classes":
            batch_id = self._require_batch_id()
            body = self._read_json_body(default={})
            classes = body.get("classes") if isinstance(body.get("classes"), list) else []
            task_type = body.get("task_type")
            if task_type:
                service.save_task(batch_id, str(task_type))
            saved = service.save_classes(batch_id, [str(x) for x in classes])
            self._send_json(200, {"message": "类别已保存", "classes": saved, "num_classes": len(saved), "task_type": service.load_task(batch_id)})
            return True
        if path == "/api/annotator/task":
            batch_id = self._require_batch_id()
            task = service.save_task(batch_id, str(self._read_json_body(default={}).get("task_type") or "detection"))
            self._send_json(200, {"message": "任务类型已保存", "task_type": task})
            return True
        if path == "/api/annotator/save":
            batch_id = self._require_batch_id()
            self._send_json(200, service.save_annotation(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/annotator/confirm-auto":
            batch_id = self._require_batch_id()
            self._send_json(200, service.confirm_auto(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/annotator/classification/assign":
            batch_id = self._require_batch_id()
            self._send_json(200, service.assign_classification_image(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/annotator/quick-train":
            batch_id = self._require_batch_id()
            self._send_json(200, service.start_quick_train(batch_id, self._read_json_body(default={}), self._project_root()))
            return True
        if path == "/api/annotator/auto-label-remaining":
            batch_id = self._require_batch_id()
            self._send_json(200, service.start_auto_label(batch_id, self._read_json_body(default={}), self._project_root()))
            return True
        if path == "/api/annotator/roi-cls/classes":
            batch_id = self._require_batch_id()  # noqa: F841 - batch id keeps API scoped to current task.
            created = service.add_roi_class(str(self._read_json_body(default={}).get("class_name") or ""))
            self._send_json(200, {"message": "类别已创建", "class": created, "classes": service.roi_classes()})
            return True
        if path == "/api/annotator/roi-cls/build-candidates":
            batch_id = self._require_batch_id()
            self._send_json(200, service.start_roi_candidates(batch_id, self._read_json_body(default={}), self._project_root()))
            return True
        if path == "/api/annotator/roi-cls/label":
            batch_id = self._require_batch_id()
            self._send_json(200, service.label_roi(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/annotator/roi-cls/skip":
            batch_id = self._require_batch_id()
            self._send_json(200, service.skip_roi(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/annotator/roi-cls/roi-policy":
            batch_id = self._require_batch_id()
            self._send_json(200, service.save_roi_policy(batch_id, self._read_json_body(default={})))
            return True
        if path == "/api/accept-reviewed":
            batch_id = self._require_batch_id()
            task_type = str(self._read_json_body(default={}).get("task_type") or "detection")
            result = service.accept_reviewed(batch_id, task_type)
            batch = result.get("batch", {}) if isinstance(result, dict) else {}
            server_task = str(batch.get("task_type") or task_type or "detection")
            dataset = self.server.dataset_service.build_dataset(task_type=server_task, batch_ids=[batch_id])
            result["dataset"] = dataset
            result["message"] = f"审核完成，已自动生成训练数据集：{dataset.get('dataset_id')}"
            self._send_json(200, result)
            return True
        return False

    def _open_local_path(self, raw_path: str) -> dict[str, Any]:
        path = Path(str(raw_path or "")).expanduser().resolve()
        allowed_roots = [self.server.config.data_root.resolve(), Path(__file__).resolve().parents[3]]
        if self.server.config.publish_root:
            allowed_roots.append(self.server.config.publish_root.resolve())
        allow_any = os.environ.get("VISIONOPS_SERVER_ALLOW_OPEN_ANY_PATH", "").strip().lower() in {"1", "true", "yes", "on"}
        if not allow_any:
            allowed = False
            for root in allowed_roots:
                try:
                    path.relative_to(root)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                roots = ", ".join(str(root) for root in allowed_roots)
                raise ValueError(f"只允许打开受信任目录下的路径: {path}; allowed={roots}")
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"path": str(path), "opened": True}

    def _upload_batch(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self._send_error(400, "EMPTY_BODY", "上传内容为空")
            return
        if length > self.server.config.max_upload_bytes:
            self._send_error(413, "UPLOAD_TOO_LARGE", "上传文件超过限制")
            return

        ctype = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        with tempfile.TemporaryDirectory(prefix="visionops-upload-") as tmp:
            upload_path = Path(tmp) / "upload.bin"
            filename = "upload.bin"
            if ctype.startswith("multipart/form-data"):
                fields, files = _parse_multipart_form_data(ctype, body)
                file_body = files.get("file")
                if not file_body:
                    self._send_error(400, "FILE_MISSING", "multipart 请求缺少 file 字段")
                    return
                filename = str(fields.get("file_filename") or fields.get("filename") or "upload.bin")
                upload_path = Path(tmp) / filename
                upload_path.write_bytes(file_body)
            else:
                upload_path.write_bytes(body)
                filename = self.headers.get("X-Filename", "upload.bin")
            batch = self.server.batch_service.create_from_upload_archive(upload_path, filename=filename)
            self._send_json(200, self._ok("server_batch_uploaded", {"batch": batch}))

    def _send_health(self) -> None:
        config = self.server.config
        self._send_json(
            200,
            self._ok(
                "server_health",
                {
                    "status": "ok",
                    "component": config.component,
                    "version": config.version,
                    "uptime_s": round(self.server.uptime_s(), 3),
                    "data_root": str(config.data_root),
                    "incoming_root": str(config.incoming_packages_root),
                    "batch_root": str(config.batches_root),
                    "dataset_root": str(config.datasets_root),
                    "model_package_root": str(config.model_packages_root),
                    "publish_root": str(config.publish_root) if config.publish_root else None,
                    "mlflow_uri": config.mlflow_uri,
                    "allowed_task_types": list(config.allowed_task_types),
                    "time_ms": int(time.time() * 1000),
                },
            ),
        )

    def _serve_static(self, request_path: str) -> None:
        relative = request_path[len("/static/"):]
        if not relative or ".." in Path(relative).parts:
            self._send_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在")
            return
        target = FRONTEND_DIR / relative
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._serve_file(target, content_type)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在")
            return
        self._send_bytes(200, body, content_type)

    def _read_json_body(self, *, default: dict[str, Any]) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return dict(default)
        raw = self.rfile.read(length)
        if not raw:
            return dict(default)
        try:
            document = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON 解析失败: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("JSON body 顶层必须是对象")
        return document

    def _ok(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": "1.0", "message_type": message_type, **payload}

    def _handle_exception(self, error: Exception) -> None:
        if isinstance(error, FileNotFoundError):
            self._send_error(404, "NOT_FOUND", str(error))
        elif isinstance(error, ValueError):
            self._send_error(400, "BAD_REQUEST", str(error))
        else:
            self._send_error(500, "INTERNAL_ERROR", str(error))

    def _send_error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"schema_version": "1.0", "message_type": "server_error", "status": "error", "code": code, "message": message})

    def _send_json(self, status: int, document: dict[str, Any]) -> None:
        body = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self._send_common_headers(0, "text/plain; charset=utf-8")
        self.end_headers()

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._send_common_headers(len(body), content_type)
        self.end_headers()
        self.wfile.write(body)

    def _send_common_headers(self, length: int, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")



def _parse_multipart_form_data(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, bytes]]:
    """解析简单 multipart/form-data。

    仅用于服务端 MVP 的 zip 上传，避免依赖已从 Python 3.13 移除的 cgi 模块。
    """
    boundary_token = "boundary="
    if boundary_token not in content_type:
        raise ValueError("multipart 请求缺少 boundary")
    boundary = content_type.split(boundary_token, 1)[1].split(";", 1)[0].strip().strip('"')
    if not boundary:
        raise ValueError("multipart boundary 为空")
    delimiter = ("--" + boundary).encode("utf-8")
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    for raw_part in body.split(delimiter):
        part = raw_part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers = raw_headers.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((line for line in headers if line.lower().startswith("content-disposition:")), "")
        params = _parse_header_params(disposition)
        name = params.get("name")
        if not name:
            continue
        if "filename" in params:
            fields[f"{name}_filename"] = params.get("filename", "")
            files[name] = content
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def _parse_header_params(header_line: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in header_line.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip().lower()] = value.strip().strip('"')
    return params

def run(config: ServerConfig) -> None:
    server = VisionOpsServer(config)
    shutting_down = threading.Event()

    def _stop(signum: int, frame: object) -> None:  # noqa: ARG001
        # ThreadingHTTPServer.shutdown() must not be called from the same
        # thread that is running serve_forever(); otherwise Ctrl+C/SIGTERM can
        # deadlock.  Signal handlers run on the main thread, so perform the
        # shutdown from a tiny helper thread.
        if shutting_down.is_set():
            return
        shutting_down.set()
        signal_name = signal.Signals(signum).name if signum else "UNKNOWN"
        print(f"\n[INFO] VisionOps Server API stopping by {signal_name} ...", flush=True)
        threading.Thread(target=server.shutdown, name="visionops-server-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    print(f"[INFO] VisionOps Server API listening on {config.host}:{config.port}")
    print(f"[INFO] data_root={config.data_root}")
    print(f"[INFO] incoming_root={config.incoming_packages_root}")
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        print("[INFO] VisionOps Server API stopped", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
