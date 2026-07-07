#!/usr/bin/env python3
"""VisionOps v3 服务端 MVP：数据批次、训练任务、模型包与设备注册表。"""

from __future__ import annotations

import json
import mimetypes
import shutil
import signal
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
            if path.startswith("/api/server/model-packages/") and path.endswith("/publish"):
                model_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                publish_root = Path(body["publish_root"]) if body.get("publish_root") else None
                result = self.server.model_package_service.publish_package(model_id, publish_root=publish_root)
                self._send_json(200, self._ok("server_model_package_published", {"publish": result}))
                return
            if path == "/api/server/devices":
                body = self._read_json_body(default={})
                device = self.server.device_service.upsert_device(body)
                self._send_json(200, self._ok("server_device_upserted", {"device": device}))
                return
            if path.startswith("/api/server/devices/") and path.endswith("/assign-model"):
                device_id = path.split("/")[-2]
                body = self._read_json_body(default={})
                device = self.server.device_service.assign_model(device_id, str(body.get("model_id") or ""))
                self._send_json(200, self._ok("server_device_model_assigned", {"device": device}))
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
