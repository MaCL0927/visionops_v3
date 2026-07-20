#!/usr/bin/env python3
"""VisionOps v3 Collector Web 最小后端与 Runtime HTTP 代理。"""

from __future__ import annotations

import json
import mimetypes
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config_loader import CollectorConfig, load_config
from .model_catalog import find_scanned_model, scan_model_catalog
from .response_utils import error_document, send_bytes, send_json, timestamp_ms
from .runtime_client import RuntimeClient, RuntimeResponse, RuntimeUnavailable
from .sdk_bridge_settings import apply_sdk_bridge_settings, get_sdk_bridge_settings_payload
from .algorithm_settings import apply_algorithm_settings, get_algorithm_settings_payload
from .vision_box_settings import apply_vision_box_settings, get_vision_box_settings_payload, load_vision_box_settings
from .timed_capture import TimedCaptureController
from .dataset_manager import (
    create_and_upload_dataset,
    create_dataset_package,
    delete_image,
    get_image_file,
    list_images,
    list_packages,
    save_runtime_snapshot,
)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
MAX_REQUEST_BODY_BYTES = 1024 * 1024
PROXY_PATHS = {
    "/api/runtime/status": "GET",
    "/api/runtime/start_preview": "POST",
    "/api/runtime/stop_preview": "POST",
    "/api/runtime/infer_once": "POST",
    "/api/runtime/latest_result": "GET",
    "/api/runtime/snapshot.jpg": "GET",
}
DOWNSTREAM_PATHS = {
    "/api/gateway/status": ("gateway", "/api/gateway/status", True),
    "/api/gateway/registers": ("gateway", "/api/gateway/registers", False),
    "/api/app/status": ("business_app", "/api/app/status", True),
    "/api/app/registers": ("business_app", "/api/app/registers", False),
    "/api/app/inference_settings": ("business_app", "/api/app/inference_settings", False),
    "/api/app/latest_decision": ("business_app", "/api/app/latest_decision", False),
    "/api/app/latest_gateway_message": ("business_app", "/api/app/latest_gateway_message", False),
}


class CollectorServer(ThreadingHTTPServer):
    """保存 Collector 运行上下文的线程化 HTTP 服务。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__((config.host, config.port), CollectorRequestHandler)
        self.config = config
        self.started_at = time.monotonic()
        self.runtime_client = RuntimeClient(config.runtime_url)
        self.gateway_client = RuntimeClient(config.gateway_url)
        self.business_app_client = RuntimeClient(config.business_app_url)
        self.timed_capture = TimedCaptureController(self.runtime_client)

    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at


class CollectorRequestHandler(BaseHTTPRequestHandler):
    """只提供静态页面、Collector 状态和 Runtime HTTP 代理。"""

    server: CollectorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self._serve_static(path)
            return
        if path == "/health":
            self._send_health()
            return
        if path == "/api/collector/status":
            self._send_collector_status()
            return
        if path == "/api/collector/config":
            self._send_frontend_config()
            return
        if path == "/api/models":
            self._send_model_catalog()
            return
        if path in {"/api/settings/sdk_bridge", "/api/settings/sdk_bridge/orbbec336l", "/api/settings/sdk_bridge/hp60c"}:
            self._send_sdk_bridge_settings(path)
            return
        if path == "/api/settings/algorithm":
            self._send_algorithm_settings()
            return
        if path == "/api/settings/vision_box":
            self._send_vision_box_settings()
            return
        if path == "/api/dataset/images":
            self._send_dataset_images()
            return
        if path == "/api/dataset/packages":
            self._send_dataset_packages()
            return
        if path == "/api/dataset/timed_capture":
            send_json(self, 200, self.server.timed_capture.status())
            return
        if path == "/api/runtime/roi":
            self._proxy_runtime(path, expected_method="GET")
            return
        if path.startswith("/api/dataset/images/") and path.endswith("/content"):
            self._send_dataset_image_content(path)
            return
        if path in DOWNSTREAM_PATHS:
            name, target, status_endpoint = DOWNSTREAM_PATHS[path]
            self._proxy_downstream(name, target, status_endpoint)
            return
        if path in PROXY_PATHS:
            self._proxy_runtime(path, expected_method=PROXY_PATHS[path])
            return
        self._send_collector_error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/api/app/evaluate_once", "/api/app/inference_settings"}:
            body = self._read_request_body()
            if body is None:
                return
            self._proxy_downstream_post("business_app", path, body)
            return
        if path == "/api/models/switch":
            self._switch_model()
            return
        if path in {"/api/settings/sdk_bridge", "/api/settings/sdk_bridge/orbbec336l", "/api/settings/sdk_bridge/hp60c"}:
            self._apply_sdk_bridge_settings(path)
            return
        if path == "/api/settings/algorithm":
            self._apply_algorithm_settings()
            return
        if path == "/api/settings/vision_box":
            self._apply_vision_box_settings()
            return
        if path == "/api/dataset/images/capture":
            self._capture_dataset_image()
            return
        if path == "/api/dataset/timed_capture":
            self._configure_timed_capture()
            return
        if path == "/api/runtime/roi":
            self._proxy_runtime(path, expected_method="POST")
            return
        if path == "/api/dataset/packages/create":
            self._create_dataset_package()
            return
        if path == "/api/dataset/upload":
            self._upload_dataset_package()
            return
        if path in PROXY_PATHS:
            self._proxy_runtime(path, expected_method=PROXY_PATHS[path])
            return
        self._send_collector_error(404, "ROUTE_NOT_FOUND", "接口不存在", True)


    def do_DELETE(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/dataset/images/"):
            self._delete_dataset_image(path)
            return
        self._send_collector_error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_collector_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在", False)
            return
        send_bytes(self, 200, body, content_type, {"Cache-Control": "no-cache"})

    def _serve_static(self, request_path: str) -> None:
        relative = request_path[len("/static/"):] if request_path.startswith("/static/") else request_path.lstrip("/")
        if not relative or ".." in Path(relative).parts:
            self._send_collector_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在", False)
            return
        target = FRONTEND_DIR / "static" / relative
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type in {"text/javascript", "text/css"}:
            content_type += "; charset=utf-8"
        self._serve_file(target, content_type)

    def _send_health(self) -> None:
        config = self.server.config
        send_json(
            self,
            200,
            {
                "schema_version": "1.0",
                "message_type": "collector_health",
                "status": "ok",
                "component": config.component,
                "device_id": config.device_id,
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(self.server.uptime_s(), 3),
                "runtime_url": config.runtime_url,
                "gateway_url": config.gateway_url,
                "business_app_url": config.business_app_url,
                "models_root": config.models_root,
                "production_inference_source": config.production_inference_source,
            },
        )

    def _send_frontend_config(self) -> None:
        config = self.server.config
        try:
            board = load_vision_box_settings(config)
            board_path = get_vision_box_settings_payload(config).get("config_path")
        except Exception:
            board = {"default_mode": "factory", "disk_warning_percent": 85}
            board_path = None
        send_json(self, 200, {
            "schema_version": "1.0",
            "message_type": "collector_frontend_config",
            "runtime_url": config.runtime_url,
            "gateway_url": config.gateway_url,
            "business_app_url": config.business_app_url,
            "models_root": config.models_root,
            "production_inference_source": config.production_inference_source,
            "device_id": config.device_id,
            "snapshot_refresh_interval_ms": config.snapshot_refresh_interval_ms,
            "status_refresh_interval_ms": board.get("status_refresh_interval_ms", config.status_refresh_interval_ms),
            "default_mode": board.get("default_mode", "factory"),
            "disk_warning_percent": board.get("disk_warning_percent", 85),
            "vision_box_settings_path": board_path,
        })

    def _send_vision_box_settings(self) -> None:
        try:
            payload = get_vision_box_settings_payload(self.server.config)
        except Exception as error:  # noqa: BLE001 - expose local settings diagnostics
            self._send_collector_error(
                500,
                "VISION_BOX_SETTINGS_READ_FAILED",
                "读取视觉盒子设置失败",
                True,
                detail=str(error),
            )
            return
        send_json(self, 200, payload)

    def _apply_vision_box_settings(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            result = apply_vision_box_settings(self.server.config, payload)
        except ValueError as error:
            self._send_collector_error(400, "VISION_BOX_SETTINGS_INVALID", str(error), True)
            return
        except PermissionError as error:
            self._send_collector_error(403, "VISION_BOX_SETTINGS_PERMISSION_DENIED", "写入视觉盒子配置权限不足", True, detail=str(error))
            return
        except Exception as error:  # noqa: BLE001 - expose local apply diagnostics
            self._send_collector_error(500, "VISION_BOX_SETTINGS_APPLY_FAILED", "应用视觉盒子设置失败", True, detail=str(error))
            return
        send_json(self, 200, result)

    def _send_dataset_images(self) -> None:
        query = parse_qs(urlsplit(self.path).query)
        try:
            offset = int((query.get("offset") or ["0"])[0])
            limit = int((query.get("limit") or ["24"])[0])
            payload = list_images(offset=offset, limit=limit)
        except Exception as error:  # noqa: BLE001 - local dataset diagnostics
            self._send_collector_error(500, "DATASET_LIST_FAILED", "读取采集图片列表失败", True, detail=str(error))
            return
        send_json(self, 200, payload)

    def _send_dataset_packages(self) -> None:
        query = parse_qs(urlsplit(self.path).query)
        try:
            limit = int((query.get("limit") or ["20"])[0])
            payload = list_packages(limit=limit)
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(500, "DATASET_PACKAGE_LIST_FAILED", "读取上传包列表失败", True, detail=str(error))
            return
        send_json(self, 200, payload)

    def _send_dataset_image_content(self, path: str) -> None:
        try:
            filename = path[len("/api/dataset/images/"):-len("/content")]
            file_path, content_type = get_image_file(filename)
            send_bytes(self, 200, file_path.read_bytes(), content_type, {"Cache-Control": "no-cache"})
        except FileNotFoundError:
            self._send_collector_error(404, "DATASET_IMAGE_NOT_FOUND", "采集图片不存在", True)
        except ValueError as error:
            self._send_collector_error(400, "DATASET_IMAGE_INVALID", str(error), True)
        except OSError as error:
            self._send_collector_error(500, "DATASET_IMAGE_READ_FAILED", "读取采集图片失败", True, detail=str(error))

    def _delete_dataset_image(self, path: str) -> None:
        try:
            filename = path[len("/api/dataset/images/"):]
            payload = delete_image(filename)
        except FileNotFoundError:
            self._send_collector_error(404, "DATASET_IMAGE_NOT_FOUND", "采集图片不存在", True)
            return
        except ValueError as error:
            self._send_collector_error(400, "DATASET_IMAGE_INVALID", str(error), True)
            return
        except OSError as error:
            self._send_collector_error(500, "DATASET_IMAGE_DELETE_FAILED", "删除采集图片失败", True, detail=str(error))
            return
        send_json(self, 200, payload)

    def _capture_dataset_image(self) -> None:
        try:
            payload = save_runtime_snapshot(self.server.runtime_client)
        except RuntimeUnavailable as error:
            self._send_collector_error(502, "RUNTIME_SNAPSHOT_UNREACHABLE", "无法从 Runtime 获取快照", True, detail=str(error))
            return
        except PermissionError as error:
            self._send_collector_error(403, "DATASET_IMAGE_PERMISSION_DENIED", "保存采集图片权限不足", True, detail=str(error))
            return
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(500, "DATASET_CAPTURE_FAILED", "保存采集图片失败", True, detail=str(error))
            return
        send_json(self, 200, payload)

    def _configure_timed_capture(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            enabled = bool(payload.get("enabled", True))
            if enabled:
                result = self.server.timed_capture.start(
                    float(payload.get("interval_seconds", 10))
                )
            else:
                result = self.server.timed_capture.stop()
        except (TypeError, ValueError) as error:
            self._send_collector_error(
                400, "TIMED_CAPTURE_INVALID", str(error), True
            )
            return
        send_json(self, 200, result)

    def _create_dataset_package(self) -> None:
        payload_body = self._read_json_body()
        if payload_body is None:
            return
        try:
            payload = create_dataset_package(payload_body)
        except ValueError as error:
            self._send_collector_error(400, "DATASET_PACKAGE_EMPTY", str(error), True)
            return
        except PermissionError as error:
            self._send_collector_error(403, "DATASET_PACKAGE_PERMISSION_DENIED", "创建采集包权限不足", True, detail=str(error))
            return
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(500, "DATASET_PACKAGE_FAILED", "创建采集包失败", True, detail=str(error))
            return
        send_json(self, 200, payload)

    def _upload_dataset_package(self) -> None:
        payload_body = self._read_json_body()
        if payload_body is None:
            return
        try:
            payload = create_and_upload_dataset(self.server.config, payload_body)
        except ValueError as error:
            self._send_collector_error(400, "DATASET_UPLOAD_INVALID", str(error), True)
            return
        except PermissionError as error:
            self._send_collector_error(403, "DATASET_UPLOAD_PERMISSION_DENIED", "打包或上传权限不足", True, detail=str(error))
            return
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(500, "DATASET_UPLOAD_FAILED", "打包上传失败", True, detail=str(error))
            return
        send_json(self, 200, payload)


    def _current_runtime_model_for_settings(self) -> dict[str, Any] | None:
        try:
            response = self.server.runtime_client.request("GET", "/api/runtime/status")
            payload = self._decode_runtime_json(response)
        except (RuntimeUnavailable, ValueError, json.JSONDecodeError):
            return None
        loaded = payload.get("loaded_model")
        return loaded if isinstance(loaded, dict) else None

    def _send_algorithm_settings(self) -> None:
        query = parse_qs(urlsplit(self.path).query)
        model_id = (query.get("model_id") or [""])[0].strip() or None
        package_dir = (query.get("package_dir") or [""])[0].strip() or None
        try:
            payload = get_algorithm_settings_payload(
                Path(self.server.config.models_root),
                current_model=self._current_runtime_model_for_settings(),
                model_id=model_id,
                package_dir=package_dir,
            )
        except Exception as error:  # noqa: BLE001 - expose local settings diagnostics
            self._send_collector_error(
                500,
                "ALGORITHM_SETTINGS_READ_FAILED",
                "读取算法设置失败",
                True,
                detail=str(error),
            )
            return
        send_json(self, 200, payload)

    def _apply_algorithm_settings(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        current_model = self._current_runtime_model_for_settings()
        try:
            result = apply_algorithm_settings(
                Path(self.server.config.models_root),
                payload,
                current_model=current_model,
            )
        except ValueError as error:
            self._send_collector_error(
                400,
                "ALGORITHM_SETTINGS_INVALID",
                str(error),
                True,
            )
            return
        except PermissionError as error:
            self._send_collector_error(
                403,
                "ALGORITHM_SETTINGS_PERMISSION_DENIED",
                "写入模型 model.yaml 权限不足",
                True,
                detail=str(error),
            )
            return
        except Exception as error:  # noqa: BLE001 - expose local apply diagnostics
            self._send_collector_error(
                500,
                "ALGORITHM_SETTINGS_APPLY_FAILED",
                "应用算法设置失败",
                True,
                detail=str(error),
            )
            return

        if result.get("reload_runtime") and payload.get("reload_runtime", True):
            selected = result.get("selected_model") or {}
            body = json.dumps(
                {"model_dir": selected.get("package_path")},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            try:
                response = self.server.runtime_client.request("POST", "/api/runtime/switch_model", body=body)
                result["runtime_reload"] = {
                    "attempted": True,
                    "ok": response.status_code == 200,
                    "status_code": response.status_code,
                }
            except RuntimeUnavailable as error:
                result["runtime_reload"] = {
                    "attempted": True,
                    "ok": False,
                    "error": str(error),
                }
        else:
            result["runtime_reload"] = {"attempted": False, "ok": None}
        send_json(self, 200, result)

    def _send_sdk_bridge_settings(self, route_path: str) -> None:
        query = parse_qs(urlsplit(self.path).query)
        requested = (query.get("camera_model") or [""])[0].strip() or None
        if route_path.endswith("/orbbec336l"):
            requested = "orbbec336l"
        elif route_path.endswith("/hp60c"):
            requested = "hp60c"
        try:
            payload = get_sdk_bridge_settings_payload(requested)
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(
                500, "SDK_BRIDGE_SETTINGS_READ_FAILED",
                "读取相机 SDK Bridge 设置失败", True, detail=str(error),
            )
            return
        send_json(self, 200, payload)

    def _apply_sdk_bridge_settings(self, route_path: str) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if route_path.endswith("/orbbec336l"):
            payload["camera_model"] = "orbbec336l"
        elif route_path.endswith("/hp60c"):
            payload["camera_model"] = "hp60c"
        try:
            result = apply_sdk_bridge_settings(payload)
        except ValueError as error:
            self._send_collector_error(400, "SDK_BRIDGE_SETTINGS_INVALID", str(error), True)
            return
        except PermissionError as error:
            self._send_collector_error(
                403, "SDK_BRIDGE_SETTINGS_PERMISSION_DENIED",
                "写入 Bridge env、相机选择文件或重启服务权限不足", True, detail=str(error),
            )
            return
        except Exception as error:  # noqa: BLE001
            self._send_collector_error(
                500, "SDK_BRIDGE_SETTINGS_APPLY_FAILED",
                "应用相机 SDK Bridge 设置失败", True, detail=str(error),
            )
            return
        send_json(self, 200, result)

    def _send_model_catalog(self) -> None:
        current_model: dict[str, Any] | None = None
        runtime: dict[str, Any]
        try:
            response = self.server.runtime_client.request("GET", "/api/runtime/status")
            payload = self._decode_runtime_json(response)
            current_model = payload.get("loaded_model") if isinstance(payload.get("loaded_model"), dict) else None
            runtime = {
                "reachable": response.status_code == 200,
                "status_code": response.status_code,
            }
        except (RuntimeUnavailable, ValueError, json.JSONDecodeError) as error:
            runtime = {
                "reachable": False,
                "error": {
                    "code": "RUNTIME_UNREACHABLE",
                    "message": "Collector 无法读取 Runtime 当前模型状态",
                    "detail": str(error),
                    "recoverable": True,
                },
            }

        send_json(self, 200, {
            "schema_version": "1.0",
            "message_type": "model_catalog",
            "models_root": self.server.config.models_root,
            "current_model": current_model,
            "runtime": runtime,
            "models": scan_model_catalog(Path(self.server.config.models_root), current_model=current_model),
        })

    def _collector_snapshot(self) -> dict[str, Any]:
        config = self.server.config
        return {
            "status": "ok",
            "component": config.component,
            "device_id": config.device_id,
            "uptime_s": round(self.server.uptime_s(), 3),
        }

    def _send_collector_status(self) -> None:
        runtime: dict[str, Any]
        try:
            health_response = self.server.runtime_client.request("GET", "/health")
            status_response = self.server.runtime_client.request("GET", "/api/runtime/status")
            runtime = {
                "health": "ok" if health_response.status_code == 200 else "error",
                "reachable": True,
                "health_status_code": health_response.status_code,
                "status_status_code": status_response.status_code,
                "health_response": self._decode_runtime_json(health_response),
                "status_response": self._decode_runtime_json(status_response),
            }
        except (RuntimeUnavailable, ValueError, json.JSONDecodeError) as error:
            runtime = {
                "health": "unreachable",
                "reachable": False,
                "error": {
                    "code": "RUNTIME_UNREACHABLE",
                    "message": "Collector 无法连接 Runtime",
                    "detail": str(error),
                    "recoverable": True,
                },
            }

        send_json(
            self,
            200,
            {
                "schema_version": "1.0",
                "message_type": "collector_status",
                "timestamp_ms": timestamp_ms(),
                "collector": self._collector_snapshot(),
                "runtime": runtime,
                "proxy": {
                    "runtime_url": self.server.config.runtime_url,
                    "gateway_url": self.server.config.gateway_url,
                    "business_app_url": self.server.config.business_app_url,
                    "timeout_s": self.server.runtime_client.timeout_s,
                    "mode": "http",
                },
            },
        )

    def _proxy_downstream(self, name: str, target: str, status_endpoint: bool) -> None:
        clients = {
            "gateway": (self.server.gateway_client, self.server.config.gateway_url),
            "business_app": (self.server.business_app_client, self.server.config.business_app_url),
        }
        client, service_url = clients[name]
        try:
            response = client.request("GET", target)
        except RuntimeUnavailable as error:
            if status_endpoint:
                send_json(self, 200, {
                    "schema_version": "1.0",
                    "message_type": f"{name}_proxy_status",
                    "status": "unreachable",
                    "health": "unreachable",
                    "reachable": False,
                    "service": name,
                    "error": {
                        "code": f"{name.upper()}_UNREACHABLE",
                        "message": f"Collector 无法连接 {name}",
                        "detail": str(error),
                        "recoverable": True,
                    },
                })
            else:
                self._send_collector_error(
                    502, f"{name.upper()}_UNREACHABLE", f"Collector 无法连接 {name}",
                    True, detail=str(error),
                )
            return
        if response.content_type != "application/json":
            self._send_collector_error(502, "INVALID_DOWNSTREAM_RESPONSE", "下游返回非 JSON 内容", True, detail={"service": name, "content_type": response.content_type})
            return
        send_bytes(self, response.status_code, response.body, "application/json; charset=utf-8", {
            "X-VisionOps-Proxied-By": self.server.config.component,
            "X-VisionOps-Downstream-Url": service_url,
        })


    def _proxy_downstream_post(self, name: str, target: str, body: bytes) -> None:
        clients = {
            "gateway": (self.server.gateway_client, self.server.config.gateway_url),
            "business_app": (self.server.business_app_client, self.server.config.business_app_url),
        }
        client, service_url = clients[name]
        try:
            response = client.request("POST", target, body=body)
        except RuntimeUnavailable as error:
            self._send_collector_error(
                502, f"{name.upper()}_UNREACHABLE", f"Collector 无法连接 {name}",
                True, detail=str(error),
            )
            return
        if response.content_type != "application/json":
            self._send_collector_error(
                502, "INVALID_DOWNSTREAM_RESPONSE", "下游返回非 JSON 内容",
                True, detail={"service": name, "content_type": response.content_type},
            )
            return
        send_bytes(self, response.status_code, response.body, "application/json; charset=utf-8", {
            "X-VisionOps-Proxied-By": self.server.config.component,
            "X-VisionOps-Downstream-Url": service_url,
        })

    def _decode_runtime_json(self, response: RuntimeResponse) -> dict[str, Any]:
        if response.content_type != "application/json":
            raise ValueError(f"Runtime 返回非 JSON 内容: {response.content_type}")
        return response.json()

    def _read_request_body(self) -> bytes | None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_collector_error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法", True)
            return None
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            self._send_collector_error(413, "REQUEST_BODY_TOO_LARGE", "请求体超过限制", True)
            return None
        return self.rfile.read(length) if length else b"{}"

    def _read_json_body(self) -> dict[str, Any] | None:
        body = self._read_request_body()
        if body is None:
            return None
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_collector_error(400, "INVALID_JSON_BODY", "请求体必须是合法 JSON", True)
            return None
        if not isinstance(document, dict):
            self._send_collector_error(400, "INVALID_JSON_BODY", "请求体顶层必须是对象", True)
            return None
        return document

    def _switch_model(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        model_id = str(payload.get("model_id") or "").strip()
        package_dir = str(payload.get("package_dir") or "").strip()
        if not model_id and not package_dir:
            self._send_collector_error(
                400,
                "MODEL_SELECTOR_REQUIRED",
                "请求体必须包含 model_id 或 package_dir",
                True,
            )
            return

        scanned_models = scan_model_catalog(Path(self.server.config.models_root))
        selected = find_scanned_model(scanned_models, model_id=model_id, package_dir=package_dir)
        if selected is None:
            self._send_collector_error(
                404,
                "MODEL_NOT_FOUND",
                "Collector 未在 models_root 中找到指定模型包",
                True,
            )
            return
        if not selected.get("valid", False):
            self._send_collector_error(
                400,
                "MODEL_PACKAGE_INVALID",
                "指定模型包未通过 Collector 校验",
                True,
                detail=selected.get("error"),
            )
            return

        body = json.dumps(
            {"model_dir": selected["package_path"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            response = self.server.runtime_client.request(
                "POST",
                "/api/runtime/switch_model",
                body=body,
            )
        except RuntimeUnavailable as error:
            self._send_collector_error(
                502,
                "RUNTIME_UNREACHABLE",
                "Collector 无法连接 Runtime",
                True,
                detail=str(error),
            )
            return
        if response.content_type != "application/json":
            self._send_collector_error(
                502,
                "INVALID_RUNTIME_RESPONSE",
                "Runtime 返回了非预期内容类型",
                True,
                detail={"content_type": response.content_type},
            )
            return
        send_bytes(
            self,
            response.status_code,
            response.body,
            "application/json; charset=utf-8",
            {
                "X-VisionOps-Proxied-By": self.server.config.component,
                "X-VisionOps-Runtime-Url": self.server.config.runtime_url,
                "X-VisionOps-Proxy-Timestamp-Ms": str(timestamp_ms()),
            },
        )

    def _proxy_runtime(self, path: str, expected_method: str) -> None:
        if self.command != expected_method:
            self._send_collector_error(
                405,
                "METHOD_NOT_ALLOWED",
                f"请求方法不支持，期望 {expected_method}",
                True,
                headers={"Allow": expected_method},
            )
            return

        body = self._read_request_body() if self.command == "POST" else None
        if self.command == "POST" and body is None:
            return
        target = path
        query = urlsplit(self.path).query
        if query:
            target = f"{target}?{query}"
        try:
            response = self.server.runtime_client.request(self.command, target, body=body)
        except RuntimeUnavailable as error:
            self._send_collector_error(
                502,
                "RUNTIME_UNREACHABLE",
                "Collector 无法连接 Runtime",
                True,
                detail=str(error),
            )
            return

        if path == "/api/runtime/snapshot.jpg" and response.content_type == "image/jpeg":
            forwarded_headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower() in {"cache-control", "x-frame-id", "x-trace-id", "x-timestamp-ms"}
            }
            send_bytes(self, response.status_code, response.body, "image/jpeg", forwarded_headers)
            return

        if response.content_type != "application/json":
            self._send_collector_error(
                502,
                "INVALID_RUNTIME_RESPONSE",
                "Runtime 返回了非预期内容类型",
                True,
                detail={"content_type": response.content_type},
            )
            return
        send_bytes(
            self,
            response.status_code,
            response.body,
            "application/json; charset=utf-8",
            {
                "X-VisionOps-Proxied-By": self.server.config.component,
                "X-VisionOps-Runtime-Url": self.server.config.runtime_url,
                "X-VisionOps-Proxy-Timestamp-Ms": str(timestamp_ms()),
            },
        )

    def _send_collector_error(
        self,
        status_code: int,
        code: str,
        message: str,
        recoverable: bool,
        detail: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        send_json(
            self,
            status_code,
            error_document(
                device_id=self.server.config.device_id,
                component=self.server.config.component,
                code=code,
                message=message,
                recoverable=recoverable,
                detail=detail,
            ),
            headers,
        )


def run(config: CollectorConfig) -> int:
    server = CollectorServer(config)
    stop_requested = threading.Event()

    def request_shutdown(_signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            stop_requested.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        f"VisionOps Collector Web 正在监听 {config.host}:{config.port}，"
        f"Runtime={config.runtime_url}，Gateway={config.gateway_url}，"
        f"App={config.business_app_url}，ModelsRoot={config.models_root}"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.timed_capture.close()
        server.server_close()
    print("VisionOps Collector Web 已停止")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(load_config(argv))


if __name__ == "__main__":
    raise SystemExit(main())
