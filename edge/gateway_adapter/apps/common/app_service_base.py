"""业务 App 的通用 HTTP、轮询和可选 Modbus 服务基座。

M11 起该基座同时服务 mock case 与真实 Runtime/Collector 上游。业务 App
只消费标准 inference_result，不直接访问相机、RKNN 或 Web 页面。
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib.parse import urlsplit

from edge.gateway_adapter.gateway_message import make_error_document, timestamp_ms
from edge.gateway_adapter.result_fetcher import UpstreamUnavailable
from edge.modbus_adapter.modbus_tcp_mock import ModbusTcpServer, start_modbus_server

from .app_config import load_app_config
from .app_decision import AppDecision
from .app_register_bank import AppRegisterBank
from .mock_result_loader import MockResultLoader


DecisionFunction = Callable[[dict, Mapping[str, Any], int, int, str], AppDecision]
RegisterValueFunction = Callable[[AppDecision], Dict[str, int]]
DefinitionFactory = Callable[[int], tuple]


@dataclass
class AppCounters:
    evaluations: int = 0
    decisions: int = 0
    no_result: int = 0
    upstream_errors: int = 0
    decision_errors: int = 0
    unchanged_results: int = 0


class BusinessAppState:
    def __init__(
        self,
        *,
        app_id: str,
        component: str,
        device_id: str,
        app_instance_id: str,
        config: dict[str, Any],
        loader: MockResultLoader,
        bank: AppRegisterBank,
        decide: DecisionFunction,
        register_values: RegisterValueFunction,
    ) -> None:
        self.app_id = app_id
        self.component = component
        self.device_id = device_id
        self.app_instance_id = app_instance_id
        self.config = config
        self.loader = loader
        self.bank = bank
        self.decide = decide
        self.register_values = register_values
        self.started_at = time.monotonic()
        self._lock = threading.RLock()
        self._sequence = 0
        self._heartbeat = 0
        self._latest_result_id: str | None = None
        self._latest_result_summary: dict[str, Any] | None = None
        self._latest_decision: dict[str, Any] | None = None
        self._latest_gateway_message: dict[str, Any] | None = None
        self._last_error: dict[str, Any] | None = None
        self._upstream: dict[str, Any] = {
            "kind": loader.upstream_kind,
            "url": loader.upstream_url,
            "health": "local_mock" if loader.upstream_kind == "file" else "unknown",
            "reachable": loader.upstream_kind == "file",
            "mock_case": loader.mock_case if loader.upstream_kind == "file" else None,
        }
        self._counters = AppCounters()

    def evaluate_once(self, *, force: bool) -> tuple[str, dict[str, Any] | None]:
        with self._lock:
            self._counters.evaluations += 1
        try:
            loaded = self.loader.load()
        except (UpstreamUnavailable, json.JSONDecodeError, ValueError) as error:
            with self._lock:
                self._counters.upstream_errors += 1
                self._last_error = {
                    "code": "UPSTREAM_UNREACHABLE",
                    "message": str(error),
                    "recoverable": True,
                    "timestamp_ms": timestamp_ms(),
                }
                self._upstream = {
                    "kind": self.loader.upstream_kind,
                    "url": self.loader.upstream_url,
                    "health": "unreachable",
                    "reachable": False,
                    "error": str(error),
                }
            return "unreachable", None

        if loaded.status_code == 404:
            with self._lock:
                self._counters.no_result += 1
                self._last_error = {
                    "code": "UPSTREAM_NO_RESULT",
                    "message": "上游尚无 latest_result",
                    "recoverable": True,
                    "timestamp_ms": timestamp_ms(),
                }
                self._upstream = {
                    "kind": self.loader.upstream_kind,
                    "url": self.loader.upstream_url,
                    "health": "no_latest_result",
                    "reachable": True,
                    "http_status": 404,
                }
            return "no_result", None
        if loaded.status_code != 200 or not isinstance(loaded.document, dict):
            with self._lock:
                self._counters.upstream_errors += 1
                self._last_error = {
                    "code": "UPSTREAM_ERROR",
                    "message": f"上游返回 HTTP {loaded.status_code}",
                    "recoverable": True,
                    "timestamp_ms": timestamp_ms(),
                }
                self._upstream = {
                    "kind": self.loader.upstream_kind,
                    "url": self.loader.upstream_url,
                    "health": "error",
                    "reachable": True,
                    "http_status": loaded.status_code,
                }
            return "upstream_error", None

        result = loaded.document
        result_id = str(result.get("result_id", ""))
        with self._lock:
            if not force and result_id and result_id == self._latest_result_id:
                self._counters.unchanged_results += 1
                return "unchanged", self._latest_decision
            sequence = self._sequence + 1
            heartbeat = self._heartbeat ^ 1
        try:
            rules = self.config["rules"]
            decision = self.decide(result, rules, sequence, heartbeat, self.device_id)
            gateway_message = self.bank.update_decision(
                decision,
                self.register_values(decision),
            )
        except (KeyError, TypeError, ValueError) as error:
            with self._lock:
                self._counters.decision_errors += 1
                self._last_error = {
                    "code": "DECISION_ERROR",
                    "message": str(error),
                    "recoverable": True,
                    "timestamp_ms": timestamp_ms(),
                }
                self._upstream = {
                    "kind": self.loader.upstream_kind,
                    "url": self.loader.upstream_url,
                    "health": "invalid_result",
                    "reachable": True,
                    "error": str(error),
                }
            return "decision_error", None

        document = decision.to_dict()
        with self._lock:
            self._sequence = sequence
            self._heartbeat = heartbeat
            self._latest_result_id = decision.result_id
            self._latest_result_summary = self._summarize_result(result)
            self._latest_decision = document
            self._latest_gateway_message = gateway_message
            self._counters.decisions += 1
            self._last_error = None if decision.ok else self._last_error
            self._upstream = {
                "kind": self.loader.upstream_kind,
                "url": self.loader.upstream_url,
                "health": "local_mock" if self.loader.upstream_kind == "file" else "ok",
                "reachable": True,
                "http_status": loaded.status_code,
                "latest_result_id": decision.result_id,
                "latest_frame_id": decision.frame_id,
                "latest_task_type": result.get("task_type"),
                "latest_model_name": (result.get("model") or {}).get("model_name") if isinstance(result.get("model"), dict) else None,
                "mock_case": (
                    self.loader.mock_case if self.loader.upstream_kind == "file" else None
                ),
            }
        return "updated", document

    @staticmethod
    def _summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
        detections = result.get("detections") if isinstance(result.get("detections"), list) else []
        model = result.get("model") if isinstance(result.get("model"), Mapping) else {}
        image = result.get("image") if isinstance(result.get("image"), Mapping) else {}
        return {
            "message_type": result.get("message_type"),
            "status": result.get("status"),
            "result_id": result.get("result_id"),
            "frame_id": result.get("frame_id"),
            "task_type": result.get("task_type"),
            "model_name": model.get("model_name"),
            "backend": model.get("backend"),
            "image": dict(image),
            "detection_count": len(detections),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            app_health = "ok"
            if self._upstream.get("health") in {"unreachable", "error", "invalid_result"}:
                app_health = "degraded"
            return {
                "schema_version": "1.0",
                "message_type": "app_status",
                "status": "ok",
                "health": app_health,
                "app_id": self.app_id,
                "app_instance_id": self.app_instance_id,
                "component": self.component,
                "device_id": self.device_id,
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "config": self.config,
                "upstream": dict(self._upstream),
                "latest_result_summary": self._latest_result_summary,
                "latest_decision": self._latest_decision,
                "latest_gateway_message": self._latest_gateway_message,
                "register_snapshot": self.bank.snapshot(),
                "register_map": self.bank.register_map(),
                "counters": vars(self._counters).copy(),
                "last_error": self._last_error,
            }

    def latest_gateway_message(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_gateway_message


class AppHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: BusinessAppState, modbus: dict) -> None:
        self.app_state = state
        self.modbus = modbus
        super().__init__(address, AppRequestHandler)


class AppRequestHandler(BaseHTTPRequestHandler):
    server: AppHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            snapshot = self.server.app_state.snapshot()
            self._json(200, {
                "schema_version": "1.0",
                "message_type": "app_health",
                "status": "ok",
                "health": snapshot["health"],
                "app_id": snapshot["app_id"],
                "app_instance_id": snapshot["app_instance_id"],
                "component": snapshot["component"],
                "device_id": snapshot["device_id"],
                "timestamp_ms": timestamp_ms(),
                "uptime_s": snapshot["uptime_s"],
                "upstream_kind": self.server.app_state.loader.upstream_kind,
                "modbus": self.server.modbus,
            })
        elif path == "/api/app/status":
            self._json(200, self.server.app_state.snapshot())
        elif path == "/api/app/latest_decision":
            decision = self.server.app_state.snapshot()["latest_decision"]
            if decision is None:
                self._error(404, "LATEST_DECISION_NOT_FOUND", "尚未生成业务决策")
            else:
                self._json(200, decision)
        elif path == "/api/app/latest_gateway_message":
            message = self.server.app_state.latest_gateway_message()
            if message is None:
                self._error(404, "LATEST_GATEWAY_MESSAGE_NOT_FOUND", "尚未生成 GatewayMessage")
            else:
                self._json(200, message)
        elif path == "/api/app/registers":
            self._json(200, {
                "schema_version": "1.0",
                "message_type": "app_register_snapshot",
                "timestamp_ms": timestamp_ms(),
                "registers": self.server.app_state.bank.snapshot(),
            })
        elif path == "/api/app/register_map":
            self._json(200, {
                "schema_version": "1.0",
                "message_type": "app_register_map",
                "app_id": self.server.app_state.app_id,
                "registers": self.server.app_state.bank.register_map(),
            })
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在")

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/app/evaluate_once":
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在")
            return
        length = self.headers.get("Content-Length", "0")
        try:
            size = int(length)
        except ValueError:
            self._error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法")
            return
        if not 0 <= size <= 1024 * 1024:
            self._error(413, "REQUEST_BODY_TOO_LARGE", "请求体超过限制")
            return
        if size:
            self.rfile.read(size)
        outcome, decision = self.server.app_state.evaluate_once(force=True)
        if outcome in {"updated", "unchanged"} and decision is not None:
            self._json(200, decision)
        elif outcome == "no_result":
            self._error(404, "UPSTREAM_NO_RESULT", "上游尚无推理结果")
        elif outcome == "unreachable":
            self._error(502, "UPSTREAM_UNREACHABLE", "业务 App 无法连接上游")
        else:
            self._error(502, "APP_EVALUATION_FAILED", f"业务决策失败: {outcome}")

    def _error(self, status: int, code: str, message: str) -> None:
        state = self.server.app_state
        self._json(status, make_error_document(
            device_id=state.device_id,
            component=state.component,
            code=code,
            message=message,
            recoverable=True,
        ))

    def _json(self, status: int, document: dict[str, Any]) -> None:
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_port: int,
    default_modbus_port: int,
    mock_cases: Sequence[str],
) -> None:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8090")
    parser.add_argument("--upstream-kind", choices=("collector", "runtime", "file"), default="collector")
    parser.add_argument("--config")
    parser.add_argument("--mock-case", choices=tuple(mock_cases), default="ok")
    parser.add_argument("--device-id", default="example-edge-001")
    parser.add_argument("--app-instance-id", default="")
    parser.add_argument("--poll-interval-ms", type=int, default=500)
    parser.add_argument("--modbus-host", default="0.0.0.0")
    parser.add_argument("--modbus-port", type=int, default=default_modbus_port)
    parser.add_argument("--enable-modbus", action="store_true")


def run_business_app(
    args: argparse.Namespace,
    *,
    defaults: Mapping[str, Any],
    mock_factories: Mapping[str, Callable[[], dict]],
    decide: DecisionFunction,
    definition_factory: DefinitionFactory,
    register_values: RegisterValueFunction,
) -> int:
    if not 1 <= args.port <= 65535 or not 1 <= args.modbus_port <= 65535:
        raise ValueError("端口必须位于 1..65535")
    if args.poll_interval_ms <= 0:
        raise ValueError("poll-interval-ms 必须大于 0")
    config = load_app_config(args.config, defaults)
    app_id = str(config["app"]["name"])
    app_instance_id = args.app_instance_id or str(config.get("app", {}).get("instance_id") or app_id)
    component = f"{app_id}_app"
    register_base = int(config["rules"]["register_base"])
    bank = AppRegisterBank(definition_factory(register_base))
    loader = MockResultLoader(
        upstream_kind=args.upstream_kind,
        upstream_url=args.upstream_url,
        mock_case=args.mock_case,
        mock_factories=mock_factories,
    )
    state = BusinessAppState(
        app_id=app_id,
        component=component,
        device_id=args.device_id,
        app_instance_id=app_instance_id,
        config=config,
        loader=loader,
        bank=bank,
        decide=decide,
        register_values=register_values,
    )
    modbus_server: ModbusTcpServer | None = None
    modbus_thread: threading.Thread | None = None
    http_server: AppHttpServer | None = None
    stop_event = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        if http_server is not None:
            threading.Thread(target=http_server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        if args.enable_modbus:
            modbus_server, modbus_thread = start_modbus_server(
                args.modbus_host, args.modbus_port, bank
            )
        modbus_info = {
            "enabled": bool(args.enable_modbus),
            "host": args.modbus_host,
            "port": args.modbus_port,
        }
        http_server = AppHttpServer((args.host, args.port), state, modbus_info)

        def poll_loop() -> None:
            while not stop_event.wait(args.poll_interval_ms / 1000.0):
                state.evaluate_once(force=False)

        poll_thread = threading.Thread(target=poll_loop, name=f"{app_id}-poller", daemon=True)
        poll_thread.start()
        print(
            f"{component} HTTP={args.host}:{args.port} upstream={args.upstream_kind} "
            f"url={args.upstream_url} instance={app_instance_id}"
        )
        http_server.serve_forever(poll_interval=0.2)
        stop_event.set()
        poll_thread.join(timeout=2)
        return 0
    finally:
        stop_event.set()
        if http_server is not None:
            http_server.server_close()
        if modbus_server is not None:
            modbus_server.shutdown()
            modbus_server.server_close()
        if modbus_thread is not None:
            modbus_thread.join(timeout=2)
