#!/usr/bin/env python3
"""标准 inference_result 到 Gateway/Modbus 的最小闭环 Mock 服务。"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence
from urllib.parse import urlparse, urlsplit

from edge.gateway_adapter.gateway_message import make_error_document, timestamp_ms
from edge.gateway_adapter.result_fetcher import ResultFetcher, UpstreamUnavailable
from edge.gateway_adapter.result_to_gateway import inference_result_to_gateway_message
from edge.modbus_adapter.modbus_registers import HoldingRegisterBank
from edge.modbus_adapter.modbus_tcp_mock import ModbusTcpServer, start_modbus_server


@dataclass
class GatewayCounters:
    polls: int = 0
    messages: int = 0
    no_result: int = 0
    upstream_errors: int = 0
    mapping_errors: int = 0


@dataclass
class GatewaySnapshot:
    uptime_s: float
    upstream: dict[str, Any]
    latest_result_id: str | None
    latest_frame_id: str | None
    latest_gateway_message: dict | None
    counters: GatewayCounters


class GatewayState:
    """聚合 Gateway、上游和寄存器状态。"""

    def __init__(
        self,
        *,
        device_id: str,
        app_id: str,
        component: str,
        fetcher: ResultFetcher,
        register_bank: HoldingRegisterBank,
    ) -> None:
        self.device_id = device_id
        self.app_id = app_id
        self.component = component
        self.fetcher = fetcher
        self.register_bank = register_bank
        self.started_at = time.monotonic()
        self._lock = threading.RLock()
        self._sequence = 0
        self._heartbeat = 0
        self._latest_result_id: str | None = None
        self._latest_frame_id: str | None = None
        self._latest_gateway_message: dict | None = None
        self._upstream: dict[str, Any] = {
            "kind": fetcher.upstream_kind,
            "url": fetcher.upstream_url,
            "health": "unknown",
            "reachable": False,
        }
        self._counters = GatewayCounters()

    def snapshot(self) -> GatewaySnapshot:
        with self._lock:
            return GatewaySnapshot(
                uptime_s=time.monotonic() - self.started_at,
                upstream=dict(self._upstream),
                latest_result_id=self._latest_result_id,
                latest_frame_id=self._latest_frame_id,
                latest_gateway_message=(
                    dict(self._latest_gateway_message) if self._latest_gateway_message else None
                ),
                counters=GatewayCounters(**vars(self._counters)),
            )

    def poll_once(self, *, force: bool) -> tuple[str, dict | None]:
        with self._lock:
            self._counters.polls += 1
        try:
            fetched = self.fetcher.fetch_latest_result()
        except (UpstreamUnavailable, json.JSONDecodeError, ValueError) as error:
            with self._lock:
                self._counters.upstream_errors += 1
                self._upstream = {
                    "kind": self.fetcher.upstream_kind,
                    "url": self.fetcher.upstream_url,
                    "health": "unreachable",
                    "reachable": False,
                    "error": {
                        "code": "UPSTREAM_UNREACHABLE",
                        "message": "Gateway 无法连接上游 latest_result",
                        "detail": str(error),
                        "recoverable": True,
                    },
                }
            return "unreachable", None

        if fetched.status_code == 404:
            with self._lock:
                self._counters.no_result += 1
                self._upstream = {
                    "kind": self.fetcher.upstream_kind,
                    "url": self.fetcher.upstream_url,
                    "health": "no_latest_result",
                    "reachable": True,
                    "http_status": 404,
                }
            return "no_latest_result", None
        if fetched.status_code != 200 or not isinstance(fetched.document, dict):
            with self._lock:
                self._counters.upstream_errors += 1
                self._upstream = {
                    "kind": self.fetcher.upstream_kind,
                    "url": self.fetcher.upstream_url,
                    "health": "error",
                    "reachable": True,
                    "http_status": fetched.status_code,
                }
            return "upstream_error", None

        result = fetched.document
        result_id = result.get("result_id")
        with self._lock:
            if not force and result_id == self._latest_result_id:
                self._upstream = {
                    "kind": self.fetcher.upstream_kind,
                    "url": self.fetcher.upstream_url,
                    "health": "ok",
                    "reachable": True,
                    "http_status": 200,
                }
                return "unchanged", self._latest_gateway_message
            sequence = self._sequence + 1
            heartbeat = self._heartbeat ^ 1

        try:
            message = inference_result_to_gateway_message(
                result,
                app_id=self.app_id,
                sequence=sequence,
                heartbeat=heartbeat,
            )
            message["device_id"] = self.device_id
            message["component"] = self.component
            message["protocol"] = "modbus_tcp"
            self.register_bank.update_from_gateway_message(message)
        except (TypeError, ValueError, KeyError) as error:
            with self._lock:
                self._counters.mapping_errors += 1
                self._upstream = {
                    "kind": self.fetcher.upstream_kind,
                    "url": self.fetcher.upstream_url,
                    "health": "invalid_result",
                    "reachable": True,
                    "http_status": 200,
                    "error": {
                        "code": "RESULT_MAPPING_FAILED",
                        "message": "上游结果无法映射为 Gateway 消息",
                        "detail": str(error),
                        "recoverable": True,
                    },
                }
            return "mapping_error", None

        with self._lock:
            self._sequence = sequence
            self._heartbeat = heartbeat
            self._latest_result_id = message["result_id"]
            self._latest_frame_id = message["frame_id"]
            self._latest_gateway_message = message
            self._counters.messages += 1
            self._upstream = {
                "kind": self.fetcher.upstream_kind,
                "url": self.fetcher.upstream_url,
                "health": "ok",
                "reachable": True,
                "http_status": 200,
            }
        return "updated", message


class GatewayHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        state: GatewayState,
        modbus_port: int,
    ) -> None:
        self.gateway_state = state
        self.modbus_port = modbus_port
        super().__init__(address, GatewayRequestHandler)


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server: GatewayHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._health()
        elif path == "/api/gateway/status":
            self._status()
        elif path == "/api/gateway/latest_message":
            self._latest_message()
        elif path == "/api/gateway/registers":
            self._registers()
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/gateway/poll_once":
            self._poll_once()
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def _health(self) -> None:
        snapshot = self.server.gateway_state.snapshot()
        state = self.server.gateway_state
        self._json(
            200,
            {
                "schema_version": "1.0",
                "message_type": "gateway_health",
                "status": "ok",
                "component": state.component,
                "device_id": state.device_id,
                "app_id": state.app_id,
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(snapshot.uptime_s, 3),
                "upstream_url": state.fetcher.upstream_url,
                "modbus_port": self.server.modbus_port,
            },
        )

    def _status(self) -> None:
        snapshot = self.server.gateway_state.snapshot()
        state = self.server.gateway_state
        self._json(
            200,
            {
                "schema_version": "1.0",
                "message_type": "gateway_status",
                "timestamp_ms": timestamp_ms(),
                "gateway": {
                    "status": "ok",
                    "component": state.component,
                    "device_id": state.device_id,
                    "app_id": state.app_id,
                    "uptime_s": round(snapshot.uptime_s, 3),
                    "modbus_port": self.server.modbus_port,
                },
                "upstream": snapshot.upstream,
                "latest_result_id": snapshot.latest_result_id,
                "latest_frame_id": snapshot.latest_frame_id,
                "latest_gateway_message": snapshot.latest_gateway_message,
                "register_snapshot": state.register_bank.snapshot(),
                "counters": vars(snapshot.counters),
            },
        )

    def _poll_once(self) -> None:
        content_length = self.headers.get("Content-Length", "0")
        try:
            size = int(content_length)
        except ValueError:
            self._error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法", True)
            return
        if size < 0 or size > 1024 * 1024:
            self._error(413, "REQUEST_BODY_TOO_LARGE", "请求体超过限制", True)
            return
        if size:
            self.rfile.read(size)
        outcome, message = self.server.gateway_state.poll_once(force=True)
        if outcome == "updated" and message is not None:
            self._json(200, message)
        elif outcome == "no_latest_result":
            self._error(404, "LATEST_RESULT_NOT_FOUND", "上游尚未生成推理结果", True)
        elif outcome == "unreachable":
            self._error(502, "UPSTREAM_UNREACHABLE", "Gateway 无法连接上游", True)
        else:
            self._error(502, "GATEWAY_POLL_FAILED", f"Gateway 拉取失败: {outcome}", True)

    def _latest_message(self) -> None:
        message = self.server.gateway_state.snapshot().latest_gateway_message
        if message is None:
            self._error(404, "LATEST_GATEWAY_MESSAGE_NOT_FOUND", "尚未生成 Gateway 消息", True)
            return
        self._json(200, message)

    def _registers(self) -> None:
        self._json(
            200,
            {
                "schema_version": "1.0",
                "message_type": "holding_register_snapshot",
                "timestamp_ms": timestamp_ms(),
                "registers": self.server.gateway_state.register_bank.snapshot(),
            },
        )

    def _error(self, status: int, code: str, message: str, recoverable: bool) -> None:
        state = self.server.gateway_state
        self._json(
            status,
            make_error_document(
                device_id=state.device_id,
                component=state.component,
                code=code,
                message=message,
                recoverable=recoverable,
            ),
        )

    def _json(self, status: int, document: dict) -> None:
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须位于 1 到 65535")
    return port


def _url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("upstream-url 必须是有效 HTTP URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("upstream-url 只能包含协议、主机和端口")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps Gateway / Modbus Mock")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_port, default=19090)
    parser.add_argument("--upstream-url", type=_url, default="http://127.0.0.1:8090")
    parser.add_argument("--upstream-kind", choices=("collector", "runtime"), default="collector")
    parser.add_argument("--modbus-host", default="0.0.0.0")
    parser.add_argument("--modbus-port", type=_port, default=1502)
    parser.add_argument("--poll-interval-ms", type=int, default=500)
    parser.add_argument("--device-id", default="example-edge-001")
    parser.add_argument("--app-id", default="generic_mock")
    parser.add_argument("--component", default="gateway_mock")
    return parser


def _poll_loop(state: GatewayState, stop_event: threading.Event, interval_ms: int) -> None:
    while not stop_event.wait(interval_ms / 1000.0):
        state.poll_once(force=False)


def run(args: argparse.Namespace) -> int:
    if args.poll_interval_ms <= 0:
        raise ValueError("poll-interval-ms 必须大于 0")
    bank = HoldingRegisterBank()
    fetcher = ResultFetcher(args.upstream_url, args.upstream_kind)
    state = GatewayState(
        device_id=args.device_id,
        app_id=args.app_id,
        component=args.component,
        fetcher=fetcher,
        register_bank=bank,
    )
    modbus_server: ModbusTcpServer | None = None
    http_server: GatewayHttpServer | None = None
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
        modbus_server, modbus_thread = start_modbus_server(args.modbus_host, args.modbus_port, bank)
        http_server = GatewayHttpServer((args.host, args.port), state, args.modbus_port)
        poll_thread = threading.Thread(
            target=_poll_loop,
            args=(state, stop_event, args.poll_interval_ms),
            name="gateway-poller",
            daemon=True,
        )
        poll_thread.start()
        print(
            f"Gateway Mock HTTP={args.host}:{args.port}，Modbus={args.modbus_host}:{args.modbus_port}，"
            f"upstream={args.upstream_url}"
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
            modbus_thread.join(timeout=2)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
