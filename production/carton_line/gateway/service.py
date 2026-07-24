#!/usr/bin/env python3
"""Unified Modbus-TCP gateway for partition, tube and coordinate tasks."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urlsplit

from edge.gateway_adapter.gateway_message import timestamp_ms
from edge.modbus_adapter.modbus_tcp_server import ModbusTcpServer, start_modbus_server

from .config import load_config
from .coordinate_mapper import CoordinateMapper
from .register_bank import (
    ProtocolRegisterBank,
    REG_COORD_BASE,
    REG_COORD_RESULT,
    REG_PARTITION_RESULT,
    REG_PRODUCT_RESULT,
    REG_TRIGGER_COORD,
    REG_TRIGGER_PARTITION,
    REG_TRIGGER_PRODUCT,
    REG_VISION_HEARTBEAT,
    RESULT_NG,
    RESULT_NONE,
    RESULT_OK,
)
from .task_executor import TaskExecution, TaskExecutor


TUBE_TRIGGER_TO_REGION = {1: "left", 2: "right", 3: "all"}


class GatewayBusyError(RuntimeError):
    pass


class GatewayState:
    def __init__(self, config: Mapping[str, Any], bank: ProtocolRegisterBank) -> None:
        self.config = config
        self.bank = bank
        self.started_at = time.monotonic()
        self.lock = threading.RLock()
        self.busy = False
        self.active_task: str | None = None
        self.latest_decisions: dict[str, dict[str, Any]] = {}
        self.latest_gateway_message: dict[str, Any] | None = None
        self.latest_gateway_messages: dict[str, dict[str, Any]] = {}
        self.last_error: dict[str, Any] | None = None
        self.counters: dict[str, int] = defaultdict(int)
        self.sequence = 0

    def begin(self, task: str) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.active_task = task
            self.counters["task_attempts"] += 1
            self.counters[f"{task}_attempts"] += 1
            return True

    def finish(self) -> None:
        with self.lock:
            self.busy = False
            self.active_task = None

    def record_success(self, task: str, decision: dict[str, Any], message: dict[str, Any]) -> None:
        with self.lock:
            self.sequence += 1
            self.latest_decisions[task] = decision
            self.latest_gateway_message = message
            self.latest_gateway_messages[task] = message
            self.last_error = None
            self.counters["task_success"] += 1
            self.counters[f"{task}_success"] += 1

    def record_failure(self, task: str, decision: dict[str, Any], message: dict[str, Any], error: Exception) -> None:
        with self.lock:
            self.sequence += 1
            self.latest_decisions[task] = decision
            self.latest_gateway_message = message
            self.latest_gateway_messages[task] = message
            self.last_error = {
                "code": type(error).__name__,
                "message": str(error),
                "task": task,
                "timestamp_ms": timestamp_ms(),
            }
            self.counters["task_failure"] += 1
            self.counters[f"{task}_failure"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema_version": "1.0",
                "message_type": "gateway_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "device_id": self.config["device_id"],
                "component": self.config["component"],
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "active_task": self.active_task,
                "modbus": dict(self.config["modbus"]),
                "runtimes": {
                    key: {"url": value["url"], "accepted_task_types": value["accepted_task_types"]}
                    for key, value in self.config["runtimes"].items()
                },
                "latest_decisions": dict(self.latest_decisions),
                "latest_gateway_message": self.latest_gateway_message,
                "latest_gateway_messages": dict(self.latest_gateway_messages),
                "last_error": self.last_error,
                "counters": dict(self.counters),
            }


class RobotProtocolService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.bank = ProtocolRegisterBank(
            address_base=int(config["modbus"]["address_base"]),
            register_count=int(config["modbus"]["register_count"]),
        )
        self.state = GatewayState(config, self.bank)
        self.executor = TaskExecutor(config)
        self.coordinates = CoordinateMapper(config["coordinates"])
        self.stop_event = threading.Event()
        self.last_commands = {
            REG_TRIGGER_PARTITION: 0,
            REG_TRIGGER_PRODUCT: 0,
            REG_TRIGGER_COORD: 0,
        }
        self.modbus_server: ModbusTcpServer | None = None
        self.modbus_thread: threading.Thread | None = None

    def start_background(self) -> None:
        if bool(self.config["modbus"].get("enabled", True)):
            self.modbus_server, self.modbus_thread = start_modbus_server(
                str(self.config["modbus"]["host"]), int(self.config["modbus"]["port"]), self.bank
            )
        threading.Thread(target=self._heartbeat_loop, name="robot-gateway-heartbeat", daemon=True).start()
        threading.Thread(target=self._poll_loop, name="robot-gateway-poller", daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.modbus_server is not None:
            self.modbus_server.shutdown()
            self.modbus_server.server_close()

    def _heartbeat_loop(self) -> None:
        heartbeat = 0
        interval = int(self.config["modbus"]["heartbeat_interval_ms"]) / 1000.0
        maximum = int(self.config["modbus"]["heartbeat_max"])
        while not self.stop_event.wait(interval):
            heartbeat = 0 if heartbeat >= maximum else heartbeat + 1
            self.bank.set(REG_VISION_HEARTBEAT, heartbeat)

    def _clear_when_idle(self, trigger: int, result_register: int) -> None:
        if self.bank.get(trigger) == 0:
            self.last_commands[trigger] = 0
            self.bank.set(result_register, RESULT_NONE)

    def _poll_loop(self) -> None:
        interval = int(self.config["service"]["poll_interval_ms"]) / 1000.0
        while not self.stop_event.wait(interval):
            try:
                self._clear_when_idle(REG_TRIGGER_PARTITION, REG_PARTITION_RESULT)
                self._clear_when_idle(REG_TRIGGER_PRODUCT, REG_PRODUCT_RESULT)
                self._clear_when_idle(REG_TRIGGER_COORD, REG_COORD_RESULT)
                if self.state.busy:
                    continue
                partition_cmd = self.bank.get(REG_TRIGGER_PARTITION)
                tube_cmd = self.bank.get(REG_TRIGGER_PRODUCT)
                coord_cmd = self.bank.get(REG_TRIGGER_COORD)
                if partition_cmd == 1 and self.last_commands[REG_TRIGGER_PARTITION] == 0:
                    if self.state.begin("partition"):
                        self.last_commands[REG_TRIGGER_PARTITION] = 1
                        threading.Thread(
                            target=self._execute_reserved, args=("partition", None, "modbus"), daemon=True
                        ).start()
                elif tube_cmd in TUBE_TRIGGER_TO_REGION and self.last_commands[REG_TRIGGER_PRODUCT] == 0:
                    if self.state.begin("tube"):
                        self.last_commands[REG_TRIGGER_PRODUCT] = int(tube_cmd)
                        threading.Thread(
                            target=self._execute_reserved, args=("tube", int(tube_cmd), "modbus"), daemon=True
                        ).start()
                elif coord_cmd == 1 and self.last_commands[REG_TRIGGER_COORD] == 0:
                    if self.state.begin("coordinate"):
                        self.last_commands[REG_TRIGGER_COORD] = 1
                        threading.Thread(
                            target=self._execute_reserved, args=("coordinate", None, "modbus"), daemon=True
                        ).start()
            except Exception as error:
                self.state.counters["poll_errors"] += 1
                self.state.last_error = {
                    "code": "POLL_ERROR", "message": str(error), "timestamp_ms": timestamp_ms()
                }

    def execute(self, task: str, trigger_cmd: int | None = None, source: str = "http") -> dict[str, Any]:
        if not self.state.begin(task):
            raise GatewayBusyError(f"Gateway 正在执行 {self.state.active_task}")
        return self._execute_reserved(task, trigger_cmd, source)

    def _execute_reserved(self, task: str, trigger_cmd: int | None, source: str) -> dict[str, Any]:
        result_register = {
            "partition": REG_PARTITION_RESULT,
            "tube": REG_PRODUCT_RESULT,
            "coordinate": REG_COORD_RESULT,
        }[task]
        self.bank.set(result_register, RESULT_NONE)
        try:
            if task == "partition":
                execution = self.executor.run_partition("partition")
            elif task == "tube":
                command = int(trigger_cmd or 3)
                if command not in TUBE_TRIGGER_TO_REGION:
                    raise ValueError("纸筒 trigger_cmd 必须为 1、2 或 3")
                execution = self.executor.run_tube(TUBE_TRIGGER_TO_REGION[command], command)
            elif task == "coordinate":
                execution = self.executor.run_partition("coordinate")
                details = execution.decision["details"]
                original = {
                    "final_result": details.get("final_result"),
                    "reason": details.get("reason"),
                    "valid_cell_count": details.get("valid_cell_count"),
                }
                updated = self.coordinates.write(self.bank, details, execution.normalized_payload)
                details["coord_original_status"] = original
                if self.coordinates.always_ok():
                    details["coord_result_override"] = {
                        "enabled": True,
                        "policy": "partial_update_always_ok",
                        "updated_slots": updated,
                    }
                    details["final_result"] = "OK"
                    details["reason"] = "NONE"
                    execution.decision["final_code"] = RESULT_OK
                    execution.decision["final_label"] = "OK"
                    execution.decision["ok"] = True
                    execution.decision["reason"] = "NONE"
                self.executor.save_debug(execution)
            else:
                raise ValueError(f"未知任务: {task}")

            code = RESULT_OK if bool(execution.decision.get("ok")) else RESULT_NG
            self.bank.set(result_register, code)
            message = self._gateway_message(execution, source, code)
            self.state.record_success(task, execution.decision, message)
            return execution.decision
        except Exception as error:
            self.bank.set(result_register, RESULT_NG)
            decision = self._error_decision(task, error)
            message = self._gateway_message_from_decision(decision, source, RESULT_NG)
            self.state.record_failure(task, decision, message, error)
            traceback.print_exc()
            return decision
        finally:
            self.state.finish()

    def _error_decision(self, task: str, error: Exception) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "message_type": "app_decision",
            "status": "error",
            "app_id": "carton_partition_check" if task in {"partition", "coordinate"} else "carton_tube_check",
            "task": task,
            "timestamp_ms": timestamp_ms(),
            "final_code": RESULT_NG,
            "final_label": "ERROR",
            "ok": False,
            "reason": type(error).__name__,
            "error": {"code": type(error).__name__, "message": str(error), "recoverable": True},
            "details": {},
        }

    def _register_rows(self, task: str) -> list[dict[str, Any]]:
        snapshot = self.bank.logical_snapshot()
        if task == "tube":
            allowed = {0, 2, 100, 102}
        elif task == "partition":
            allowed = {0, 1, 100, 101}
        elif task == "coordinate":
            allowed = {0, 3, 100, 103} | set(range(REG_COORD_BASE, REG_COORD_BASE + 80))
        else:
            return snapshot
        return [row for row in snapshot if int(row["logical_address"]) in allowed]

    def _gateway_message(self, execution: TaskExecution, source: str, code: int) -> dict[str, Any]:
        return self._gateway_message_from_decision(execution.decision, source, code)

    def _gateway_message_from_decision(self, decision: Mapping[str, Any], source: str, code: int) -> dict[str, Any]:
        task = str(decision.get("task") or "unknown")
        return {
            "schema_version": "1.0",
            "message_type": "gateway_message",
            "status": "ok" if code == RESULT_OK else "error",
            "device_id": self.config["device_id"],
            "component": self.config["component"],
            "timestamp_ms": timestamp_ms(),
            "protocol": "modbus_tcp",
            "source": source,
            "task": task,
            "app_id": decision.get("app_id"),
            "frame_id": decision.get("frame_id"),
            "result_id": decision.get("result_id"),
            "final_code": code,
            "final_label": decision.get("final_label"),
            "ok": bool(decision.get("ok")),
            "reason": decision.get("reason"),
            "registers": self._register_rows(task),
        }

    def gateway_snapshot(self) -> dict[str, Any]:
        snapshot = self.state.snapshot()
        snapshot["coordinate_mapping"] = self.coordinates.summary()
        return snapshot

    def app_snapshot(self, task: str) -> dict[str, Any]:
        gateway = self.gateway_snapshot()
        latest = gateway["latest_decisions"].get(task)
        app_id = "carton_tube_check" if task == "tube" else "carton_partition_check"
        return {
            "schema_version": "1.0",
            "message_type": "app_status",
            "status": "ok",
            "health": gateway["health"],
            "app_id": app_id,
            "app_instance_id": f"{app_id}-production",
            "component": f"{app_id}_app",
            "device_id": self.config["device_id"],
            "uptime_s": gateway["uptime_s"],
            "busy": gateway["busy"],
            "active_task": gateway["active_task"],
            "latest_decision": latest,
            "latest_gateway_message": gateway["latest_gateway_messages"].get(task),
            "register_snapshot": self._register_rows(task),
            "counters": gateway["counters"],
            "last_error": gateway["last_error"],
        }


class GatewayHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: RobotProtocolService, app_task: str | None = None) -> None:
        self.gateway_service = service
        self.app_task = app_task
        super().__init__(address, GatewayRequestHandler)


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server: GatewayHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        service = self.server.gateway_service
        if self.server.app_task is not None:
            self._app_get(path, self.server.app_task)
            return
        if path == "/health":
            status = service.state.snapshot()
            self._json(200, {
                "schema_version": "1.0", "message_type": "gateway_health", "status": "ok",
                "health": status["health"], "busy": status["busy"], "timestamp_ms": timestamp_ms(),
            })
        elif path == "/api/gateway/status":
            self._json(200, service.gateway_snapshot())
        elif path == "/api/gateway/registers":
            self._json(200, {
                "schema_version": "1.0", "message_type": "gateway_register_snapshot",
                "timestamp_ms": timestamp_ms(), "registers": service.bank.logical_snapshot(),
            })
        elif path == "/api/gateway/register_map":
            self._json(200, {
                "schema_version": "1.0", "message_type": "gateway_register_map",
                "registers": service.bank.logical_snapshot(),
            })
        elif path == "/api/gateway/tasks":
            self._json(200, {
                "schema_version": "1.0", "message_type": "gateway_tasks",
                "latest_decisions": service.state.snapshot()["latest_decisions"],
            })
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在")

    def _app_get(self, path: str, task: str) -> None:
        service = self.server.gateway_service
        app = service.app_snapshot(task)
        if path == "/health":
            self._json(200, {
                "schema_version": "1.0", "message_type": "app_health", "status": "ok",
                "health": app["health"], "app_id": app["app_id"], "device_id": app["device_id"],
                "timestamp_ms": timestamp_ms(), "modbus": service.config["modbus"],
            })
        elif path == "/api/app/status":
            self._json(200, app)
        elif path == "/api/app/registers":
            self._json(200, {
                "schema_version": "1.0", "message_type": "app_register_snapshot",
                "timestamp_ms": timestamp_ms(), "registers": app["register_snapshot"],
            })
        elif path == "/api/app/latest_decision":
            if app["latest_decision"] is None:
                self._error(404, "LATEST_DECISION_NOT_FOUND", "尚未生成业务决策")
            else:
                self._json(200, app["latest_decision"])
        elif path == "/api/app/latest_gateway_message":
            if app["latest_gateway_message"] is None:
                self._error(404, "LATEST_GATEWAY_MESSAGE_NOT_FOUND", "尚未生成 GatewayMessage")
            else:
                self._json(200, app["latest_gateway_message"])
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        body = self._body_json()
        if body is None:
            return
        service = self.server.gateway_service
        try:
            if self.server.app_task is not None:
                if path != "/api/app/evaluate_once":
                    self._error(404, "ROUTE_NOT_FOUND", "接口不存在")
                    return
                if self.server.app_task == "tube":
                    command = self._tube_command(body)
                    decision = service.execute("tube", command, "app_http")
                else:
                    decision = service.execute("partition", None, "app_http")
                self._json(200 if decision.get("status") == "ok" else 502, decision)
                return

            if path == "/api/gateway/trigger/partition":
                decision = service.execute("partition", None, "gateway_http")
            elif path == "/api/gateway/trigger/tube":
                decision = service.execute("tube", self._tube_command(body), "gateway_http")
            elif path == "/api/gateway/trigger/coordinate":
                decision = service.execute("coordinate", None, "gateway_http")
            else:
                self._error(404, "ROUTE_NOT_FOUND", "接口不存在")
                return
            self._json(200 if decision.get("status") == "ok" else 502, decision)
        except GatewayBusyError as error:
            self._error(409, "GATEWAY_BUSY", str(error))
        except ValueError as error:
            self._error(400, "INVALID_REQUEST", str(error))

    @staticmethod
    def _tube_command(body: Mapping[str, Any]) -> int:
        if "trigger_cmd" in body:
            command = int(body["trigger_cmd"])
        else:
            region = str(body.get("region") or "all").lower()
            reverse = {"left": 1, "right": 2, "all": 3}
            if region not in reverse:
                raise ValueError("region 必须为 left、right 或 all")
            command = reverse[region]
        if command not in TUBE_TRIGGER_TO_REGION:
            raise ValueError("trigger_cmd 必须为 1、2 或 3")
        return command

    def _body_json(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法")
            return None
        if not 0 <= length <= 1024 * 1024:
            self._error(413, "REQUEST_BODY_TOO_LARGE", "请求体超过限制")
            return None
        if length == 0:
            return {}
        try:
            document = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "INVALID_JSON", "请求体必须是 JSON 对象")
            return None
        if not isinstance(document, dict):
            self._error(400, "INVALID_JSON", "请求体顶层必须是对象")
            return None
        return document

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {
            "schema_version": "1.0", "message_type": "error", "status": "error",
            "timestamp_ms": timestamp_ms(), "error": {"code": code, "message": message, "recoverable": True},
        })

    def _json(self, status: int, document: Mapping[str, Any]) -> None:
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps v3 unified Robot Protocol Gateway")
    parser.add_argument("--config", default="production/carton_line/config/line.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    service = RobotProtocolService(config)
    host = str(config["service"]["listen_host"])
    servers = [
        GatewayHttpServer((host, int(config["service"]["listen_port"])), service, None),
        GatewayHttpServer((host, int(config["service"]["partition_app_port"])), service, "partition"),
        GatewayHttpServer((host, int(config["service"]["tube_app_port"])), service, "tube"),
    ]
    service.start_background()

    def shutdown(_signum: int, _frame: object) -> None:
        service.stop()
        for server in servers:
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    print(
        "VisionOps Robot Gateway: "
        f"HTTP={host}:{config['service']['listen_port']} "
        f"partition_app={host}:{config['service']['partition_app_port']} "
        f"tube_app={host}:{config['service']['tube_app_port']} "
        f"Modbus={config['modbus']['host']}:{config['modbus']['port']}",
        flush=True,
    )
    coordinate_summary = service.coordinates.summary()
    print(
        "Coordinate mapper: "
        f"output_frame={coordinate_summary['output_frame']} "
        f"register_order={coordinate_summary['register_order']} "
        f"dual_arm={int(coordinate_summary['dual_arm_enabled'])} "
        f"four_zone={int(coordinate_summary['four_zone_enabled'])} "
        f"left_cols={coordinate_summary['left_columns']} "
        f"right_cols={coordinate_summary['right_columns']} "
        f"top_rows={coordinate_summary['top_rows']} "
        f"bottom_rows={coordinate_summary['bottom_rows']}",
        flush=True,
    )
    for transform_name in (
        "left_top_affine", "left_bottom_affine",
        "right_top_affine", "right_bottom_affine",
    ):
        affine = coordinate_summary["transforms"].get(transform_name)
        if affine is not None:
            print(
                f"Coordinate {transform_name}: "
                f"A=[[{affine['a00']:.8f},{affine['a01']:.8f}],"
                f"[{affine['a10']:.8f},{affine['a11']:.8f}]] "
                f"b=[{affine['b0']:.8f},{affine['b1']:.8f}]",
                flush=True,
            )
    try:
        servers[0].serve_forever(poll_interval=0.2)
    finally:
        service.stop()
        for server in servers:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
