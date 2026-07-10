"""Active task execution against two independent v3 Runtime services."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .inference_normalizer import normalize_inference_result
from .runtime_client import HttpClient, RuntimeClient, UpstreamError
from production.carton_line.tasks.carton_partition_check import algorithm as partition_algorithm
from production.carton_line.tasks.carton_tube_check import algorithm as tube_algorithm


@dataclass
class TaskExecution:
    task: str
    decision: dict[str, Any]
    runtime_result: dict[str, Any]
    normalized_payload: dict[str, Any]
    rgb_bytes: bytes = b""
    depth_bytes: bytes = b""


class ProductionAlgorithms:
    """Configured task algorithms for this production line."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.partition = partition_algorithm
        self.tube = tube_algorithm
        self.partition.configure(config["partition"].get("algorithm", {}))
        self.tube.configure(config["tube"].get("algorithm", {}))


class TaskExecutor:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        timeout_s = int(config["service"]["request_timeout_ms"]) / 1000.0
        self.partition_runtime = RuntimeClient(config["runtimes"]["partition"]["url"], timeout_s)
        self.tube_runtime = RuntimeClient(config["runtimes"]["tube"]["url"], timeout_s)
        self.http = HttpClient(timeout_s=timeout_s)
        self.algorithms = ProductionAlgorithms(config)
        self.template_path = Path(str(config["partition"]["template_path"]))
        if not self.template_path.is_file():
            raise FileNotFoundError(f"隔板模板不存在: {self.template_path}")
        self.depth_url = str(config["camera_bridge"]["depth_url"])
        self.save_enabled = bool(config["debug"].get("save_every_trigger", True))
        self.save_root = Path(str(config["debug"]["save_root"]))

    @staticmethod
    def _model_values(result: Mapping[str, Any]) -> set[str]:
        model = result.get("model") if isinstance(result.get("model"), Mapping) else {}
        values = set()
        for key in ("model_id", "model_name", "package_id", "model_dir", "path"):
            value = model.get(key)
            if value:
                values.add(str(value))
                values.add(Path(str(value)).name)
        return values

    def _validate(self, task: str, result: Mapping[str, Any]) -> None:
        runtime_config = self.config["runtimes"][task]
        task_type = str(result.get("task_type") or "").lower()
        accepted_types = set(runtime_config.get("accepted_task_types", []))
        if accepted_types and task_type not in accepted_types:
            raise ValueError(f"{task} Runtime task_type={task_type!r} 不在白名单 {sorted(accepted_types)}")
        accepted_models = set(runtime_config.get("accepted_model_ids", [])) | set(
            runtime_config.get("accepted_model_names", [])
        )
        if accepted_models and not (accepted_models & self._model_values(result)):
            raise ValueError(
                f"{task} Runtime 当前模型不在白名单: current={sorted(self._model_values(result))}, "
                f"accepted={sorted(accepted_models)}"
            )

    @staticmethod
    def _decision(task: str, result: dict[str, Any], runtime_result: Mapping[str, Any], elapsed_ms: float) -> dict[str, Any]:
        final_result = str(result.get("final_result") or "ERROR").upper()
        ok = final_result == "OK"
        return {
            "schema_version": "1.0",
            "message_type": "app_decision",
            "status": "ok",
            "app_id": "carton_partition_check" if task in {"partition", "coordinate"} else "carton_tube_check",
            "task": task,
            "timestamp_ms": int(time.time() * 1000),
            "frame_id": runtime_result.get("frame_id"),
            "result_id": runtime_result.get("result_id"),
            "final_code": 1 if ok else 2,
            "final_label": final_result,
            "ok": ok,
            "reason": result.get("reason") or "UNKNOWN",
            "object_count": result.get("valid_cell_count", result.get("valid_prediction_count", 0)),
            "timing": {"gateway_task_ms": round(elapsed_ms, 3)},
            "details": result,
        }

    def run_partition(self, task: str = "partition") -> TaskExecution:
        started = time.monotonic()
        runtime_result = self.partition_runtime.infer_once()
        self._validate("partition", runtime_result)
        payload = normalize_inference_result(runtime_result)
        result = self.algorithms.partition.analyze(payload, template_path=self.template_path)
        rgb = self._safe_snapshot(self.partition_runtime)
        result["runtime"] = self._runtime_summary(runtime_result)
        decision = self._decision(task, result, runtime_result, (time.monotonic() - started) * 1000.0)
        execution = TaskExecution(task, decision, runtime_result, payload, rgb_bytes=rgb)
        self.save_debug(execution)
        return execution

    def run_tube(self, region: str = "all", trigger_cmd: int = 3) -> TaskExecution:
        started = time.monotonic()
        runtime_result = self.tube_runtime.infer_once()
        self._validate("tube", runtime_result)
        payload = normalize_inference_result(runtime_result)
        depth_bytes = self.http.get_bytes(self.depth_url)
        depth = self.algorithms.tube.decode_depth_png(depth_bytes)
        result = self.algorithms.tube.analyze(payload, depth, region=region)
        result["trigger_cmd"] = int(trigger_cmd)
        result["trigger_region"] = region
        result["runtime"] = self._runtime_summary(runtime_result)
        rgb = self._safe_snapshot(self.tube_runtime)
        decision = self._decision("tube", result, runtime_result, (time.monotonic() - started) * 1000.0)
        execution = TaskExecution("tube", decision, runtime_result, payload, rgb_bytes=rgb, depth_bytes=depth_bytes)
        self.save_debug(execution)
        return execution

    @staticmethod
    def _runtime_summary(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "device_id": result.get("device_id"),
            "component": result.get("component"),
            "frame_id": result.get("frame_id"),
            "result_id": result.get("result_id"),
            "task_type": result.get("task_type"),
            "model": result.get("model"),
            "timing": result.get("timing"),
        }

    @staticmethod
    def _safe_snapshot(runtime: RuntimeClient) -> bytes:
        try:
            return runtime.snapshot()
        except UpstreamError:
            return b""

    def save_debug(self, execution: TaskExecution) -> None:
        if not self.save_enabled:
            return
        folder = self.save_root / execution.task
        folder.mkdir(parents=True, exist_ok=True)
        if execution.rgb_bytes:
            (folder / "rgb.jpg").write_bytes(execution.rgb_bytes)
        if execution.depth_bytes:
            (folder / "depth.png").write_bytes(execution.depth_bytes)
        (folder / "runtime_result.json").write_text(
            json.dumps(execution.runtime_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (folder / "normalized_payload.json").write_text(
            json.dumps(execution.normalized_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (folder / "decision.json").write_text(
            json.dumps(execution.decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            details = execution.decision["details"]
            if execution.task in {"partition", "coordinate"} and execution.rgb_bytes:
                self.algorithms.partition.draw_overlay(execution.rgb_bytes, details, folder / "overlay.jpg")
            elif execution.task == "tube" and execution.rgb_bytes:
                self.algorithms.tube.draw_tube_overlay(
                    execution.rgb_bytes, execution.normalized_payload, details, folder / "overlay.jpg"
                )
        except Exception:
            # Debug rendering must never change the PLC result.
            pass
