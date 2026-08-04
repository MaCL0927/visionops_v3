#!/usr/bin/env python3
"""M37.4 depth-layer-first hybrid trigger service for foam-ring exact RGB-D geometry.

The service keeps the synchronized RGB-D cache, Runtime client, geometry config
and box calibration resident. Explicit trigger requests are serialized by one
worker and retained in a bounded request registry, so a request_id is never
replaced by continuous/latest-frame traffic.
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    OnlineGeometryProcessor,
    OnlineProcessResult,
)

MAX_HTTP_BODY = 1024 * 1024


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * float(percentile)
    low = int(index)
    high = min(len(ordered) - 1, low + 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def load_service_settings(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    section = raw_config.get("online_service") or {}
    if not isinstance(section, Mapping):
        raise ValueError("line.yaml 中 online_service 必须是对象")
    return {
        "enabled": bool(section.get("enabled", True)),
        "listen_host": str(section.get("listen_host") or "0.0.0.0"),
        "listen_port": int(section.get("listen_port", 19213)),
        "trigger_queue_capacity": max(1, int(section.get("trigger_queue_capacity", 4))),
        "geometry_queue_capacity": max(1, int(section.get("geometry_queue_capacity", 4))),
        "request_registry_capacity": max(8, int(section.get("request_registry_capacity", 64))),
        "default_wait_timeout_ms": max(1000, int(section.get("default_wait_timeout_ms", 15000))),
        "maximum_wait_timeout_ms": max(1000, int(section.get("maximum_wait_timeout_ms", 60000))),
        "default_save_debug": bool(section.get("default_save_debug", False)),
        "latest_overlay_enabled": bool(section.get("latest_overlay_enabled", True)),
        "overlay_jpeg_quality": min(100, max(40, int(section.get("overlay_jpeg_quality", 90)))),
        "runtime_status_ttl_ms": max(0, int(section.get("runtime_status_ttl_ms", 1000))),
        "output_root": str(section.get("output_root") or DEFAULT_OUTPUT_ROOT),
        "max_http_body_bytes": max(1024, int(section.get("max_http_body_bytes", MAX_HTTP_BODY))),
    }


@dataclass
class TriggerJob:
    request_id: str
    save_debug: bool
    submitted_timestamp_ms: int
    submitted_monotonic: float
    status: str = "queued"
    started_timestamp_ms: int = 0
    inference_completed_timestamp_ms: int = 0
    geometry_started_timestamp_ms: int = 0
    completed_timestamp_ms: int = 0
    started_monotonic: float = 0.0
    result: Dict[str, Any] | None = None
    error: Dict[str, Any] | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def status_document(self, *, replay: bool = False) -> Dict[str, Any]:
        document: Dict[str, Any] = {
            "schema_version": "1.0",
            "message_type": "foam_ring_trigger_status",
            "status": self.status,
            "request_id": self.request_id,
            "submitted_timestamp_ms": self.submitted_timestamp_ms,
            "started_timestamp_ms": self.started_timestamp_ms or None,
            "inference_completed_timestamp_ms": self.inference_completed_timestamp_ms or None,
            "geometry_started_timestamp_ms": self.geometry_started_timestamp_ms or None,
            "completed_timestamp_ms": self.completed_timestamp_ms or None,
            "save_debug": self.save_debug,
            "idempotent_replay": bool(replay),
        }
        if self.result is not None:
            result = deepcopy(self.result)
            result["idempotent_replay"] = bool(replay)
            return result
        if self.error is not None:
            document["error"] = deepcopy(self.error)
        return document


class QueueCapacityError(RuntimeError):
    pass


class FoamRingOnlineService:
    def __init__(
        self,
        *,
        processor: OnlineGeometryProcessor,
        settings: Mapping[str, Any],
    ) -> None:
        self.processor = processor
        self.settings = dict(settings)
        self.trigger_queue: queue.Queue[TriggerJob | None] = queue.Queue(
            maxsize=int(self.settings["trigger_queue_capacity"])
        )
        self.geometry_queue: queue.Queue[tuple[TriggerJob, Any] | None] = queue.Queue(
            maxsize=int(self.settings["geometry_queue_capacity"])
        )
        # Compatibility alias for older local tests/tools.
        self.queue = self.trigger_queue
        self.registry: "OrderedDict[str, TriggerJob]" = OrderedDict()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.inference_worker: threading.Thread | None = None
        self.geometry_worker: threading.Thread | None = None
        self.started_timestamp_ms = 0
        self.manual_sequence = 0
        self.latest_result: Dict[str, Any] | None = None
        self.latest_overlay_jpeg: bytes | None = None
        self.latest_overlay_timestamp_ms = 0
        self.inference_request_id: str | None = None
        self.geometry_request_id: str | None = None
        self.last_error: Dict[str, Any] | None = None
        self.accepted_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.rejected_queue_full_count = 0
        self.duplicate_request_count = 0
        self.m36_branch_count = 0
        self.m37_branch_count = 0
        self.no_target_count = 0
        self.processing_times_ms: list[float] = []
        self.trigger_queue_wait_times_ms: list[float] = []
        self.geometry_queue_wait_times_ms: list[float] = []

    @property
    def worker(self) -> threading.Thread | None:
        """Compatibility: historical code expected one worker attribute."""
        return self.geometry_worker

    @property
    def current_request_id(self) -> str | None:
        return self.geometry_request_id or self.inference_request_id

    def start(self) -> None:
        if self.geometry_worker is not None and self.geometry_worker.is_alive():
            return
        self.processor.start()
        self.stop_event.clear()
        self.started_timestamp_ms = _timestamp_ms()
        self.inference_worker = threading.Thread(
            target=self._inference_loop,
            name="foam-ring-inference-worker",
            daemon=True,
        )
        self.geometry_worker = threading.Thread(
            target=self._geometry_loop,
            name="foam-ring-geometry-worker",
            daemon=True,
        )
        self.inference_worker.start()
        self.geometry_worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._fail_pending_trigger_jobs("SERVICE_STOPPING", "服务正在停止")
        self._fail_pending_geometry_jobs("SERVICE_STOPPING", "服务正在停止")
        for target_queue in (self.trigger_queue, self.geometry_queue):
            try:
                target_queue.put_nowait(None)
            except queue.Full:
                pass
        for worker in (self.inference_worker, self.geometry_worker):
            if worker is not None:
                worker.join(timeout=5.0)
        self.processor.stop()

    def _fail_job(self, job: TriggerJob, code: str, message: str) -> None:
        with self.lock:
            if job.done.is_set():
                return
            job.status = "error"
            job.error = {"code": code, "message": message}
            job.completed_timestamp_ms = _timestamp_ms()
            self.failed_count += 1
            self.last_error = {
                "code": code,
                "message": message,
                "request_id": job.request_id,
                "timestamp_ms": job.completed_timestamp_ms,
            }
            job.done.set()

    def _fail_pending_trigger_jobs(self, code: str, message: str) -> None:
        while True:
            try:
                item = self.trigger_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is not None:
                    self._fail_job(item, code, message)
            finally:
                self.trigger_queue.task_done()

    def _fail_pending_geometry_jobs(self, code: str, message: str) -> None:
        while True:
            try:
                item = self.geometry_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is not None:
                    job, _prepared = item
                    self._fail_job(job, code, message)
            finally:
                self.geometry_queue.task_done()

    def _next_request_id(self) -> str:
        with self.lock:
            self.manual_sequence += 1
            return f"manual-{_timestamp_ms()}-{self.manual_sequence:06d}"

    @staticmethod
    def _validate_request_id(value: Any) -> str:
        request_id = str(value or "").strip()
        if not request_id:
            raise ValueError("request_id不能为空")
        if len(request_id) > 128:
            raise ValueError("request_id长度不能超过128")
        if any(ord(char) < 32 for char in request_id):
            raise ValueError("request_id包含控制字符")
        return request_id

    def submit(
        self,
        *,
        request_id: str | None,
        save_debug: bool | None,
    ) -> tuple[TriggerJob, bool]:
        resolved_id = self._validate_request_id(request_id or self._next_request_id())
        resolved_save_debug = (
            bool(self.settings["default_save_debug"])
            if save_debug is None
            else bool(save_debug)
        )
        with self.lock:
            existing = self.registry.get(resolved_id)
            if existing is not None:
                self.duplicate_request_count += 1
                self.registry.move_to_end(resolved_id)
                return existing, True
            self._trim_registry_locked()
            if self.trigger_queue.full():
                self.rejected_queue_full_count += 1
                raise QueueCapacityError("显式触发队列已满")
            job = TriggerJob(
                request_id=resolved_id,
                save_debug=resolved_save_debug,
                submitted_timestamp_ms=_timestamp_ms(),
                submitted_monotonic=time.monotonic(),
            )
            self.registry[resolved_id] = job
            self.accepted_count += 1
            self.trigger_queue.put_nowait(job)
            return job, False

    def _trim_registry_locked(self) -> None:
        capacity = int(self.settings["request_registry_capacity"])
        while len(self.registry) >= capacity:
            removable: str | None = None
            for key, job in self.registry.items():
                if job.done.is_set():
                    removable = key
                    break
            if removable is None:
                break
            self.registry.pop(removable, None)

    def get_request(self, request_id: str) -> TriggerJob | None:
        with self.lock:
            job = self.registry.get(str(request_id))
            if job is not None:
                self.registry.move_to_end(str(request_id))
            return job

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.trigger_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                self.trigger_queue.task_done()
                break
            with self.lock:
                job.status = "inferring"
                job.started_timestamp_ms = _timestamp_ms()
                job.started_monotonic = time.monotonic()
                self.inference_request_id = job.request_id
                trigger_wait_ms = (
                    job.started_monotonic - job.submitted_monotonic
                ) * 1000.0
                self.trigger_queue_wait_times_ms.append(trigger_wait_ms)
                self.trigger_queue_wait_times_ms = self.trigger_queue_wait_times_ms[-256:]
            try:
                prepared = self.processor.prepare(request_id=job.request_id)
                with self.lock:
                    job.inference_completed_timestamp_ms = _timestamp_ms()
                    job.status = "geometry_queued"
                while not self.stop_event.is_set():
                    try:
                        self.geometry_queue.put((job, prepared), timeout=0.25)
                        break
                    except queue.Full:
                        continue
                else:
                    self._fail_job(job, "SERVICE_STOPPING", "服务在几何入队前停止")
            except Exception as error:
                self._complete_error(job, error)
            finally:
                with self.lock:
                    self.inference_request_id = None
                self.trigger_queue.task_done()

    def _geometry_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.geometry_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                self.geometry_queue.task_done()
                break
            job, prepared = item
            with self.lock:
                job.status = "geometry"
                job.geometry_started_timestamp_ms = _timestamp_ms()
                self.geometry_request_id = job.request_id
                geometry_wait_ms = max(
                    0.0,
                    (time.monotonic() - prepared.prepared_monotonic) * 1000.0,
                )
                self.geometry_queue_wait_times_ms.append(geometry_wait_ms)
                self.geometry_queue_wait_times_ms = self.geometry_queue_wait_times_ms[-256:]
            try:
                process_result = self.processor.finish(
                    prepared,
                    save_debug=job.save_debug,
                    generate_overlay=bool(self.settings["latest_overlay_enabled"]),
                    stage="M37.4_depth_layered_hybrid_persistent_trigger_service",
                    geometry_queue_wait_ms=geometry_wait_ms,
                )
                service_processing_ms = max(
                    0.0,
                    (time.monotonic() - job.started_monotonic) * 1000.0,
                )
                service_result = self._compact_result(
                    process_result,
                    job=job,
                    trigger_queue_wait_ms=(
                        job.started_monotonic - job.submitted_monotonic
                    ) * 1000.0,
                    geometry_queue_wait_ms=geometry_wait_ms,
                    service_processing_ms=service_processing_ms,
                )
                with self.lock:
                    job.result = service_result
                    job.status = "ok"
                    self.latest_result = deepcopy(service_result)
                    if process_result.overlay_jpeg is not None:
                        self.latest_overlay_jpeg = bytes(process_result.overlay_jpeg)
                        self.latest_overlay_timestamp_ms = int(
                            service_result.get("capture_timestamp_ms") or _timestamp_ms()
                        )
                    self.completed_count += 1
                    branch = str(service_result.get("selected_grasp_branch") or "none")
                    if branch == "m36_mouth_visible_rim_pinch":
                        self.m36_branch_count += 1
                    elif branch == "m37_side_ring_near_visible_crown":
                        self.m37_branch_count += 1
                    else:
                        self.no_target_count += 1
                    self.last_error = None
            except Exception as error:
                self._complete_error(job, error)
            finally:
                completed_ms = _timestamp_ms()
                elapsed_ms = max(
                    0.0,
                    (time.monotonic() - job.started_monotonic) * 1000.0,
                )
                with self.lock:
                    job.completed_timestamp_ms = completed_ms
                    self.geometry_request_id = None
                    self.processing_times_ms.append(elapsed_ms)
                    self.processing_times_ms = self.processing_times_ms[-256:]
                    job.done.set()
                    self._trim_registry_locked()
                self.geometry_queue.task_done()

    def _complete_error(self, job: TriggerJob, error: Exception) -> None:
        error_document = {
            "code": type(error).__name__,
            "message": str(error),
        }
        with self.lock:
            if job.done.is_set():
                return
            job.status = "error"
            job.error = error_document
            job.completed_timestamp_ms = _timestamp_ms()
            self.failed_count += 1
            self.last_error = {
                **error_document,
                "request_id": job.request_id,
                "timestamp_ms": job.completed_timestamp_ms,
            }
            job.done.set()

    @staticmethod
    def _compact_result(
        process_result: OnlineProcessResult,
        *,
        job: TriggerJob,
        trigger_queue_wait_ms: float,
        geometry_queue_wait_ms: float,
        service_processing_ms: float,
    ) -> Dict[str, Any]:
        payload = process_result.payload
        scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else {}
        geometry_optimization = (
            scene.get("geometry_optimization")
            if isinstance(scene.get("geometry_optimization"), Mapping)
            else {}
        )
        hybrid = (
            scene.get("hybrid_grasp")
            if isinstance(scene.get("hybrid_grasp"), Mapping)
            else {}
        )
        side_branch = (
            scene.get("side_ring_branch")
            if isinstance(scene.get("side_ring_branch"), Mapping)
            else {}
        )
        hybrid_timing = (
            hybrid.get("timing_ms")
            if isinstance(hybrid.get("timing_ms"), Mapping)
            else {}
        )
        timing = payload.get("timing_ms") if isinstance(payload.get("timing_ms"), Mapping) else {}
        runtime_timing = (
            ((payload.get("runtime") or {}).get("timing"))
            if isinstance(payload.get("runtime"), Mapping)
            else {}
        )
        if not isinstance(runtime_timing, Mapping):
            runtime_timing = {}
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else None
        end_to_end_ms = (
            time.monotonic() - job.submitted_monotonic
        ) * 1000.0
        return {
            "schema_version": "1.0",
            "message_type": "foam_ring_trigger_result",
            "stage": str(payload.get("stage") or "M37.4_depth_layered_hybrid_persistent_trigger_service"),
            "status": "ok",
            "request_id": job.request_id,
            "idempotent_replay": False,
            "target_found": candidate is not None,
            "selected_grasp_branch": hybrid.get("selected_branch") or scene.get("selected_grasp_branch") or "none",
            "fallback_triggered": bool(hybrid.get("fallback_triggered", False)),
            "robot_ready": False,
            "robot_ready_reason": payload.get("robot_ready_reason"),
            "capture_timestamp_ms": payload.get("capture_timestamp_ms"),
            "rgbd_timestamp_delta_ms": (payload.get("rgbd_match") or {}).get("timestamp_delta_ms"),
            "runtime_result_id": (payload.get("runtime") or {}).get("result_id"),
            "runtime_frame_id": (payload.get("runtime") or {}).get("frame_id"),
            "scene_summary": {
                "rings_detected": scene.get("rings_detected"),
                "mouths_detected": scene.get("mouths_detected"),
                "matched_pairs": scene.get("matched_pairs"),
                "eligible_count": scene.get("eligible_count"),
                "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
                "selected_clock_hour": scene.get("selected_clock_hour"),
                "selected_clock_angle_deg_cw_from_12": scene.get("selected_clock_angle_deg_cw_from_12"),
                "selected_clock_search_batch": scene.get("selected_clock_search_batch"),
                "geometry_mode": geometry_optimization.get("mode"),
                "fully_analyzed_pair_count": geometry_optimization.get("fully_analyzed_pair_count"),
                "full_candidate_evaluated_count": geometry_optimization.get("full_candidate_evaluated_count"),
                "adaptive_fallback_used": geometry_optimization.get("adaptive_fallback_used"),
                "early_exit_triggered": geometry_optimization.get("early_exit_triggered"),
                "selected_grasp_branch": hybrid.get("selected_branch") or scene.get("selected_grasp_branch"),
                "m36_candidate_found": hybrid.get("m36_candidate_found"),
                "m37_candidate_found": hybrid.get("m37_candidate_found"),
                "m37_candidate_count": side_branch.get("candidate_count"),
                "m37_evaluated_count": side_branch.get("evaluated_count"),
                "m37_deferred_count": side_branch.get("deferred_count"),
                "m37_selected_ring_instance_id": side_branch.get("selected_ring_instance_id"),
                "selected_depth_layer_index": (scene.get("depth_layering") or {}).get("selected_layer_index"),
                "selected_surface_depth_mm": (scene.get("depth_layering") or {}).get("selected_surface_depth_mm"),
                "selected_depth_rank": (scene.get("depth_layering") or {}).get("selected_depth_rank"),
                "m37_fast_attempt_count": side_branch.get("fast_attempt_count"),
                "m37_accurate_refinement_count": side_branch.get("accurate_refinement_count"),
            },
            "candidate": deepcopy(candidate),
            "timing_ms": {
                "trigger_queue_wait_ms": round(float(trigger_queue_wait_ms), 3),
                "geometry_queue_wait_ms": round(float(geometry_queue_wait_ms), 3),
                "runtime_total_ms": runtime_timing.get("total_ms"),
                "prepare_total_ms": timing.get("prepare_total_ms"),
                "polygon_to_mask_ms": timing.get("polygon_to_mask_ms"),
                "geometry_ms": timing.get("geometry_ms"),
                "association_prepass_ms": hybrid_timing.get("association_prepass_ms"),
                "depth_preselection_ms": hybrid_timing.get("depth_preselection_ms"),
                "depth_layer_build_ms": hybrid_timing.get("depth_layer_build_ms"),
                "m36_branch_ms": hybrid_timing.get("m36_branch_ms"),
                "m37_fast_total_ms": hybrid_timing.get("m37_fast_total_ms"),
                "m37_local_accurate_ms": hybrid_timing.get("m37_local_accurate_ms"),
                # M37.3 compatibility aliases for existing clients/tests.
                "m37_candidate_filter_sort_ms": hybrid_timing.get("m37_candidate_filter_sort_ms", hybrid_timing.get("depth_preselection_ms")),
                "m37_fit_loop_ms": hybrid_timing.get("m37_fit_loop_ms", (
                    float(hybrid_timing.get("m37_fast_total_ms") or 0.0)
                    + float(hybrid_timing.get("m37_local_accurate_ms") or 0.0)
                ) if (hybrid_timing.get("m37_fast_total_ms") is not None or hybrid_timing.get("m37_local_accurate_ms") is not None) else None),
                "m37_evaluated_instance_total_ms": hybrid_timing.get("m37_evaluated_instance_total_ms"),
                "hybrid_branch_total_ms": hybrid_timing.get("total_ms"),
                "visualization_ms": timing.get("visualization_ms"),
                "save_outputs_ms": timing.get("save_outputs_ms"),
                "processor_total_ms": timing.get("total_ms"),
                "service_processing_ms": round(float(service_processing_ms), 3),
                "service_end_to_end_ms": round(float(end_to_end_ms), 3),
            },
            "files": deepcopy(payload.get("files") or {}),
            "save_debug": bool(job.save_debug),
            "submitted_timestamp_ms": job.submitted_timestamp_ms,
            "started_timestamp_ms": job.started_timestamp_ms,
            "inference_completed_timestamp_ms": job.inference_completed_timestamp_ms,
            "geometry_started_timestamp_ms": job.geometry_started_timestamp_ms,
            "completed_timestamp_ms": _timestamp_ms(),
        }

    def health_document(self) -> Dict[str, Any]:
        processor_status = self.processor.status(refresh_runtime=False)
        cache = processor_status.get("cache") if isinstance(processor_status.get("cache"), Mapping) else {}
        runtime = processor_status.get("runtime") if isinstance(processor_status.get("runtime"), Mapping) else {}
        frame_source = runtime.get("frame_source") if isinstance(runtime.get("frame_source"), Mapping) else {}
        cache_ok = bool(cache.get("running", self.processor.started)) and not cache.get("last_error")
        runtime_ok = (
            processor_status.get("runtime_error") is None
            and str((runtime.get("loaded_model") or {}).get("task_type") or "") == "segmentation"
            and str(frame_source.get("transport") or "") == "posix_shared_memory"
            and not bool(frame_source.get("fallback_active"))
        )
        inference_ok = bool(self.inference_worker is not None and self.inference_worker.is_alive())
        geometry_ok = bool(self.geometry_worker is not None and self.geometry_worker.is_alive())
        health = "ok" if cache_ok and runtime_ok and inference_ok and geometry_ok else "degraded"
        return {
            "schema_version": "1.0",
            "message_type": "foam_ring_service_health",
            "status": "ok",
            "health": health,
            "component": "foam_ring_grasp_online_service",
            "timestamp_ms": _timestamp_ms(),
            "uptime_ms": max(0, _timestamp_ms() - self.started_timestamp_ms),
            "inference_worker_alive": inference_ok,
            "geometry_worker_alive": geometry_ok,
            "cache_ok": cache_ok,
            "runtime_ok": runtime_ok,
            "busy": self.current_request_id is not None,
            "inference_request_id": self.inference_request_id,
            "geometry_request_id": self.geometry_request_id,
            "last_error": deepcopy(self.last_error),
        }

    def status_document(self) -> Dict[str, Any]:
        processor_status = self.processor.status(refresh_runtime=False)
        with self.lock:
            process_values = list(self.processing_times_ms)
            trigger_wait_values = list(self.trigger_queue_wait_times_ms)
            geometry_wait_values = list(self.geometry_queue_wait_times_ms)
            latest = deepcopy(self.latest_result)
            request_states: Dict[str, int] = {}
            for job in self.registry.values():
                request_states[job.status] = request_states.get(job.status, 0) + 1
            return {
                "schema_version": "1.0",
                "message_type": "foam_ring_service_status",
                "status": "ok",
                "timestamp_ms": _timestamp_ms(),
                "service": {
                    "started_timestamp_ms": self.started_timestamp_ms,
                    "uptime_ms": max(0, _timestamp_ms() - self.started_timestamp_ms),
                    "busy": self.current_request_id is not None,
                    "inference_request_id": self.inference_request_id,
                    "geometry_request_id": self.geometry_request_id,
                    "trigger_queue_size": self.trigger_queue.qsize(),
                    "trigger_queue_capacity": self.trigger_queue.maxsize,
                    "geometry_queue_size": self.geometry_queue.qsize(),
                    "geometry_queue_capacity": self.geometry_queue.maxsize,
                    "registry_size": len(self.registry),
                    "registry_capacity": int(self.settings["request_registry_capacity"]),
                    "request_states": request_states,
                    "accepted_count": self.accepted_count,
                    "completed_count": self.completed_count,
                    "failed_count": self.failed_count,
                    "duplicate_request_count": self.duplicate_request_count,
                    "rejected_queue_full_count": self.rejected_queue_full_count,
                    "branch_counts": {
                        "m36_mouth_visible_rim_pinch": self.m36_branch_count,
                        "m37_side_ring_near_visible_crown": self.m37_branch_count,
                        "none": self.no_target_count,
                    },
                    "default_save_debug": bool(self.settings["default_save_debug"]),
                    "latest_overlay_enabled": bool(self.settings["latest_overlay_enabled"]),
                    "latest_overlay_available": self.latest_overlay_jpeg is not None,
                    "latest_overlay_timestamp_ms": self.latest_overlay_timestamp_ms or None,
                },
                "latency_ms": {
                    "processing_p50": round(_percentile(process_values, 0.50), 3),
                    "processing_p95": round(_percentile(process_values, 0.95), 3),
                    "processing_max": round(max(process_values), 3) if process_values else 0.0,
                    "trigger_queue_wait_p50": round(_percentile(trigger_wait_values, 0.50), 3),
                    "trigger_queue_wait_p95": round(_percentile(trigger_wait_values, 0.95), 3),
                    "geometry_queue_wait_p50": round(_percentile(geometry_wait_values, 0.50), 3),
                    "geometry_queue_wait_p95": round(_percentile(geometry_wait_values, 0.95), 3),
                },
                "processor": processor_status,
                "latest_result": latest,
                "last_error": deepcopy(self.last_error),
            }

    def latest_result_document(self) -> Dict[str, Any]:
        with self.lock:
            if self.latest_result is None:
                return {
                    "schema_version": "1.0",
                    "message_type": "foam_ring_trigger_result",
                    "status": "empty",
                }
            return deepcopy(self.latest_result)

    def snapshot(self) -> tuple[bytes | None, int]:
        with self.lock:
            return (
                bytes(self.latest_overlay_jpeg) if self.latest_overlay_jpeg is not None else None,
                int(self.latest_overlay_timestamp_ms),
            )


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class FoamRingRequestHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsFoamRing/1.0"

    @property
    def service(self) -> FoamRingOnlineService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, code: int, document: Mapping[str, Any]) -> None:
        body = _json_bytes(document)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, body: bytes, timestamp_ms: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-VisionOps-Capture-Timestamp-Ms", str(timestamp_ms))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length无效") from error
        limit = int(self.service.settings["max_http_body_bytes"])
        if size < 0 or size > limit:
            raise ValueError("请求体超过大小限制")
        raw = self.rfile.read(size) if size else b"{}"
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("请求体必须是JSON对象") from error
        if not isinstance(document, dict):
            raise ValueError("请求JSON顶层必须是对象")
        return document

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, self.service.health_document())
            return
        if path in {"/status", "/api/foam_ring/status"}:
            self._send_json(200, self.service.status_document())
            return
        if path == "/api/foam_ring/latest_result":
            self._send_json(200, self.service.latest_result_document())
            return
        if path.startswith("/api/foam_ring/request/"):
            request_id = unquote(path[len("/api/foam_ring/request/"):])
            job = self.service.get_request(request_id)
            if job is None:
                self._send_json(404, {
                    "status": "error",
                    "error": {"code": "REQUEST_NOT_FOUND", "message": request_id},
                })
            else:
                self._send_json(200, job.status_document())
            return
        if path == "/snapshot.jpg":
            snapshot, timestamp_ms = self.service.snapshot()
            if snapshot is None:
                self._send_json(404, {
                    "status": "empty",
                    "error": {"code": "SNAPSHOT_NOT_AVAILABLE", "message": "尚无触发结果"},
                })
            else:
                self._send_jpeg(snapshot, timestamp_ms)
            return
        self._send_json(404, {
            "status": "error",
            "error": {"code": "NOT_FOUND", "message": path},
        })

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/foam_ring/infer_once":
            self._send_json(404, {
                "status": "error",
                "error": {"code": "NOT_FOUND", "message": path},
            })
            return
        try:
            document = self._read_json()
            request_id = document.get("request_id")
            save_debug = document.get("save_debug")
            if save_debug is not None and not isinstance(save_debug, bool):
                raise ValueError("save_debug必须是布尔值")
            wait = document.get("wait", True)
            if not isinstance(wait, bool):
                raise ValueError("wait必须是布尔值")
            timeout_ms = int(
                document.get("timeout_ms")
                or self.service.settings["default_wait_timeout_ms"]
            )
            timeout_ms = max(0, min(
                timeout_ms,
                int(self.service.settings["maximum_wait_timeout_ms"]),
            ))
            job, replay = self.service.submit(
                request_id=str(request_id) if request_id is not None else None,
                save_debug=save_debug,
            )
            if wait and not job.done.is_set():
                job.done.wait(float(timeout_ms) / 1000.0)
            if job.done.is_set():
                code = 200 if job.status == "ok" else 500
                self._send_json(code, job.status_document(replay=replay))
            else:
                self._send_json(202, job.status_document(replay=replay))
        except QueueCapacityError as error:
            self._send_json(429, {
                "status": "busy",
                "error": {"code": "TRIGGER_QUEUE_FULL", "message": str(error)},
            })
        except (ValueError, TypeError) as error:
            self._send_json(400, {
                "status": "error",
                "error": {"code": "INVALID_REQUEST", "message": str(error)},
            })
        except Exception as error:
            self._send_json(500, {
                "status": "error",
                "error": {"code": type(error).__name__, "message": str(error)},
            })


def run(
    *,
    config_path: Path,
    runtime_url: str | None,
    host: str | None,
    port: int | None,
    geometry_mode: str | None,
) -> int:
    raw_config = load_yaml(config_path.expanduser().resolve())
    settings = load_service_settings(raw_config)
    if not bool(settings["enabled"]):
        raise ValueError("online_service.enabled=false")
    if host is not None:
        settings["listen_host"] = str(host)
    if port is not None:
        settings["listen_port"] = int(port)
    processor = OnlineGeometryProcessor(
        config_path=config_path,
        runtime_url=runtime_url,
        output_root=Path(str(settings["output_root"])),
        geometry_mode=geometry_mode,
        runtime_status_ttl_ms=int(settings["runtime_status_ttl_ms"]),
        overlay_jpeg_quality=int(settings["overlay_jpeg_quality"]),
    )
    service = FoamRingOnlineService(processor=processor, settings=settings)
    server = ReusableThreadingHTTPServer(
        (str(settings["listen_host"]), int(settings["listen_port"])),
        FoamRingRequestHandler,
    )
    server.service = service  # type: ignore[attr-defined]
    stop_once = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    service.start()
    thread = threading.Thread(
        target=server.serve_forever,
        name="foam-ring-http",
        daemon=True,
    )
    thread.start()
    print(
        "Foam Ring M37.4 Depth-Layered Hybrid Service started: http={}:{} runtime={} "
        "m36_mode={} hybrid={} queue={}".format(
            settings["listen_host"],
            settings["listen_port"],
            processor.runtime_url,
            (processor.raw_config.get("geometry_optimization") or {}).get("mode"),
            bool((processor.raw_config.get("hybrid_grasp") or {}).get("enabled", False)),
            settings["trigger_queue_capacity"],
        ),
        flush=True,
    )
    try:
        while not stop_once.wait(1.0):
            pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        service.stop()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M37.4 depth-layered foam-ring persistent trigger service (M36.5-compatible API)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-url")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--geometry-mode",
        choices=("first_valid", "staged", "exhaustive"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(
            config_path=args.config,
            runtime_url=args.runtime_url,
            host=args.host,
            port=args.port,
            geometry_mode=args.geometry_mode,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[FAIL] M37.4 hybrid service startup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
