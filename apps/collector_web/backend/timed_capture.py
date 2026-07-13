"""Collector 后台定时采图控制器。"""

from __future__ import annotations

import threading
import time
from typing import Any

from .dataset_manager import save_runtime_snapshot
from .response_utils import timestamp_ms
from .runtime_client import RuntimeClient


class TimedCaptureController:
    """在 Collector 进程内按固定间隔保存 Runtime 快照。"""

    def __init__(self, runtime_client: RuntimeClient) -> None:
        self._runtime_client = runtime_client
        self._condition = threading.Condition()
        self._enabled = False
        self._closed = False
        self._interval_seconds = 10.0
        self._generation = 0
        self._capture_count = 0
        self._failure_count = 0
        self._started_at_ms: int | None = None
        self._next_capture_at_ms: int | None = None
        self._last_capture_at_ms: int | None = None
        self._last_image: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="visionops-timed-capture",
            daemon=True,
        )
        self._thread.start()

    def start(self, interval_seconds: float) -> dict[str, Any]:
        interval = float(interval_seconds)
        if not 0.5 <= interval <= 86400:
            raise ValueError("定时采图间隔必须位于 0.5 到 86400 秒")
        with self._condition:
            self._enabled = True
            self._interval_seconds = interval
            self._generation += 1
            self._started_at_ms = timestamp_ms()
            self._next_capture_at_ms = self._started_at_ms + int(interval * 1000)
            self._last_error = None
            self._condition.notify_all()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._condition:
            self._enabled = False
            self._generation += 1
            self._next_capture_at_ms = None
            self._condition.notify_all()
        return self.status()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._enabled = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "schema_version": "1.0",
                "message_type": "timed_capture_status",
                "status": "ok",
                "timestamp_ms": timestamp_ms(),
                "enabled": self._enabled,
                "interval_seconds": self._interval_seconds,
                "started_at_ms": self._started_at_ms,
                "next_capture_at_ms": self._next_capture_at_ms,
                "last_capture_at_ms": self._last_capture_at_ms,
                "capture_count": self._capture_count,
                "failure_count": self._failure_count,
                "last_image": dict(self._last_image) if self._last_image else None,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._enabled:
                    self._condition.wait()
                if self._closed:
                    return
                generation = self._generation
                interval = self._interval_seconds
                deadline = time.monotonic() + interval
                while not self._closed and self._enabled and generation == self._generation:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                if self._closed:
                    return
                if not self._enabled or generation != self._generation:
                    continue

            try:
                payload = save_runtime_snapshot(self._runtime_client, prefix="visionops_auto")
                image = payload.get("image") if isinstance(payload, dict) else None
                with self._condition:
                    self._capture_count += 1
                    self._last_capture_at_ms = timestamp_ms()
                    self._last_image = dict(image) if isinstance(image, dict) else None
                    self._last_error = None
            except Exception as error:  # noqa: BLE001 - status must expose capture failures
                with self._condition:
                    self._failure_count += 1
                    self._last_error = str(error)
            finally:
                with self._condition:
                    if self._enabled and generation == self._generation:
                        self._next_capture_at_ms = timestamp_ms() + int(self._interval_seconds * 1000)
