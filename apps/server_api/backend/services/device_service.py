"""设备注册表。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..storage.json_store import JsonStore


class DeviceService:
    def __init__(self, devices_path: Path) -> None:
        self.store = JsonStore(Path(devices_path), default={"schema_version": "1.0", "devices": []})

    def list_devices(self) -> list[dict[str, Any]]:
        document = self.store.read()
        return list(document.get("devices", [])) if isinstance(document, dict) else []

    def get_device(self, device_id: str) -> dict[str, Any]:
        device_id = _safe_id(device_id)
        for item in self.list_devices():
            if item.get("device_id") == device_id:
                return item
        raise FileNotFoundError(f"设备不存在: {device_id}")

    def upsert_device(self, payload: dict[str, Any]) -> dict[str, Any]:
        device_id = _safe_id(str(payload.get("device_id") or ""))
        now = int(time.time() * 1000)
        def mutate(document: dict[str, Any]) -> dict[str, Any]:
            devices = document.setdefault("devices", [])
            for index, item in enumerate(devices):
                if item.get("device_id") == device_id:
                    updated = {**item, **payload, "device_id": device_id, "updated_at_ms": now}
                    devices[index] = updated
                    return updated
            created = {
                "device_id": device_id,
                "device_name": payload.get("device_name") or device_id,
                "device_type": payload.get("device_type") or "lb3576",
                "ip": payload.get("ip") or "",
                "model_root": payload.get("model_root") or "",
                "current_model": payload.get("current_model") or "",
                "target_model": payload.get("target_model") or "",
                "sync_status": payload.get("sync_status") or "unknown",
                "runtime_status": payload.get("runtime_status") or "unknown",
                "collector_status": payload.get("collector_status") or "unknown",
                "created_at_ms": now,
                "updated_at_ms": now,
            }
            devices.append(created)
            return created
        return self.store.update(mutate)

    def assign_model(self, device_id: str, model_id: str) -> dict[str, Any]:
        device_id = _safe_id(device_id)
        model_id = _safe_id(model_id)
        now = int(time.time() * 1000)
        def mutate(document: dict[str, Any]) -> dict[str, Any]:
            devices = document.setdefault("devices", [])
            for item in devices:
                if item.get("device_id") == device_id:
                    item["target_model"] = model_id
                    item["sync_status"] = "assigned"
                    item["updated_at_ms"] = now
                    return item
            raise FileNotFoundError(f"设备不存在: {device_id}")
        return self.store.update(mutate)


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法 ID")
    return safe
