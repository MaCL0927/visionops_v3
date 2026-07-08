"""设备注册表与目标模型 SSH 同步。"""

from __future__ import annotations

import os
import subprocess
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
        normalized = self._normalize_device_payload(payload)
        # 登记/更新时顺手做一次轻量 SSH 连接检查，用 collector_status 表示设备连通性。
        collector_status, collector_error = self._check_ssh_status(normalized)
        normalized["collector_status"] = collector_status
        normalized["last_ssh_check_at_ms"] = now
        if collector_error:
            normalized["last_ssh_error"] = collector_error
        else:
            normalized.pop("last_ssh_error", None)

        def mutate(document: dict[str, Any]) -> dict[str, Any]:
            devices = document.setdefault("devices", [])
            for index, item in enumerate(devices):
                if item.get("device_id") == device_id:
                    updated = {**item, **normalized, "device_id": device_id, "updated_at_ms": now}
                    devices[index] = updated
                    return updated
            created = {
                "device_id": device_id,
                "device_name": normalized.get("device_name") or device_id,
                "device_type": normalized.get("device_type") or "lb3576",
                "device_user": normalized.get("device_user") or "neardi",
                "ssh_user": normalized.get("ssh_user") or normalized.get("device_user") or "neardi",
                "ssh_port": int(normalized.get("ssh_port") or os.environ.get("VISIONOPS_DEVICE_SSH_PORT", "22")),
                "ssh_key": normalized.get("ssh_key") or os.environ.get("VISIONOPS_DEVICE_SSH_KEY", ""),
                "ip": normalized.get("ip") or "",
                "model_root": normalized.get("model_root") or "/opt/visionops_v3/models",
                "current_model": normalized.get("current_model") or "",
                "target_model": normalized.get("target_model") or "",
                "sync_status": normalized.get("sync_status") or "unknown",
                "runtime_status": normalized.get("runtime_status") or "unknown",
                "collector_status": normalized.get("collector_status") or "unknown",
                "last_ssh_check_at_ms": normalized.get("last_ssh_check_at_ms") or now,
                "created_at_ms": now,
                "updated_at_ms": now,
            }
            if normalized.get("last_ssh_error"):
                created["last_ssh_error"] = normalized["last_ssh_error"]
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

    def delete_device(self, device_id: str) -> dict[str, Any]:
        device_id = _safe_id(device_id)
        now = int(time.time() * 1000)

        def mutate(document: dict[str, Any]) -> dict[str, Any]:
            devices = document.setdefault("devices", [])
            for index, item in enumerate(list(devices)):
                if item.get("device_id") == device_id:
                    deleted = dict(item)
                    del devices[index]
                    deleted["deleted"] = True
                    deleted["deleted_at_ms"] = now
                    return deleted
            raise FileNotFoundError(f"设备不存在: {device_id}")

        return self.store.update(mutate)

    def sync_model_to_device(self, device_id: str, model_id: str, package_dir: Path) -> dict[str, Any]:
        """通过 ssh/scp 将模型包同步到目标设备。

        注意：这和 ModelPackageService.publish_package() 不同。publish 是复制到
        本机 published_models/Syncthing 共享目录；assign/sync 是直接走 SSH
        推送到设备 registry 中指定的 model_root。
        """
        device_id = _safe_id(device_id)
        model_id = _safe_id(model_id)
        package_dir = Path(package_dir)
        model_rknn = package_dir / "model.rknn"
        model_yaml = package_dir / "model.yaml"
        if not model_rknn.is_file() or not model_yaml.is_file():
            raise FileNotFoundError(f"模型包缺少 model.rknn 或 model.yaml: {package_dir}")

        device = self.get_device(device_id)
        host = str(device.get("ip") or "").strip()
        if not host:
            raise ValueError(f"设备 {device_id} 缺少 ip，无法 SSH 同步")
        user = self._device_user(device)
        port = int(device.get("ssh_port") or os.environ.get("VISIONOPS_DEVICE_SSH_PORT", "22"))
        key = str(device.get("ssh_key") or os.environ.get("VISIONOPS_DEVICE_SSH_KEY", "")).strip()
        model_root = str(device.get("model_root") or "/opt/visionops_v3/models").rstrip("/")
        remote_dir = f"{model_root}/{model_id}"
        remote = f"{user}@{host}" if user else host

        ssh_base = self._ssh_base(port, key)
        scp_base = self._scp_base(port, key)

        commands = [
            ssh_base + [remote, "mkdir", "-p", remote_dir],
            scp_base + [str(model_rknn), str(model_yaml), f"{remote}:{remote_dir}/"],
        ]

        logs: list[str] = []
        now_start = int(time.time() * 1000)
        self._update_device_sync(device_id, model_id, "syncing", current_model=None)
        try:
            for cmd in commands:
                display = " ".join(cmd)
                proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
                logs.append(f"$ {display}")
                if proc.stdout:
                    logs.append(proc.stdout.strip())
                if proc.stderr:
                    logs.append(proc.stderr.strip())
                if proc.returncode != 0:
                    raise RuntimeError(f"SSH 同步命令失败 returncode={proc.returncode}: {display}\n{proc.stderr}")
        except Exception as exc:
            self._update_device_sync(device_id, model_id, "sync_failed", current_model=None, extra={"last_sync_error": str(exc)})
            raise

        updated = self._update_device_sync(
            device_id,
            model_id,
            "synced",
            current_model=model_id,
            extra={
                "last_sync_at_ms": int(time.time() * 1000),
                "last_sync_remote_dir": remote_dir,
                "last_sync_method": "ssh_scp",
                "last_sync_log": "\n".join(logs[-20:]),
                "collector_status": "connect",
            },
        )
        return {
            "device": updated,
            "model_id": model_id,
            "remote_dir": remote_dir,
            "files": ["model.rknn", "model.yaml"],
            "started_at_ms": now_start,
            "finished_at_ms": int(time.time() * 1000),
            "log": "\n".join(logs),
        }

    def _normalize_device_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        device_user = str(
            payload.get("device_user")
            or payload.get("ssh_user")
            or payload.get("user")
            or os.environ.get("VISIONOPS_DEVICE_SSH_USER", "neardi")
        ).strip() or "neardi"
        normalized = dict(payload)
        normalized["device_user"] = device_user
        normalized["ssh_user"] = device_user
        normalized["device_type"] = payload.get("device_type") or "lb3576"
        normalized["ssh_port"] = int(payload.get("ssh_port") or os.environ.get("VISIONOPS_DEVICE_SSH_PORT", "22"))
        normalized["ssh_key"] = payload.get("ssh_key") or os.environ.get("VISIONOPS_DEVICE_SSH_KEY", "")
        normalized["model_root"] = payload.get("model_root") or "/opt/visionops_v3/models"
        normalized["ip"] = str(payload.get("ip") or "").strip()
        return normalized

    def _check_ssh_status(self, device: dict[str, Any]) -> tuple[str, str]:
        host = str(device.get("ip") or "").strip()
        if not host:
            return "unknown", ""
        user = self._device_user(device)
        port = int(device.get("ssh_port") or os.environ.get("VISIONOPS_DEVICE_SSH_PORT", "22"))
        key = str(device.get("ssh_key") or os.environ.get("VISIONOPS_DEVICE_SSH_KEY", "")).strip()
        remote = f"{user}@{host}" if user else host
        cmd = self._ssh_base(port, key, timeout=5) + [remote, "true"]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=8)
        except Exception as exc:
            return "fail", str(exc)
        if proc.returncode == 0:
            return "connect", ""
        error = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()
        return "fail", error

    def _device_user(self, device: dict[str, Any]) -> str:
        return str(
            device.get("device_user")
            or device.get("ssh_user")
            or device.get("user")
            or os.environ.get("VISIONOPS_DEVICE_SSH_USER", "neardi")
        ).strip()

    def _ssh_base(self, port: int, key: str = "", *, timeout: int = 10) -> list[str]:
        base = [
            "ssh",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        if key:
            base.extend(["-i", key])
        return base

    def _scp_base(self, port: int, key: str = "") -> list[str]:
        base = [
            "scp",
            "-P",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        if key:
            base.extend(["-i", key])
        return base

    def _update_device_sync(
        self,
        device_id: str,
        model_id: str,
        sync_status: str,
        *,
        current_model: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = int(time.time() * 1000)

        def mutate(document: dict[str, Any]) -> dict[str, Any]:
            devices = document.setdefault("devices", [])
            for item in devices:
                if item.get("device_id") == device_id:
                    item["target_model"] = model_id
                    item["sync_status"] = sync_status
                    if current_model is not None:
                        item["current_model"] = current_model
                    item["updated_at_ms"] = now
                    if extra:
                        item.update(extra)
                    return item
            raise FileNotFoundError(f"设备不存在: {device_id}")

        return self.store.update(mutate)


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法 ID")
    return safe
