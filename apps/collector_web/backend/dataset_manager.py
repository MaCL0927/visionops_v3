"""VisionOps v3 数据集采集、打包与上传管理。

M16.1 约定：
- 图片保存到 /opt/visionops_v3/data/images。
- 上传包保存到 /opt/visionops_v3/data/upload_packages。
- Web 列表分页读取，避免一次性加载大量图片造成页面卡顿。
- 上传配置读取 vision_box_settings.json 中的 upload 字段。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import CollectorConfig
from .response_utils import timestamp_ms
from .runtime_client import RuntimeClient, RuntimeUnavailable
from .vision_box_settings import DEFAULT_PROJECT_ROOT, load_vision_box_settings

IMAGE_DIR = Path(os.environ.get("VISIONOPS_DATASET_IMAGE_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "images")))
PACKAGE_DIR = Path(os.environ.get("VISIONOPS_DATASET_PACKAGE_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "upload_packages")))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
PACKAGE_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_LIMIT = 24
MAX_LIMIT = 100


def _safe_id(value: str, fallback: str) -> str:
    cleaned = PACKAGE_ID_SAFE.sub("-", str(value or "").strip()).strip(".-_")
    return cleaned or fallback


def _created_at_now() -> tuple[str, str]:
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(), now.strftime("%Y%m%d_%H%M%S")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int(time.time() * 1000) % 1000:03d}"


def _ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name != filename or not SAFE_NAME.match(name):
        raise ValueError("非法文件名")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("不支持的图片扩展名")
    return name


def _image_path(filename: str) -> Path:
    return IMAGE_DIR / _safe_filename(filename)


def _package_path(filename: str) -> Path:
    name = Path(filename).name
    if not name or name != filename or not SAFE_NAME.match(name) or not name.endswith(".tar.gz"):
        raise ValueError("非法压缩包文件名")
    return PACKAGE_DIR / name


def _image_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    name = path.name
    return {
        "id": path.stem,
        "filename": name,
        "url": f"/api/dataset/images/{name}/content",
        "delete_url": f"/api/dataset/images/{name}",
        "mtime_ms": int(stat.st_mtime * 1000),
        "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "size_bytes": stat.st_size,
    }


def list_images(offset: int = 0, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    _ensure_dirs()
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))
    files = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(files)
    selected = files[safe_offset:safe_offset + safe_limit]
    return {
        "schema_version": "1.0",
        "message_type": "dataset_image_list",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "image_dir": str(IMAGE_DIR),
        "offset": safe_offset,
        "limit": safe_limit,
        "total": total,
        "has_more": safe_offset + safe_limit < total,
        "images": [_image_record(path) for path in selected],
    }


def get_image_file(filename: str) -> tuple[Path, str]:
    path = _image_path(filename)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(filename)
    ext = path.suffix.lower()
    content_type = "image/png" if ext == ".png" else "image/jpeg"
    return path, content_type


def delete_image(filename: str) -> dict[str, Any]:
    path = _image_path(filename)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(filename)
    path.unlink()
    return {
        "schema_version": "1.0",
        "message_type": "dataset_image_delete_result",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "deleted": True,
        "filename": path.name,
        "image_dir": str(IMAGE_DIR),
    }


def save_runtime_snapshot(runtime_client: RuntimeClient, prefix: str = "visionops") -> dict[str, Any]:
    _ensure_dirs()
    response = runtime_client.request("GET", f"/api/runtime/snapshot.jpg?t={timestamp_ms()}")
    if response.status_code != 200 or not response.body:
        raise RuntimeUnavailable(f"Runtime snapshot failed: HTTP {response.status_code}")
    ext = ".jpg"
    if response.content_type == "image/png":
        ext = ".png"
    filename = f"{prefix}_{_now_stamp()}{ext}"
    path = IMAGE_DIR / filename
    path.write_bytes(response.body)
    return {
        "schema_version": "1.0",
        "message_type": "dataset_capture_result",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "image": _image_record(path),
        "image_dir": str(IMAGE_DIR),
        "content_type": response.content_type,
    }


def _package_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ms": int(stat.st_mtime * 1000),
        "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def list_packages(limit: int = 20) -> dict[str, Any]:
    _ensure_dirs()
    files = [p for p in PACKAGE_DIR.iterdir() if p.is_file() and p.name.endswith(".tar.gz")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "schema_version": "1.0",
        "message_type": "dataset_package_list",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "package_dir": str(PACKAGE_DIR),
        "packages": [_package_record(p) for p in files[:max(1, min(100, int(limit or 20)))]],
    }


def create_dataset_package(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    images = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError("没有可打包的图片，请先拍照采集")

    metadata = metadata if isinstance(metadata, dict) else {}
    device_id = _safe_id(str(metadata.get("device_id") or "rk3576-001"), "rk3576-001")
    customer_id = _safe_id(str(metadata.get("customer_id") or "CUST-001"), "CUST-001")
    contact_info = str(metadata.get("contact_info") or "").strip()
    remark = str(metadata.get("remark") or "").strip()
    created_at, created_stamp = _created_at_now()
    package_name = f"{device_id}_{customer_id}_{created_stamp}.tar.gz"
    package_path = PACKAGE_DIR / package_name

    suffix = 1
    while package_path.exists():
        package_name = f"{device_id}_{customer_id}_{created_stamp}_{suffix:02d}.tar.gz"
        package_path = PACKAGE_DIR / package_name
        suffix += 1

    manifest = {
        "device_id": device_id,
        "customer_id": customer_id,
        "contact_info": contact_info,
        "remark": remark,
        "created_at": created_at,
        "counts": {"all": len(images)},
        "package_name": package_name,
    }
    with tarfile.open(package_path, "w:gz") as tar:
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        import io
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = time.time()
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for path in images:
            tar.add(path, arcname=f"images/{path.name}")
    return {
        "schema_version": "1.0",
        "message_type": "dataset_package_create_result",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "package": _package_record(package_path),
        "image_count": len(images),
        "image_dir": str(IMAGE_DIR),
        "package_dir": str(PACKAGE_DIR),
        "manifest": manifest,
    }

def _validate_upload_config(config: dict[str, Any]) -> dict[str, Any]:
    upload = config.get("upload") if isinstance(config, dict) else {}
    upload = upload if isinstance(upload, dict) else {}
    server_ip = str(upload.get("server_ip") or "").strip()
    ssh_user = str(upload.get("ssh_user") or "").strip()
    if not server_ip or not ssh_user:
        raise ValueError("请先在视觉盒子设置中配置服务端 IP 和 SSH 用户")
    return {
        "server_ip": server_ip,
        "ssh_user": ssh_user,
        "ssh_password": str(upload.get("ssh_password") or ""),
        "ssh_port": int(upload.get("ssh_port") or 22),
        "remote_dir": str(upload.get("remote_dir") or "/opt/visionops_uploads").strip() or "/opt/visionops_uploads",
        "timeout_s": int(upload.get("timeout_s") or 60),
    }


def _upload_with_paramiko(package_path: Path, upload: dict[str, Any]) -> dict[str, Any] | None:
    if importlib.util.find_spec("paramiko") is None:
        return None
    import paramiko  # type: ignore

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        upload["server_ip"],
        port=upload["ssh_port"],
        username=upload["ssh_user"],
        password=upload["ssh_password"] or None,
        timeout=upload["timeout_s"],
        banner_timeout=upload["timeout_s"],
        auth_timeout=upload["timeout_s"],
    )
    try:
        command = "mkdir -p " + _shell_quote(upload["remote_dir"])
        _stdin, stdout, stderr = client.exec_command(command, timeout=upload["timeout_s"])
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(stderr.read().decode("utf-8", errors="ignore") or f"mkdir failed: {exit_code}")
        sftp = client.open_sftp()
        try:
            remote_path = upload["remote_dir"].rstrip("/") + "/" + package_path.name
            sftp.put(str(package_path), remote_path)
        finally:
            sftp.close()
        return {"method": "paramiko", "remote_path": remote_path}
    finally:
        client.close()


def _shell_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def _run_upload_command(args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)


def _upload_with_system_tools(package_path: Path, upload: dict[str, Any]) -> dict[str, Any]:
    ssh = shutil.which("ssh")
    scp = shutil.which("scp")
    if not ssh or not scp:
        raise RuntimeError("系统缺少 ssh/scp，且未安装 paramiko")
    base_ssh = [ssh, "-p", str(upload["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    base_scp = [scp, "-P", str(upload["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if upload["ssh_password"]:
        sshpass = shutil.which("sshpass")
        if not sshpass:
            raise RuntimeError("使用密码上传需要安装 sshpass，或在 Collector venv 中安装 paramiko")
        base_ssh = [sshpass, "-p", upload["ssh_password"], *base_ssh]
        base_scp = [sshpass, "-p", upload["ssh_password"], *base_scp]
    remote_user_host = f"{upload['ssh_user']}@{upload['server_ip']}"
    mkdir_cmd = [*base_ssh, remote_user_host, "mkdir", "-p", upload["remote_dir"]]
    timeout = max(5, int(upload["timeout_s"]))
    mkdir = _run_upload_command(mkdir_cmd, timeout)
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr.strip() or mkdir.stdout.strip() or f"mkdir failed: {mkdir.returncode}")
    remote_path = upload["remote_dir"].rstrip("/") + "/" + package_path.name
    scp_cmd = [*base_scp, str(package_path), f"{remote_user_host}:{remote_path}"]
    scp_result = _run_upload_command(scp_cmd, timeout)
    if scp_result.returncode != 0:
        raise RuntimeError(scp_result.stderr.strip() or scp_result.stdout.strip() or f"scp failed: {scp_result.returncode}")
    return {"method": "scp", "remote_path": remote_path}


def create_and_upload_dataset(config: CollectorConfig, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    package_result = create_dataset_package(metadata)
    package_path = Path(package_result["package"]["path"])
    started = time.perf_counter()
    upload: dict[str, Any] = {}
    try:
        settings = load_vision_box_settings(config)
        upload = _validate_upload_config(settings)
        upload_result = _upload_with_paramiko(package_path, upload)
        if upload_result is None:
            upload_result = _upload_with_system_tools(package_path, upload)
    except Exception as error:  # noqa: BLE001 - return package path for retry/debug
        return {
            "schema_version": "1.0",
            "message_type": "dataset_upload_result",
            "status": "error",
            "upload_ok": False,
            "timestamp_ms": timestamp_ms(),
            "package": package_result["package"],
            "image_count": package_result["image_count"],
            "manifest": package_result.get("manifest"),
            "upload": {
                "server_ip": upload.get("server_ip"),
                "ssh_user": upload.get("ssh_user"),
                "ssh_port": upload.get("ssh_port"),
                "remote_dir": upload.get("remote_dir"),
                "error": str(error),
            },
            "message": "上传失败，压缩包已保留在本地，可稍后重试或手动拷贝。",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "schema_version": "1.0",
        "message_type": "dataset_upload_result",
        "status": "ok",
        "upload_ok": True,
        "timestamp_ms": timestamp_ms(),
        "package": package_result["package"],
        "image_count": package_result["image_count"],
        "manifest": package_result.get("manifest"),
        "upload": {
            "server_ip": upload.get("server_ip"),
            "ssh_user": upload.get("ssh_user"),
            "ssh_port": upload.get("ssh_port"),
            "remote_dir": upload.get("remote_dir"),
            **upload_result,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
