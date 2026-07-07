"""VisionOps v3 服务端配置。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 18100
    component: str = "visionops_server_api"
    version: str = "v0.1.0-server-mvp"
    data_root: Path = PROJECT_ROOT / "server_data"
    incoming_root: Path | None = None
    publish_root: Path | None = None
    mlflow_uri: str = "http://127.0.0.1:5000"
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    allowed_task_types: tuple[str, ...] = ("detection", "classification", "obb_detection", "segmentation")
    default_target_platform: str = "rk3576"

    @property
    def incoming_packages_root(self) -> Path:
        return self.incoming_root or (self.data_root / "incoming")

    @property
    def batches_root(self) -> Path:
        return self.data_root / "batches"

    @property
    def datasets_root(self) -> Path:
        return self.data_root / "datasets"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def model_packages_root(self) -> Path:
        return self.data_root / "model_packages"

    @property
    def registry_root(self) -> Path:
        return self.data_root / "registry"

    @property
    def devices_path(self) -> Path:
        return self.registry_root / "devices.json"

    def ensure_dirs(self) -> None:
        for path in [
            self.data_root,
            self.incoming_packages_root,
            self.batches_root,
            self.datasets_root,
            self.jobs_root,
            self.model_packages_root,
            self.registry_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        if self.publish_root:
            self.publish_root.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> ServerConfig:
    parser = argparse.ArgumentParser(description="VisionOps v3 Server API")
    parser.add_argument("--host", default=os.getenv("VISIONOPS_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISIONOPS_SERVER_PORT", "18100")))
    parser.add_argument("--data-root", default=os.getenv("VISIONOPS_SERVER_DATA_ROOT", str(PROJECT_ROOT / "server_data")))
    parser.add_argument("--incoming-root", default=os.getenv("VISIONOPS_SERVER_INCOMING_ROOT", ""))
    parser.add_argument("--publish-root", default=os.getenv("VISIONOPS_SERVER_PUBLISH_ROOT", ""))
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--max-upload-mb", type=int, default=int(os.getenv("VISIONOPS_SERVER_MAX_UPLOAD_MB", "2048")))
    parser.add_argument("--target-platform", default=os.getenv("VISIONOPS_SERVER_TARGET_PLATFORM", "rk3576"))
    ns = parser.parse_args(argv)
    incoming_root = Path(ns.incoming_root).expanduser().resolve() if ns.incoming_root else None
    publish_root = Path(ns.publish_root).expanduser().resolve() if ns.publish_root else None
    return ServerConfig(
        host=ns.host,
        port=ns.port,
        data_root=Path(ns.data_root).expanduser().resolve(),
        incoming_root=incoming_root,
        publish_root=publish_root,
        mlflow_uri=ns.mlflow_uri,
        max_upload_bytes=ns.max_upload_mb * 1024 * 1024,
        default_target_platform=ns.target_platform,
    )
