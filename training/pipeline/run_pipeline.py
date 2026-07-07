#!/usr/bin/env python3
"""VisionOps v3 训练流水线入口占位。

当前服务端 MVP 通过 apps.server_api.backend.services.training_job_service 中的
mock runner 验证任务编排和 v3 模型包契约。真实训练接入时，本文件应承接
preprocess -> train -> evaluate -> export_onnx -> convert_rknn -> package_v3_model。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionOps v3 training pipeline placeholder")
    parser.add_argument("--job-config", required=True, help="训练任务 JSON 配置路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()
    job_config = json.loads(Path(args.job_config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_report.json").write_text(
        json.dumps(
            {
                "status": "placeholder",
                "message": "真实训练流水线尚未接入；当前仅保留 v3 stage 边界。",
                "job_config": job_config,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
