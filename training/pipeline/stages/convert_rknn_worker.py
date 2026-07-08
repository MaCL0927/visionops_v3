#!/usr/bin/env python3
"""Worker for RKNN conversion.

This module is intentionally small because it is executed inside the rknn311
conda environment by the parent pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_with_rknn(payload: dict[str, Any]) -> dict[str, Any]:
    onnx_path = Path(str(payload.get("onnx_path") or ""))
    rknn_path = Path(str(payload.get("rknn_path") or ""))
    target_platform = str(payload.get("target_platform") or "rk3576")
    do_quantization = bool(payload.get("do_quantization", False))
    dataset_txt = str(payload.get("dataset_txt") or "")
    log_lines: list[str] = []

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

    try:
        from rknn.api import RKNN  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError(
            "当前环境无法导入 rknn.api.RKNN。请确认该 worker 在 rknn311 环境中运行，"
            "并且 rknn-toolkit2 已安装。"
        ) from exc

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform=target_platform,
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            optimization_level=3,
        )
        log_lines.append(f"config ret={ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        log_lines.append(f"load_onnx ret={ret}")
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx 失败，ret={ret}")
        ret = rknn.build(do_quantization=do_quantization, dataset=dataset_txt or None)
        log_lines.append(f"build ret={ret}")
        if ret != 0:
            raise RuntimeError(f"RKNN build 失败，ret={ret}")
        ret = rknn.export_rknn(str(rknn_path))
        log_lines.append(f"export_rknn ret={ret}")
        if ret != 0:
            raise RuntimeError(f"RKNN export_rknn 失败，ret={ret}")
    finally:
        try:
            rknn.release()
        except Exception:
            pass

    return {
        "status": "success",
        "onnx_path": str(onnx_path),
        "rknn_path": str(rknn_path),
        "target_platform": target_platform,
        "do_quantization": do_quantization,
        "dataset_txt": dataset_txt,
        "log": log_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionOps RKNN conversion worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    report = convert_with_rknn(payload)
    report_path = Path(str(payload.get("report_path") or config_path.with_name("convert_rknn_report.json")))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
