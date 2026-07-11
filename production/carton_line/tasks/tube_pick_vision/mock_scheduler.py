#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal scheduler-side TCP server for on-device protocol verification."""
from __future__ import annotations

import argparse
import json
import socket
import time

from production.carton_line.tasks.tube_pick_vision.tcp_client import StarHashJsonCodec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock VisionInterfacer TCP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--camera", default="cam_1")
    parser.add_argument("--task-id", default="tube_pick_vision")
    parser.add_argument("--function", default="vision0")
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Waiting for visual client on {args.host}:{args.port} ...")
        connection, address = server.accept()
        with connection:
            print(f"Connected: {address}")
            connection.settimeout(args.timeout)
            now_ns = time.time_ns()
            timestamp = [now_ns // 1_000_000_000, now_ns % 1_000_000_000]
            request = {
                "function": args.function,
                "timestamp": timestamp,
                "triggerpos": timestamp[0],
                "triggerindex": 1,
                "camera": args.camera,
                "waist_angle": 0.0,
                "task_id": args.task_id,
                "left_arm": {"x": 0, "y": 0, "z": 0, "ox": 0, "oy": 0, "oz": 0, "ow": 1},
                "right_arm": {"x": 0, "y": 0, "z": 0, "ox": 0, "oy": 0, "oz": 0, "ow": 1},
            }
            connection.sendall(StarHashJsonCodec.encode(request))
            codec = StarHashJsonCodec()
            while True:
                data = connection.recv(65536)
                if not data:
                    raise ConnectionError("visual client disconnected before response")
                messages = codec.feed(data)
                if messages:
                    print(json.dumps(messages[0], ensure_ascii=False, indent=2))
                    return 0


if __name__ == "__main__":
    raise SystemExit(main())
