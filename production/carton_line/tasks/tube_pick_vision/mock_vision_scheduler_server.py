#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time


def now_ros():
    ns = time.time_ns()
    return ns // 1_000_000_000, ns % 1_000_000_000


def make_trigger(index, camera, task_id):
    sec, nsec = now_ros()
    return {
        "function": "vision0",
        "timestamp": [sec, nsec],
        "triggerpos": sec,
        "triggerindex": index,
        "camera": camera,
        "waist_angle": 0.0,
        "task_id": task_id,
        "left_arm": {"x": 0.0, "y": 0.0, "z": 0.0, "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0},
        "right_arm": {"x": 0.0, "y": 0.0, "z": 0.0, "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0},
    }


def encode_frame(obj):
    return ("*" + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "#").encode("utf-8")


def recv_frame(conn, buffer):
    while True:
        start = buffer.find(b"*")
        if start < 0:
            buffer = b""
        elif start > 0:
            buffer = buffer[start:]

        if buffer.startswith(b"*"):
            end = buffer.find(b"#", 1)
            if end >= 0:
                raw = buffer[:end + 1]
                remain = buffer[end + 1:]
                obj = json.loads(raw[1:-1].decode("utf-8"))
                return obj, remain, raw

        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError("视觉盒已断开连接")
        buffer += chunk


def main():
    parser = argparse.ArgumentParser(description="模拟上位机 TCP Server，触发 3576 视觉检测")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--camera", default="cam_1")
    parser.add_argument("--task-id", default="tube_pick_vision")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)

    print(f"[LISTEN] {args.host}:{args.port}")
    print("[WAIT] 等待 3576 作为 TCP Client 连接...")
    conn, addr = server.accept()
    conn.settimeout(30.0)
    print(f"[CONNECTED] {addr[0]}:{addr[1]}")

    index = 1
    buffer = b""

    try:
        while True:
            cmd = input("\n按 Enter 触发一次；输入 q 退出：").strip().lower()
            if cmd == "q":
                break

            req = make_trigger(index, args.camera, args.task_id)
            frame = encode_frame(req)
            print("\n[SEND RAW]")
            print(frame.decode("utf-8"))
            conn.sendall(frame)

            resp, buffer, raw = recv_frame(conn, buffer)
            print("\n[RECV RAW]")
            print(raw.decode("utf-8"))
            print("\n[RECV JSON]")
            print(json.dumps(resp, ensure_ascii=False, indent=2))

            if resp.get("triggerindex") != index:
                print(f"[WARN] triggerindex 不一致: 请求={index}, 响应={resp.get('triggerindex')}")
            index += 1
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
