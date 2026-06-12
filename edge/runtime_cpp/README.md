# VisionOps C++ Runtime Mock

本目录实现 M3 阶段的最小 HTTP Mock，用于在没有相机、RKNN、NPU、模型文件和现场通信设备的环境中验证 Runtime 接口契约。

Mock 不包含生产推理能力，也不是 Python RKNN 链路的替代实现。后续真实 Runtime 仍应保持 `Camera Bridge -> C++ RKNN Runtime -> Collector Web -> Gateway/Modbus` 主链路，并复用 M2 定义的标准接口。

## 构建

从仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build -j4
```

生成程序：

```text
build/edge/runtime_cpp/visionops_runtime_mock
```

## 启动

```bash
./build/edge/runtime_cpp/visionops_runtime_mock \
  --host 0.0.0.0 \
  --port 18080 \
  --device-id example-edge-001 \
  --component rknn_runtime \
  --mock-task-type detection
```

支持的 Mock 任务类型：

```text
detection
obb
segmentation
roi_classification
classification
```

查看参数：

```bash
./build/edge/runtime_cpp/visionops_runtime_mock --help
```

## HTTP API

服务实现以下接口：

```text
GET  /health
GET  /api/runtime/status
POST /api/runtime/start_preview
POST /api/runtime/stop_preview
POST /api/runtime/infer_once
GET  /api/runtime/latest_result
GET  /api/runtime/snapshot.jpg
```

完整契约见 `interfaces/protocols/runtime_http_api.md`。当前控制接口读取有界请求体，但不解析业务参数；这是 M3 Mock 的明确限制，后续实现请求 schema 时再增加严格 JSON 解析。

`infer_once` 每次生成新的 `frame_id` 和 `result_id`，并更新状态计数器。快照由编译进程序的 1x1 JPEG 占位数据生成，不读取或提交图片文件。

## 冒烟测试

```bash
bash edge/runtime_cpp/tests/smoke_test.sh
```

脚本会构建程序、选择本机临时端口、启动服务、调用全部接口并停止进程。日志和临时 JPEG 只写入 `/tmp`，退出时自动清理。

## 实现边界

- C++17 与 Linux/POSIX socket。
- 不依赖第三方 HTTP 或 JSON 库。
- 每个连接处理一个请求后关闭，适合契约验证，不用于性能结论。
- 单线程顺序处理请求；状态仍通过互斥锁封装，便于后续演进。
- 请求头限制为 64 KiB，请求体限制为 1 MiB。
- SIGINT 与 SIGTERM 设置停止标记，监听循环在短超时后退出。
