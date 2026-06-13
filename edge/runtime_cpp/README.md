# VisionOps C++ Runtime Mock

本目录实现 M3 阶段的 HTTP Mock，并在 M8 完成第一期结构拆分，用于在没有相机、RKNN、NPU、模型文件和现场通信设备的环境中验证 Runtime 接口契约与模块边界。

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

## M9.1 模型包配置

M9.1 只读取模型包 manifest、YAML 配置和标签文本，不加载 `.rknn`，不调用 RKNN SDK，也不执行真实推理。示例启动方式：

```bash
./build/edge/runtime_cpp/visionops_runtime_mock \
  --model-dir edge/runtime_cpp/examples/mock_model_package \
  --model-manifest manifest.json \
  --model-config model.yaml
```

支持的参数：

- `--model-manifest <path>`：轻量 JSON manifest。
- `--model-config <path>`：简单 key-value 与行内 list YAML。
- `--model-dir <path>`：相对文件路径基准；未显式指定 manifest 时可发现目录中的 `manifest.json`。

读取优先级为内置 Mock 默认值、manifest、YAML。YAML 可覆盖模型名称、版本、任务类型、输入尺寸和阈值；标签数量优先读取 manifest 指向的标签文本，否则使用 YAML `class_names` 数量。

`loaded_model` 和 `inference_result.model` 会包含模型标识、任务类型、占位 `.rknn` 路径、配置路径、标签数量、输入尺寸、score/NMS 阈值。显式配置文件不存在或无法解析时，服务仍会启动，`health` 变为 `degraded`，详细原因写入 `model_load_error`。

仓库中的 `examples/mock_model_package/` 只有解析测试所需的 manifest、YAML 和 labels，不包含真实模型。真实模型文件、完整模型包及 `.rknn/.pt/.onnx` 制品不得提交到 Git。真实 RKNN Runner 将在 M9.2 接入。

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

## M8 模块边界

M8 是结构重构，不是接入真实 RKNN、RGA 或相机。M3 的接口路径、错误语义和 Mock 结果保持兼容。

| 模块 | 当前职责 | 后续演进 |
| --- | --- | --- |
| `main.cpp` | 解析 CLI、注册信号、组装并启动服务 | 保持薄入口，不承载业务 JSON 或运行状态 |
| `AppConfig / CliArgs` | 默认值、参数解析与合法性检查 | 后续可接入统一配置渲染结果 |
| `RuntimeApp` | 编排状态、取帧、预处理、推理、后处理和快照 | 保持 HTTP 之外的 Runtime 对外能力入口 |
| `RuntimeState` | 线程安全维护模式、计数器、序号和最近结果 | 为多线程取流与推理队列保留互斥边界 |
| `HttpServer` | POSIX socket、请求解析、路由和 JSON/JPEG 响应 | 不生成推理结果，不维护业务状态 |
| `JsonUtils` | 时间戳、JSON 转义和统一错误响应 | 继续保持无第三方 JSON 依赖 |
| `RknnRunnerMock` | 生成 Mock 推理输出 | M9.2 在 `rknn_runner` 边界迁入真实 RKNN 能力 |
| `StreamWorkerMock` | 生成 Mock Frame、维护预览开关 | M10 在 `stream_worker` 边界迁入真实相机取流 |
| `Postprocess` | 按 detection、OBB、segmentation 等任务生成标准结果片段 | 后续接真实张量解析，但输出继续遵守 M2 契约 |
| `SnapshotProvider` | 返回内置极小 JPEG | 后续从受控帧缓存产生低频快照 |

`RuntimeState` 当前仍运行在单线程 HTTP 请求模型下，但所有状态读写均通过互斥锁保护。后续加入取流线程和推理线程时，不应绕过该边界直接修改计数器。

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
