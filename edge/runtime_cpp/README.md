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

仓库中的 `examples/mock_model_package/` 只有解析测试所需的 manifest、YAML 和 labels，不包含真实模型。真实模型文件、完整模型包及 `.rknn/.pt/.onnx` 制品不得提交到 Git。

## M9.2 RKNN Runner 外壳

M9.2 引入统一 `RknnRunner` 接口和可选 `RknnRunnerReal` 构建路径。默认仍使用 `mock` backend，不依赖 RKNN SDK：

```bash
cmake -S . -B build
cmake --build build -j4
./build/edge/runtime_cpp/visionops_runtime_mock --backend mock
```

在 RK3576/RK3588 的部署环境中，可显式提供 RKNN Runtime SDK：

```bash
cmake -S . -B build-rknn \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/opt/rknn/include \
  -DVISIONOPS_RKNN_LIBRARY=/opt/rknn/lib/librknnrt.so
cmake --build build-rknn -j4

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --model-dir /opt/visionops/models/example \
  --model-manifest manifest.json
```

当 `VISIONOPS_ENABLE_RKNN=ON` 时，CMake 会验证 `rknn_api.h` 和 RKNN Runtime 库路径，并编译 `rknn_runner_real.cpp`。

默认构建中使用 `--backend rknn` 不会导致进程退出。Runtime 会保持 HTTP 可诊断，状态显示 `degraded`、`runner_loaded=false`、`rknn_compiled=false`，`infer_once` 返回标准 JSON 错误。

## M9.3 真实推理与后处理

M9.3 将 RKNN Runner、CPU letterbox 和 YOLO 后处理接入 `RuntimeApp`。启用 RKNN 的构建会执行模型加载、输入设置、`rknn_run`、float 输出获取和资源释放，再按模型 `task_type` 路由：

- detection：支持常见单输出 YOLOv8 `[1,C,N]` / `[1,N,C]`，以及 v2 使用的 Rockchip split-DFL `[1,64,H,W] + [1,nc,H,W]`。
- OBB：支持单输出 `xywh + class scores + angle`，输出四点、角度和外接框；当前使用外接矩形 NMS。
- segmentation：支持单输出候选 tensor 加 proto tensor，默认无 OpenCV 路径输出受 bbox 约束的简化 polygon，不返回原始 tensor。

所有后处理应用 `score_threshold`、`nms_threshold`、labels 和 letterbox 坐标回映。暂不支持的多头 OBB/seg shape 会返回 `UNSUPPORTED_OUTPUT_SHAPE`，不会崩溃或伪造检测结果。`roi_classification` 的真实 RKNN 后处理仍明确返回 `UNSUPPORTED_TASK_TYPE`。

新增运行参数：

- `--test-image <path>`：默认支持 P6 PPM；开启 OpenCV 后支持 JPEG/PNG。
- `--dump-rknn-io`：打印输入输出 tensor 属性。
- `--score-threshold` / `--nms-threshold`：覆盖模型配置阈值。
- `--save-debug-output <dir>`：仅保存 result id、shape 数量和 warning 等轻量摘要，不保存原始 tensor 或图片。

没有 `--test-image` 时，M9.3 使用内存 Mock RGB frame 测试真实 Runner；真实相机取流留到 M10。

### RKNN 后端手动测试

在 RK3576/RK3588 部署环境构建：

```bash
cmake -S . -B build-rknn \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/path/to/rknn/include \
  -DVISIONOPS_RKNN_LIBRARY=/path/to/librknnrt.so \
  -DVISIONOPS_ENABLE_OPENCV=ON

cmake --build build-rknn -j4
```

手动运行：

```bash
./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --model-manifest /path/to/model_package/manifest.json \
  --model-config /path/to/model_package/model.yaml \
  --model-dir /path/to/model_package \
  --test-image /path/to/test.jpg \
  --dump-rknn-io \
  --host 0.0.0.0 \
  --port 18080
```

若部署环境不提供 OpenCV，请关闭 `VISIONOPS_ENABLE_OPENCV` 并使用 P6 PPM 测试图。真实测试图片、模型包和 `.rknn/.pt/.onnx` 文件不得进入 Git。

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

`infer_once` 每次生成新的 `frame_id` 和 `result_id`，并更新状态计数器。`snapshot.jpg` 会优先将 Runtime 最新 RGB888 帧编码为 JPEG；如果尚无最新帧或编码失败，则回退到编译进程序的 1x1 JPEG 占位数据，不读取或提交图片文件。

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
| `RknnRunner` | 统一模型加载、backend 状态、推理和错误接口 | M9.2 已接入 Mock/Real/Unavailable 三种实现 |
| `RknnRunnerReal` | 可选 RKNN Context、输入输出查询、执行和资源释放 | 只返回结构化原始 tensor，不包含 YOLO decode |
| `StreamWorkerMock` | 生成 Mock Frame、维护预览开关 | M10 在 `stream_worker` 边界迁入真实相机取流 |
| `Postprocess` | 独立完成 detect/OBB/seg tensor 解析、NMS 和坐标回映 | 输出继续遵守 M2 契约 |
| `SnapshotProvider` | 将最新 RGB888 帧编码为 JPEG，失败时回退内置极小 JPEG | 后续可替换为硬件 JPEG、libjpeg 或 OpenCV 编码 |

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

## M10：真实相机取流接入一期

M10 在 `stream_worker` 边界加入统一帧源抽象，Runtime 现在支持：

- `--frame-source mock`：默认模式，继续生成灰色 Mock Frame；
- `--frame-source test_image`：使用 `--test-image` 指定的本地测试图片，默认无 OpenCV 时仅支持 P6 PPM；
- `--frame-source v4l2`：在 Linux 上通过 V4L2 读取 `/dev/videoX`，M10 一期优先支持 YUYV 输入并转换为 RGB888。

新增常用参数：

```text
--frame-source mock/test_image/v4l2
--camera-device /dev/video0
--camera-width 640
--camera-height 480
--camera-fps 30
--camera-pixel-format YUYV
--enable-camera-thread true/false
--camera-read-timeout-ms 1000
```

`GET /api/runtime/status` 会返回 `frame_source` 字段，包含帧源类型、设备、打开状态、尺寸、帧率、像素格式、最近帧时间戳和最近错误。Collector Web 仍然只访问 Runtime HTTP API，不直接访问相机。

M10.1 已支持真实 `snapshot.jpg` 输出：当 Runtime 已经通过 `start_preview` 或 `infer_once` 获得最新 RGB888 帧时，`SnapshotProvider` 会用内置轻量 JPEG 编码器返回该帧，`frame_source.snapshot_encoder=rgb888_jpeg`。如果尚无最新帧或编码失败，则稳定回退到内置 1x1 Mock JPEG，并显示 `snapshot_encoder=mock_jpeg`。当前实现不强制依赖 libjpeg/OpenCV。

### 3576 V4L2 手动测试

查看设备能力：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

构建 RKNN 版本：

```bash
cmake -S . -B build-rknn \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so
cmake --build build-rknn -j4
```

启动真实相机帧源与 RKNN 后端：

```bash
MODEL_DIR=/opt/visionops_v3/models/test_rknn_model

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --frame-source v4l2 \
  --camera-device /dev/video0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-pixel-format YUYV \
  --model-manifest "$MODEL_DIR/manifest.json" \
  --model-config "$MODEL_DIR/model.yaml" \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 18081 \
  --device-id lb3576-dev
```

测试接口：

```bash
curl http://127.0.0.1:18081/api/runtime/status | python3 -m json.tool
curl -X POST http://127.0.0.1:18081/api/runtime/start_preview | python3 -m json.tool
curl -X POST http://127.0.0.1:18081/api/runtime/infer_once | python3 -m json.tool
curl http://127.0.0.1:18081/api/runtime/latest_result | python3 -m json.tool
curl -I http://127.0.0.1:18081/api/runtime/snapshot.jpg
```

M10 的核心验收标准是：`frame_source.type=v4l2`、`camera_connected=true`、`infer_once` 的 `image.width/height` 来自真实相机帧，并且 `backend=rknn` 仍然能完成推理闭环。是否检测到目标取决于现场画面、模型和阈值，不作为 M10 一期唯一通过条件。

## M10.2：HP60C SDK HTTP Bridge 帧源

在 LB3576 上如果已经安装并启动 `visionops-hp60c-sdk-bridge.service`，Runtime 可以不直接链接 Angstrong SDK，而是通过本机 HTTP Bridge 读取 HP60C 图像。

Bridge 默认接口来自 v2 做法：

- `GET http://127.0.0.1:18181/health`
- `GET http://127.0.0.1:18181/stream/snapshot.jpg`

Runtime 新增帧源：

```bash
--frame-source hp60c_bridge
--hp60c-url http://127.0.0.1:18181
--hp60c-snapshot-path /stream/snapshot.jpg
--hp60c-health-path /health
```

说明：

- `snapshot.jpg` 预览会优先直接返回 HP60C Bridge 提供的 JPEG，因此 Web 端可以看到真实 HP60C 画面。
- 若需要把 HP60C JPEG 解码为 RGB888 并进入 RKNN 推理，需要在构建 Runtime 时启用 OpenCV：`-DVISIONOPS_ENABLE_OPENCV=ON`。
- 默认构建不强制依赖 OpenCV；未启用 OpenCV 时，HP60C 快照仍可用于 Web 预览，但 `infer_once` 会返回需要启用 OpenCV 的稳定 JSON 错误。

3576 上检查 Bridge：

```bash
sudo systemctl status visionops-hp60c-sdk-bridge.service --no-pager -l
curl -s http://127.0.0.1:18181/health | python3 -m json.tool
curl -o /tmp/hp60c_bridge.jpg http://127.0.0.1:18181/stream/snapshot.jpg
file /tmp/hp60c_bridge.jpg
```

3576 上构建 Runtime：

```bash
cmake -S . -B build-rknn \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DVISIONOPS_ENABLE_OPENCV=ON

cmake --build build-rknn -j4
```

启动 Runtime 使用 HP60C Bridge：

```bash
MODEL_DIR=/opt/visionops_v3/models/test_rknn_model

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18181 \
  --hp60c-snapshot-path /stream/snapshot.jpg \
  --hp60c-health-path /health \
  --model-manifest "$MODEL_DIR/manifest.json" \
  --model-config "$MODEL_DIR/model.yaml" \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 18081 \
  --device-id lb3576-dev
```

测试接口：

```bash
curl -X POST http://127.0.0.1:18081/api/runtime/start_preview | python3 -m json.tool
curl -I http://127.0.0.1:18081/api/runtime/snapshot.jpg
curl http://127.0.0.1:18081/api/runtime/snapshot.jpg -o /tmp/v3_hp60c_snapshot.jpg
file /tmp/v3_hp60c_snapshot.jpg
curl -X POST http://127.0.0.1:18081/api/runtime/infer_once | python3 -m json.tool
```

验收时重点查看：

- `frame_source.type = hp60c_bridge`
- `frame_source.opened = true`
- `frame_source.snapshot_encoder = hp60c_bridge_jpeg`
- `model.backend = rknn`
- `debug.rknn_runner_called = true`


### M10.2 HP60C Bridge 状态修复

M10.2 修复了 HP60C Bridge 帧源的运行状态同步问题：`start_preview` 会立即抓取一帧用于刷新 `latest_frame_id` 与 `snapshot_encoder`；`snapshot.jpg` 支持 GET 与 HEAD；`status.source` 会随 backend 显示为 `runtime:mock` 或 `runtime:rknn`；模型 YAML 的 `input_size` 同时兼容 `[640, 640]` 与 `640` 写法。
