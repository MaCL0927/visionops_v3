# VisionOps C++ Runtime

> 总体项目入口请优先阅读仓库根目录 `README.md`。本文档聚焦 Runtime 的构建、启动、模型包与 HTTP API。

本目录已从早期 HTTP Mock 演进为可在 `RK3576 / LB3576` 上运行的 Runtime，支持：

- `mock` backend：用于 x86 开发、接口契约验证和无硬件环境调试。
- `rknn` backend：用于 RK3576 / RK3588 上的真实模型加载、RKNN 推理与标准 `inference_result` 输出。
- `v4l2` 与 `hp60c_bridge` 帧源。
- `snapshot.jpg` 真实帧输出。
- 运行期 `switch_model` 模型切换接口。

Runtime 在系统中的职责边界保持为：

```text
Camera Bridge / HP60C Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
```

Runtime 负责生产推理、预处理、后处理、状态输出和快照；Collector Web 只做代理、展示和管理，不直接连接相机，也不解析 RKNN 原始 tensor。

## 构建

x86 / 默认 mock 构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

生成程序：

```text
build/edge/runtime_cpp/visionops_runtime_mock
```

RKNN 真机构建示例：

```bash
cmake -S . -B build-rknn \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/path/to/rknn/include \
  -DVISIONOPS_RKNN_LIBRARY=/path/to/librknnrt.so

cmake --build build-rknn -j4
```

说明：

- 若未显式指定 `CMAKE_BUILD_TYPE`，当前仓库默认会走 `Release`。
- 默认构建不依赖 RKNN SDK。
- `VISIONOPS_ENABLE_RKNN=ON` 时才编译真实 RKNN Runner。
- `VISIONOPS_ENABLE_OPENCV=ON` 后，HP60C JPEG 可解码为 `RGB888` 进入 RKNN 推理。

## RGA 预处理与性能优化

当前 Runtime 已接入：

- 可选 `RGA` 预处理入口
- 更深一层的 RKNN input / output buffer 复用
- HP60C Bridge 可选原始帧入口

RGA 真机构建示例：

```bash
cmake -S . -B build-rknn-rga-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_ENABLE_RGA=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DVISIONOPS_RGA_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RGA_LIBRARY=/usr/lib/librga.so

cmake --build build-rknn-rga-release -j4
```

启动时通过参数选择预处理后端：

```bash
--preprocess-backend cpu   # 默认 CPU letterbox
--preprocess-backend rga   # 强制使用 RGA resize + CPU letterbox paste
--preprocess-backend auto  # RGA 可用时优先使用，失败时回退 CPU
--rga-mode resize_rgb      # 当前推荐模式
```

`/api/runtime/status` 会展示 `frame_source`、`loaded_model` 和当前运行状态；`infer_once` 的耗时字段可用于确认预处理与推理路径是否变化。

如果 HP60C Bridge 提供原始帧入口，当前还支持：

```bash
--hp60c-raw-path /stream/frame.rgb
--hp60c-raw-width 1280
--hp60c-raw-height 720
--hp60c-raw-pixel-format RGB888
```

当前支持：

- `application/octet-stream` 的 `RGB888 / BGR888`
- `image/bmp`

## 常用启动方式

### 1. 默认 Mock

```bash
./build/edge/runtime_cpp/visionops_runtime_mock \
  --host 0.0.0.0 \
  --port 18080 \
  --device-id example-edge-001 \
  --component rknn_runtime \
  --mock-task-type detection
```

支持的 `mock-task-type`：

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

### 2. HP60C / 336lsdk 18182 Bridge

```bash
MODEL_DIR=/opt/visionops_v3/models/test_rknn_model

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18182 \
  --hp60c-snapshot-path /stream/snapshot.jpg \
  --hp60c-health-path /health \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 28081 \
  --device-id lb3576-dev
```

说明：

- `snapshot.jpg` 可以直接返回 HP60C Bridge 的 JPEG。
- 开启 OpenCV 构建后，`infer_once` 可以把 JPEG 解码为 `RGB888` 再送入 RKNN。
- 这条链路只负责推理和标准结果输出，不承载业务判断。

## 模型包与配置

支持读取：

- `--model-dir <path>`：标准模型目录，目录内必须包含 `model.rknn` 和 `model.yaml`。

`loaded_model` 与 `inference_result.model` 会优先使用模型包中的：

- `model_id`
- `model_name`
- `model_version`
- `task_type`
- `backend`
- `input_size`
- `score_threshold`
- `nms_threshold`
- `labels_count`

仓库中的 `examples/mock_model_package/` 只用于解析测试，不包含真实 `.rknn`。

### 标准模型包目录

当前切换模型时，Runtime 期望接收到一个标准模型包目录：

```text
models/
└── carton_tube_check/
    ├── model.rknn
    └── model.yaml
```

要求：

- 一个目录只表示一个模型包
- 必须包含 `model.rknn` 和 `model.yaml`
- `model.yaml` 是模型元信息唯一来源
- 不再读取 `manifest.json` / `labels.txt`
- 当前不自动识别同目录中的额外 `model2.rknn`


## OBB 1280 / 动态输入尺寸兼容

OBB RKNN split-DFL 后处理现在不再写死 640 输入或固定 8400 candidates。
对 Rockchip YOLOv8-OBB 多输出：

```text
[1, 64 + nc, H, W] 或 [1, 64 + nc + 1, H, W]
[1, 1, sum(H*W)]
```

都会按输出 shape 动态识别。例如 1280 输入常见为：

```text
[1,67,160,160]
[1,67,80,80]
[1,67,40,40]
[1,1,33600]
```

其中 `33600 = 160*160 + 80*80 + 40*40`。如果 head 比 `64 + labels_count` 多 1 个辅助通道，后处理会使用配置中的类别通道并忽略额外辅助通道，避免误判为不支持。

## 耗时字段说明

`POST /api/runtime/infer_once` 当前会返回两组耗时信息。

兼容字段 `timing`：

- `capture_ms`
- `decode_ms`
- `preprocess_ms`
- `inference_ms`
- `postprocess_ms`
- `result_build_ms`
- `total_ms`

细分字段 `timing_detail`：

- `capture_ms`
- `decode_ms`
- `preprocess_ms`
- `rknn_set_input_ms`
- `rknn_run_ms`
- `rknn_get_output_ms`
- `postprocess_ms`
- `result_build_ms`
- `total_ms`

说明：

- `inference_ms = rknn_set_input_ms + rknn_run_ms + rknn_get_output_ms`
- `snapshot.jpg` 编码、Web overlay 绘制、HTTP 响应发送不计入 `preprocess / inference / postprocess`
- HP60C Bridge 的 HTTP 拉图与 JPEG 解码会单独进入 `capture_ms / decode_ms`
- `result_build_ms` 只统计 Runtime 构建标准 JSON 结果的时间

## HTTP API

```text
GET  /health
GET  /api/runtime/status
POST /api/runtime/start_preview
POST /api/runtime/stop_preview
POST /api/runtime/infer_once
POST /api/runtime/switch_model
GET  /api/runtime/latest_result
GET  /api/runtime/snapshot.jpg
```

完整协议见 `interfaces/protocols/runtime_http_api.md`。

### Preview 与 snapshot 行为

- Web 侧实时预览依赖 Collector 在页面初始化后调用 `POST /api/runtime/start_preview`。
- Runtime 进入 preview 状态后，后台取帧线程会持续刷新最新帧。
- `GET /api/runtime/snapshot.jpg` 会优先返回最新真实帧。
- 如果当前没有可用真实帧，Runtime 会稳定回退到内置占位 JPEG，而不是读取仓库图片文件。

对 `hp60c_bridge` 帧源来说：

- Runtime 可以直接缓存并转发 HP60C 的 JPEG。
- 当 OpenCV 可用时，同一帧也可解码用于 `infer_once`。

### 模型切换行为

`POST /api/runtime/switch_model` 请求体示例：

```json
{
  "model_dir": "/opt/visionops_v3/models/carton_tube_check"
}
```

切换规则：

1. Runtime 先解析新模型包。
2. 创建新的 Runner。
3. 加载新模型。
4. 只有新模型加载成功后，才替换旧 `rknn_runner_` 和 `loaded_model`。
5. 如果新模型加载失败，旧模型会被保留，不会进入无模型状态。

## 如何验证 snapshot 是否实时更新

在 3576 真机上可以比较两次快照摘要：

```bash
curl -s http://127.0.0.1:18182/stream/snapshot.jpg | sha256sum
sleep 1
curl -s http://127.0.0.1:18182/stream/snapshot.jpg | sha256sum

curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg | sha256sum
sleep 1
curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg | sha256sum
```

如果 preview 已启动且现场画面有变化，两次摘要通常不应长期完全一致。还建议同时检查：

```bash
curl -s http://127.0.0.1:28081/api/runtime/status
```

重点关注：

- `frame_source.frames_captured`
- `frame_source.latest_timestamp_ms`
- `frame_source.last_error`

连续对比推理耗时：

```bash
for i in $(seq 1 20); do
  curl -s -X POST http://127.0.0.1:28081/api/runtime/infer_once > /tmp/v3_infer_$i.json
done

python3 tools/benchmark_runtime.py \
  --runtime-url http://127.0.0.1:28081 \
  --count 50 \
  --warmup 5 \
  --output /tmp/v3_runtime_benchmark.json
```

### 3576 上验证模型切换

```bash
curl -s http://127.0.0.1:18091/api/models | python3 -m json.tool

curl -X POST http://127.0.0.1:18091/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"package_dir":"carton_tube_check"}' | python3 -m json.tool

curl -s http://127.0.0.1:28081/api/runtime/status | python3 -m json.tool

curl -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
```

真实 RKNN 模型热切换仍需要在 3576 真机验证；x86 环境当前只验证 mock backend 的接口逻辑和失败语义。

## 模块边界

- `main.cpp`：薄入口，负责 CLI、信号和服务启动。
- `HttpServer`：只负责 HTTP 路由、JSON/JPEG 响应，不拼业务结果。
- `RuntimeApp`：编排取帧、推理、后处理、快照与状态。
- `StreamWorker`：负责 `mock / test_image / v4l2 / hp60c_bridge` 帧源。
- `RknnRunner`：负责模型加载、RKNN 调用与底层错误。
- `Postprocess`：负责 detection / OBB / segmentation 标准化输出。

不要把业务判断、Gateway 映射或 Web 逻辑写进 Runtime。

## 冒烟测试

```bash
bash edge/runtime_cpp/tests/smoke_test.sh
```

该脚本适合 x86 / mock 环境接口验证。`hp60c_bridge`、OpenCV 解码、真实 RKNN、3576 设备驱动仍需要在真机上手动验证。

### LB3576 librga 兼容注意

本 RGA-only 包已经处理 LB3576 当前 Rockchip librga 头文件的两个兼容问题：

- `wrapbuffer_virtualaddr` 显式传入 `wstride/hstride`，避免 4 参数宏触发 `zero-size array`。
- `imcheck` 显式传入 `src_rect/dst_rect` 和 `mode_usage=0`，避免 `imcheck(src, dst, {}, {})` 的空 `__VA_ARGS__` 触发 `zero-size array`。
- 启用 RGA 时链接 `${CMAKE_DL_LIBS}`，避免 `dlclose@@GLIBC_2.17` 的链接错误。


### Segmentation split-DFL 输出兼容

Runtime segmentation 后处理支持 Rockchip YOLOv8-seg split-DFL 多输出格式。该格式通常包含 3 个尺度，每个尺度分别输出 bbox DFL、class、objectness、mask coefficients，并额外输出 proto。后处理根据 head 的 H/W 与模型输入尺寸动态推导 stride，不写死 640 或 8400。

当前 segmentation mask 使用 bbox polygon 简化表示，满足 Web 可视化与基础结果查看；真正基于 proto 的实例 mask 栅格化后续再实现。

## 统一输出 ROI

Runtime 支持通过 `--roi-config <path>` 为每个实例指定 ROI 配置文件，并提供：

```text
GET  /api/runtime/roi
POST /api/runtime/roi
```

配置示例：

```json
{"enabled":true,"x1":0.1,"y1":0.2,"x2":0.9,"y2":0.8}
```

ROI 坐标归一化到原始图像宽高。模型仍对整幅图像完成预处理和推理；detection、OBB、segmentation 在 NMS 后按目标中心点过滤。该过滤发生在标准 `inference_result` 生成前，因此所有 Runtime 调用方自动获得一致结果。分类任务没有空间目标，当前不应用 ROI 过滤。

## 第一阶段实时吞吐优化

Runtime 新增模型包/命令行后处理上限：

```text
--max-detections N
--mask-max-points N
```

模型包也可在 `model.yaml` 中声明：

```yaml
max_detections: 1
mask_max_points: 64
```

`loaded_model` 会返回实际生效值。纸箱抓取任务默认只保留 1 个候选并将 mask
polygon 限制为 64 点，避免 Runtime 生成大量最终不会被业务层使用的实例 mask。

`/api/runtime/status` 中的 `fps.inference_fps` 和 `latency_ms` 已改为真实完成记录，
不再使用固定占位值。FPS 使用最近 2 秒完成窗口；停止推理后会回落到 0。

HTTP Bridge 帧源的后台取图循环按 `camera_fps` 使用绝对截止时间节流。快照下载和
JPEG 解码耗时计入周期，不会在处理完成后再额外休眠完整周期，也不会对 Bridge
缓存快照进行无上限空转读取。
