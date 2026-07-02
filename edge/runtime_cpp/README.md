# VisionOps C++ Runtime

本目录已经不只是早期 M3 的 HTTP Mock。当前 Runtime 已进入 `RK3576 / LB3576` 真机联调阶段，支持：

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

## RGA 预处理加速（可选）

本版本只加入 RGA 预处理入口，不包含 RKNN input/output buffer 深度复用，也不包含 HP60C raw 原始帧入口。

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
--rga-mode resize_rgb      # 当前唯一支持模式
```

`/api/runtime/status` 会展示 `preprocess.backend_requested / backend_active / rga_available`，`infer_once` 的 `debug` 字段会展示 `preprocess_backend_requested / preprocess_backend_active / rga_used`，用于确认是否真正走到 RGA。

注意：HP60C Bridge 当前仍使用 `/stream/snapshot.jpg` JPEG 路径；没有加入 `--hp60c-raw-path` 等 raw 原始帧参数。

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
  --model-manifest "$MODEL_DIR/manifest.json" \
  --model-config "$MODEL_DIR/model.yaml" \
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

- `--model-manifest <path>`
- `--model-config <path>`
- `--model-dir <path>`

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
    ├── manifest.json
    ├── model.yaml
    ├── labels.txt
    └── model.rknn
```

要求：

- 一个目录只表示一个模型包
- `manifest.json` 必须存在
- `manifest.json` 指向的 `rknn / yaml / labels` 必须都存在
- 当前不自动识别同目录中的额外 `model2.rknn`

## M13 耗时字段说明

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

