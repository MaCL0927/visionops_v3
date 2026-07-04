# M13 RGA-only 修改说明

本包基于用户还原后的 VisionOps v3 代码，只添加 RGA 预处理入口，未加入以下两个实验功能：

- RKNN input / output buffer 更深层复用
- HP60C Bridge 可选 raw 原始帧入口

## 新增能力

- 新增 CMake 选项：`VISIONOPS_ENABLE_RGA`
- 新增 CLI 参数：
  - `--preprocess-backend cpu|rga|auto`
  - `--rga-mode resize_rgb`
- `cpu` 为默认行为，保持原始 CPU letterbox 预处理。
- `rga` 使用 RGA 做 RGB888 resize，然后将 resized 图像贴入 letterbox 画布。
- `auto` 在 RGA 可用时优先使用 RGA，失败时回退 CPU。
- `/api/runtime/status` 增加 `preprocess` 状态字段。
- `infer_once` 的 `debug` 字段增加实际预处理后端和 `rga_used` 标识。

## 3576 RGA 构建示例

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

## 启动示例

```bash
./build-rknn-rga-release/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --preprocess-backend rga \
  --rga-mode resize_rgb \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18182 \
  --hp60c-snapshot-path /stream/snapshot.jpg \
  --hp60c-health-path /health \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 28081 \
  --device-id lb3576-dev
```

## 验证命令

```bash
curl -s http://127.0.0.1:28081/api/runtime/status | python3 -m json.tool
curl -s -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
python3 tools/benchmark_runtime.py --runtime-url http://127.0.0.1:28081 --warmup 10 --count 50 --output /tmp/v3_rga_jpeg_benchmark.json
```

重点确认：

- `status.preprocess.rga_available=true`
- `infer_once.debug.preprocess_backend_active="rga"`
- `infer_once.debug.rga_used=true`
- `timing.preprocess_ms` 低于 CPU 模式

### LB3576 librga 兼容注意

本 RGA-only 包已经处理 LB3576 当前 Rockchip librga 头文件的两个兼容问题：

- `wrapbuffer_virtualaddr` 显式传入 `wstride/hstride`，避免 4 参数宏触发 `zero-size array`。
- `imcheck` 显式传入 `src_rect/dst_rect` 和 `mode_usage=0`，避免 `imcheck(src, dst, {}, {})` 的空 `__VA_ARGS__` 触发 `zero-size array`。
- 启用 RGA 时链接 `${CMAKE_DL_LIBS}`，避免 `dlclose@@GLIBC_2.17` 的链接错误。

