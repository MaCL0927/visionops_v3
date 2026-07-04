# VisionOps v3 当前状态交接（2026-06-30）

## 1. 当前项目状态

VisionOps v3 已经进入 `RK3576 / LB3576` 真机联调阶段，不再是早期仅有架构与文档骨架的仓库。

当前系统需要按真实边缘端主链路理解和维护：

```text
Camera Bridge / HP60C Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
  -> PLC 或上位机
```

## 2. 关键边界

- `C++ Runtime` 负责生产推理、取帧、预处理、RKNN 调用、后处理和标准 `inference_result` 输出。
- `Collector Web` 只负责管理、展示和代理，不直接连相机、不加载模型、不解析 RKNN 原始 tensor。
- `Business App` 负责业务判断、业务决策和业务寄存器。
- `Gateway / Modbus` 负责通信适配。
- `interfaces/schemas` 与 `interfaces/protocols` 是模块间契约来源。

明确禁止：

- 不要恢复 v2 Python RKNN 生产主链路。
- 不要让 Web 直接访问相机。
- 不要把业务判断写进 C++ Runtime。
- 不要让 Gateway / Business App 解析 RKNN 原始 tensor。
- 不要覆盖 M10.2 的 HP60C Bridge 帧源适配。
- 不要覆盖 M11.1 的 OBB 多输出适配。

## 3. 已完成能力

- 配置骨架。
- 标准接口 schema 和 examples。
- C++ Runtime HTTP 服务。
- Collector Web 代理 Runtime / Gateway / Business App。
- Gateway / Modbus Mock。
- `carton_tube_check` 和 `carton_partition_check` 业务 App。
- Collector Web 前端基本页面。
- Collector Web 前端已调整为参考 v2 的现场大屏 / 触屏友好风格，但接口和架构保持 v3。
- Runtime 模块化拆分。
- 模型包读取已在 M15 简化为 `model.rknn + model.yaml`，`model.yaml` 是唯一元信息来源。
- `RknnRunnerReal / RknnRunnerMock / RknnRunnerUnavailable`。
- detection / OBB / segmentation 后处理基础。
- `v4l2` 和 `hp60c_bridge` 帧源。
- Runtime `snapshot.jpg` 输出真实帧。
- HP60C / 336lsdk `18182` HTTP Bridge 接入。
- 双任务拓扑设计。
- YOLOv8-OBB RKNN 多输出适配。

## 4. 最近修复

- Web 打开后自动调用 `POST /api/runtime/start_preview`。
- 避免 `snapshot / next_frame` 在未启动 preview 时一直返回首帧缓存。
- Collector Web “模型验证”页支持扫描 `models_root` 并点击切换模型。
- Runtime 新增 `POST /api/runtime/switch_model`，新模型加载失败时保留旧模型。
- M13：Runtime 新增更细的耗时统计，Collector 设置页支持 `preview_refresh_interval_ms / inference_interval_ms`。
- M13 RGA-only：在还原后的 Runtime 基础上只加入 RGA 预处理入口；未加入 RKNN input/output buffer 深度复用，未加入 HP60C raw 原始帧入口。
- M13 OBB dynamic input：OBB split-DFL 后处理已兼容 1280×1280 等动态输入尺寸，支持 `[1,64+nc(+1),H,W] + [1,1,sum(H*W)]` 输出。

## 4.1 当前模型包目录规范

当前按“一目录一个模型包”管理：

```text
/opt/visionops_v3/models/
├── carton_tube_check/
│   ├── model.rknn
│   └── model.yaml
└── test_rknn_model/
    ├── model.rknn
    └── model.yaml
```

Collector 只扫描 `models_root` 下的一级子目录，且当前仅把以下目录识别为标准模型包：

- 存在 `model.rknn` 和 `model.yaml`
- `model.yaml` 是唯一元信息来源
- 不再读取 `manifest.json` / `labels.txt`
- 文件路径解析后仍位于模型包目录内部

当前不会把同目录中的额外 `model2.rknn` 自动识别为第二个模型。

## 5. 当前 3576 手动启动命令

### Runtime

```bash
cd /opt/visionops_v3

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

### Collector Web

```bash
source /opt/visionops/venv/bin/activate

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 \
  --port 18091 \
  --runtime-url http://127.0.0.1:28081 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
  --models-root /opt/visionops_v3/models \
  --device-id lb3576-dev
```

## 6. 当前建议验证命令

18182 snapshot 两次摘要：

```bash
curl -s http://127.0.0.1:18182/stream/snapshot.jpg | sha256sum
sleep 1
curl -s http://127.0.0.1:18182/stream/snapshot.jpg | sha256sum
```

Runtime 28081 snapshot 两次摘要：

```bash
curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg | sha256sum
sleep 1
curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg | sha256sum
```

Collector 18091 snapshot 两次摘要：

```bash
curl -s http://127.0.0.1:18091/api/runtime/snapshot.jpg | sha256sum
sleep 1
curl -s http://127.0.0.1:18091/api/runtime/snapshot.jpg | sha256sum
```

检查 Runtime 状态：

```bash
curl -s http://127.0.0.1:28081/api/runtime/status
```

重点关注：

- `frame_source.frames_captured`
- `frame_source.latest_timestamp_ms`
- `frame_source.last_error`

扫描模型目录：

```bash
curl -s http://127.0.0.1:18091/api/models | python3 -m json.tool
```

切换模型：

```bash
curl -X POST http://127.0.0.1:18091/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"package_dir":"carton_tube_check"}' | python3 -m json.tool
```

切换后检查 Runtime 当前模型与推理输出：

```bash
curl -s http://127.0.0.1:28081/api/runtime/status | python3 -m json.tool

curl -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
```

## 7. 后续优先事项

- `systemd` 服务化。
- 模型包部署规范与 `current` 软链接。
- 3576 真机验证 RKNN 模型切换稳定性。
- 3576 真机对比 v2 / v3 `capture / decode / preprocess / rknn_run / postprocess` 口径与真实性能。
- 评估是否在 M13.2 引入 RGA / 更深的 RKNN output buffer 复用。
- Collector Web 真实采集保存和打包上传。
- `carton_tube_check` 接真实检测结果。
- `carton_partition_check` 接真实 OBB 结果。
- 双 Runtime / 双 Collector / 双 Business App 并行验证。

## 7.1 当前代码同步方式

为避免“本地改代码 -> push GitHub -> 3576 git pull”的重复链路，当前仓库已提供：

```bash
bash edge/deploy/push.sh --host <3576-ip> --user <ssh-user>
```

用途：

- 通过 `ssh + rsync` 直接把 v3 边缘端所需代码同步到 `3576:/opt/visionops_v3`
- 先解决代码与配置同步问题
- 模型目录 `models/` 的同步参数先预留，后续再独立扩展

当前脚本默认同步：

- `apps/collector_web`
- `edge/`
- `interfaces/`
- `configs/`
- `deploy/`
- `tools/`
- 根目录下的 `README.md / AGENTS.md / CMakeLists.txt / .gitignore`

当前默认不通过该脚本同步：

- `build/`
- `training/`
- `tests/`
- `apps/server_api/`
- `models/`
- `.git/`
- `__pycache__/`
- `.pt / .onnx / .rknn`

## 8. 给后续 Codex / 开发者的注意事项

- 这个仓库已经不是“只剩 Mock”的阶段，修改 Runtime 时要尊重真实主链路。
- `main.cpp` 继续保持薄入口。
- `HttpServer` 只负责 HTTP。
- `RuntimeApp` 负责编排。
- `StreamWorker` 负责取帧。
- `RknnRunner` 负责 RKNN。
- `Postprocess` 负责后处理。
- Web 前端只能访问 Collector 同源接口。
- 不要整目录复制 `visionops_v2`，只迁必要函数和能力边界。

## 9. M13 当前验证入口

Release 构建建议：

```bash
cmake -S . -B build-rknn-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so

cmake --build build-rknn-release -j4
```

基准脚本：

```bash
python3 tools/benchmark_runtime.py \
  --runtime-url http://127.0.0.1:28081 \
  --count 50 \
  --warmup 5 \
  --output /tmp/v3_runtime_benchmark.json
```

说明：

- `timing` 保留兼容字段，同时增加 `capture_ms / decode_ms / result_build_ms`
- `timing_detail` 增加 `rknn_set_input_ms / rknn_run_ms / rknn_get_output_ms`
- Collector Web 当前已改为参考 v2 的现场大屏风格，但接口和架构保持 v3

## M13 segmentation split-DFL 兼容更新

- Runtime segmentation 后处理新增 Rockchip YOLOv8-seg split-DFL 多输出格式支持。
- 支持形如 13 输出的 RKNN seg 模型：每个尺度包含 `[1,64,H,W]` bbox DFL、`[1,nc,H,W]` class、`[1,1,H,W]` objectness、`[1,mask_dim,H,W]` mask coefficients，最后一个 proto 为 `[1,mask_dim,proto_h,proto_w]`。
- stride 由 `letterbox.input_width / W` 与 `letterbox.input_height / H` 动态推导，不写死 640 或 8400；因此 640、1280 或其他输入尺寸只要保持 YOLOv8-seg split-DFL 输出结构即可兼容。
- 当前 mask 输出仍采用 bbox polygon 简化表示，用于 Web 可视化；proto 已识别，真正 mask 栅格化留作后续优化。

## M14 设置界面优化

- 设置弹窗重构为三页：相机设置、视觉盒子设置、算法设置。
- 当前设置保存到浏览器 localStorage，不写入 `.env`、`model.yaml` 或 systemd。
- 算法设置新增模型验证页可视化开关：Detection bbox、OBB 旋转框、OBB 外接水平框、Seg bbox、Seg mask polygon、标签、中心点、mask 透明度。
- OBB 外接水平框默认关闭，避免 Web 端同时显示旋转框和水平外接框。
- Segmentation 当前仍显示 Runtime 返回的 polygon；若 Runtime 输出为 bbox polygon，则 Web 端显示的仍是矩形简化 mask，真实 mask 栅格化属于后续 Runtime 后处理工作。

## M14 补充：SDK Bridge 相机设置页优化

- 设置中心弹窗上下占比进一步加长，接近全屏设置面板。
- 相机设置页统一使用 SDK Bridge 命名，适配 HP60C 与 Orbbec Gemini 336L 两类 SDK + HTTP Bridge 取流方式。
- 固定 Bridge URL 与 snapshot path 不再作为用户编辑项展示。
- 预览刷新间隔与快照刷新间隔合并为“画面帧率 FPS”，保存时同步换算为两个旧 interval 字段。
- RGB 分辨率 / FPS、Depth 分辨率 / FPS 改为 profile 下拉框，并增加 RGB / Depth 匹配提示。
- 新增 JPEG 质量、RGB 数据优先级、垂直/水平翻转、RGB 顺序、深度单位、Orbbec 序列号等 SDK Bridge 相关配置入口。
- 当前仍为前端本地设置，不写入 SDK Bridge env，不重启 systemd 服务。

## M14 Orbbec 336L SDK Bridge 设置 API

- Collector Web 新增 `GET/POST /api/settings/sdk_bridge/orbbec336l`。
- 相机设置页的 RGB / Depth profile 下拉框改为后端动态加载，不在前端写死。
- 后端优先从 Orbbec Bridge `GET /stream/profiles` 读取 SDK 实际支持组合。
- 保存设置会写入 `/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env` 并重启 `visionops-orbbec336l-bridge.service`。
- 新增 `edge/camera_bridge/orbbec336l_bridge/`，提供带 `/stream/profiles` 的 Orbbec Bridge 源码、CMakeLists、env 和安装脚本。
- 当前真实设置应用优先支持 Orbbec Gemini 336L；HP60C 后续再接入。

## M14 Orbbec 设置 API 路径与耗时修正

- Orbbec 336L 设置 API 默认 env 路径已改为 `/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env`。
- Orbbec Bridge install 脚本默认部署到 `/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge`，二进制安装到 `/opt/visionops_v3/bin`。
- 保存相机设置时不再生成 `orbbec336l_bridge.env.bak.*`。
- POST 保存设置时不再重复访问 `/stream/profiles`；前端会把 GET 时已枚举的 `known_profiles` 提交给后端做校验，减少等待时间。
- 保存 API 返回 `apply_timings_ms`，用于定位 read env、profile 校验、写 env、restart、health 检查等步骤的耗时。
