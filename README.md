# VisionOps v3

VisionOps v3 是面向工业视觉场景重建的端到端视觉 AI 平台，当前已经进入 `RK3576 / LB3576` 真机联调阶段，不再是早期“只有架构骨架”的仓库。

当前生产主链路默认保持为：

```text
Camera Bridge / HP60C Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
  -> PLC 或上位机
```

其中：

- `C++ Runtime` 负责生产推理、预处理、RKNN 调用和标准 `inference_result` 输出。
- `Collector Web` 只负责配置、展示、状态聚合和代理，不直接连接相机、不加载模型、不解析 RKNN 原始 tensor。
- `Business App` 负责纸筒、隔板等业务规则和业务决策。
- `Gateway / Modbus` 负责把标准结果映射为现场通信寄存器或协议消息。
- `interfaces/schemas` 与 `interfaces/protocols` 是跨模块契约来源。

## 当前已具备的能力

- 统一配置骨架与配置校验工具。
- 标准接口 schema、example 与协议文档。
- C++ Runtime HTTP 服务与模块化拆分。
- 模型包读取：M15 后固定为 `model.rknn + model.yaml`，`model.yaml` 是唯一元信息来源。
- `RknnRunnerMock`、`RknnRunnerReal`、`RknnRunnerUnavailable`。
- detection / OBB / segmentation 基础后处理。
- `v4l2` 与 `hp60c_bridge` 帧源。
- Runtime `snapshot.jpg` 输出真实帧。
- HP60C / 336lsdk `18182` HTTP Bridge 接入。
- Collector Web 代理 Runtime / Gateway / Business App，并提供边缘端三件套页面。
- Collector Web 模型包扫描与点击切换。
- Gateway / Modbus Mock。
- `carton_tube_check` 与 `carton_partition_check` 业务 App Mock。
- YOLOv8-OBB RKNN 多输出适配。

## 3576 典型启动方式

Runtime：

```bash
cd /opt/visionops_v3

MODEL_DIR=/opt/visionops_v3/models/test_rknn_model

cmake -S . -B build-rknn \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so

cmake --build build-rknn -j4

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

Collector Web：

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

`--models-root` 未传入时，Collector 会优先使用仓库根目录下的 `models/`，若不存在则回退为 `/opt/visionops_v3/models`。

## 模型包目录规范

当前阶段按“一目录一个标准模型包”管理：

```text
/opt/visionops_v3/models/
├── carton_tube_check/
│   ├── model.rknn
│   └── model.yaml
└── test_rknn_model/
    ├── model.rknn
    └── model.yaml
```

Collector Web 只扫描 `models_root` 下的一级子目录。只有满足以下条件的目录才会被视为标准模型包：

- 目录内存在 `model.rknn` 和 `model.yaml`
- `model.yaml` 是唯一元信息来源
- 不再读取 `manifest.json` / `labels.txt`

当前不会把同目录下额外的 `model2.rknn` 自动识别为第二个模型。

## 文档入口

- [当前状态交接文档](docs/handoff/current_state_2026_06_30.md)
- [边缘端运行时架构](docs/architecture/edge_runtime.md)
- [配置设计](docs/architecture/config_design.md)
- [从 v2 迁移](docs/migration/from_v2.md)
- [协作约束](AGENTS.md)

## 当前下一步重点

- `systemd` 服务化与设备启动编排。
- 模型包部署规范与 `current` 软链接约定。
- v2 / v3 推理耗时口径对齐与后续 RGA 优化评估。
- 3576 真机验证模型热切换。
- Collector Web 真实采集保存与采集包上传。
- `carton_tube_check` 接真实检测结果。
- `carton_partition_check` 接真实 OBB 结果。
- 双 Runtime / 双 Collector / 双 Business App 并行验证。

## M13 性能对比入口

当前 Runtime 已补充更细的耗时字段。

兼容字段仍保留在 `timing`：

- `preprocess_ms`
- `inference_ms`
- `postprocess_ms`
- `total_ms`

新增字段包括：

- `timing.capture_ms`
- `timing.decode_ms`
- `timing.result_build_ms`
- `timing_detail.rknn_set_input_ms`
- `timing_detail.rknn_run_ms`
- `timing_detail.rknn_get_output_ms`

Collector Web 设置页当前支持：

- `preview_refresh_interval_ms`
- `inference_interval_ms`

这两个设置保存在浏览器 `localStorage`，不会写入 `.env` 或源 YAML。

3576 上可使用：

```bash
python3 tools/benchmark_runtime.py \
  --runtime-url http://127.0.0.1:28081 \
  --count 50 \
  --warmup 5 \
  --output /tmp/v3_runtime_benchmark.json
```

## 边缘端代码同步

本地修改代码后，如果不想走 “先 push GitHub，再在 3576 上 git pull” 的流程，当前仓库提供了一个直接同步到板端固定目录的脚本：

```bash
bash edge/deploy/push.sh --host <3576-ip> --user <ssh-user>
```

默认同步到：

```text
/opt/visionops_v3
```

当前默认同步的是边缘端运行所需代码与配置：

- `apps/collector_web`
- `edge/`
- `interfaces/`
- `configs/`
- `deploy/`
- `tools/`
- `README.md`
- `AGENTS.md`
- `CMakeLists.txt`
- `.gitignore`

当前默认不同步：

- `build/`
- `training/`
- `tests/`
- `apps/server_api/`
- `models/`
- `.git/`
- `__pycache__/`
- `*.pyc`
- `*.pt / *.onnx / *.rknn`

模型同步后续会单独扩展，当前脚本只先解决代码与配置同步。

## 仓库约束

不要提交真实模型、图片、视频、日志、`.env`、密钥、压缩包或现场私密信息，包括但不限于 `.pt`、`.onnx`、`.rknn`、采集数据和部署制品。

> M13 RGA-only 说明：当前代码只加入可选 RGA 预处理加速，使用 `-DVISIONOPS_ENABLE_RGA=ON` 构建，并通过 `--preprocess-backend rga` 启用；没有加入 RKNN output buffer 预分配复用和 HP60C raw 原始帧入口。

