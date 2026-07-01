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
- 模型包读取 `manifest.json / model.yaml / labels.txt`。
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

## 4.1 当前模型包目录规范

当前按“一目录一个模型包”管理：

```text
/opt/visionops_v3/models/
├── carton_tube_check/
│   ├── manifest.json
│   ├── model.yaml
│   ├── labels.txt
│   └── model.rknn
└── test_rknn_model/
    ├── manifest.json
    ├── model.yaml
    ├── labels.txt
    └── model.rknn
```

Collector 只扫描 `models_root` 下的一级子目录，且当前仅把以下目录识别为标准模型包：

- 存在 `manifest.json`
- `manifest.json` 指向的 `rknn / yaml / labels` 都存在
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
  --model-manifest "$MODEL_DIR/manifest.json" \
  --model-config "$MODEL_DIR/model.yaml" \
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
- Collector Web 真实采集保存和打包上传。
- `carton_tube_check` 接真实检测结果。
- `carton_partition_check` 接真实 OBB 结果。
- 双 Runtime / 双 Collector / 双 Business App 并行验证。

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
