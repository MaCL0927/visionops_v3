# VisionOps v3 边缘端视觉 AI 平台

## 1. 项目定位

VisionOps v3 是面向 `RK3576 / LB3576 / RK3588` 边缘视觉盒子的端到端视觉 AI 平台，负责相机接入、模型推理、Web 管理、业务判断，以及和 PLC / 上位机之间的通信适配。

当前仓库已经进入边缘端真机联调阶段，不再是早期仅包含架构骨架与接口占位的仓库。

## 2. 当前主链路

```text
Camera Bridge / SDK Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
  -> PLC / 上位机
```

模块职责与边界：

- `Camera Bridge / SDK Bridge`
  - 负责相机或厂商 SDK 接入，向 Runtime 提供快照、视频流或桥接接口。
  - 当前仓库内已包含 `Orbbec Gemini 336L` Bridge；`HP60C / 336lsdk` 通过 HTTP Bridge 接入。
- `C++ RKNN Runtime`
  - 负责取帧、预处理、RKNN 推理、后处理、标准 `inference_result` 输出。
  - 生产推理主链路固定在 C++，不回退到 Python RKNN。
- `Collector Web`
  - 只负责配置、展示、状态聚合和代理 Runtime / Gateway / Business App。
  - 不直接访问相机，不加载模型，不解析 RKNN 原始 tensor。
- `Business App`
  - 负责纸筒、隔板等业务规则和业务判断。
- `Gateway / Modbus`
  - 负责通信适配、寄存器映射、PLC / 上位机协议交互。

## 3. 仓库目录结构

- `apps/collector_web`
  - 边缘端 Web 后端与前端，提供校验、采集上传、模型验证、设置、生产模式五个页面。
- `edge/runtime_cpp`
  - C++ Runtime，包含取帧、预处理、RKNN Runner、后处理、HTTP API、快照与模型切换。
- `edge/camera_bridge`
  - 相机 / SDK Bridge。当前仓库内主要是 `orbbec336l_bridge`。
- `edge/gateway_adapter`
  - 标准 `inference_result -> gateway_message -> Holding Registers` 转换，以及业务 App 层。
- `edge/modbus_adapter`
  - 最小 Modbus TCP mock、寄存器 Bank 和测试客户端。
- `interfaces`
  - 模块间契约来源，包括 `schemas / examples / protocols`。
- `configs`
  - 示例配置，按 `edge / task / app / runtime` 分层。
- `deploy`
  - 部署占位目录、systemd 预留目录和边缘端运行脚本示例。
- `tools`
  - 配置校验、接口摘要、网关寄存器摘要、Runtime benchmark 等开发工具。
- `tests`
  - 单元测试、集成测试、smoke test。
- `docs`
  - 架构、迁移、legacy notes、handoff 与归档文档。

## 4. 已完成功能

### 4.1 推理运行时

- C++ Runtime RKNN 推理主链路。
- 可选 `RGA` 预处理入口。
- Detection / OBB / Segmentation 后处理。
- Runtime HTTP API：
  - `health`
  - `status`
  - `start_preview`
  - `stop_preview`
  - `infer_once`
  - `latest_result`
  - `snapshot.jpg`
  - `switch_model`
- Runtime `snapshot.jpg` 输出真实帧。
- `mock / test_image / v4l2 / hp60c_bridge` 帧源。
- `RknnRunnerMock / RknnRunnerReal / RknnRunnerUnavailable`。
- OBB 多输出与动态输入尺寸兼容。

### 4.2 相机与 Bridge

- `HP60C / 336lsdk` HTTP Bridge 帧源接入。
- `Orbbec Gemini 336L` SDK Bridge 源码、env 与安装脚本。
- Orbbec Bridge `profiles / status` 接口。

### 4.3 Collector Web

- 五个页面：
  - 校验
  - 采集上传
  - 模型验证
  - 设置
  - 生产模式
- 模型验证页支持实时检测可视化。
- 模型包扫描与切换。
- 采集上传页当前支持：
  - 快照采集
  - 本地列表查看
  - 打包导出 / 上传入口
- 设置中心当前已接入：
  - 相机设置
  - 算法设置
  - 视觉盒子设置
- 双网口 `eth0 / eth1` 读取、展示和配置应用。
- 生产模式实时检测大屏。
- 从生产模式返回工厂模式需要固定管理员验证：`admin / admin`。

### 4.4 业务闭环

- Gateway / Modbus Mock 闭环。
- `carton_tube_check` 业务 App Mock。
- `carton_partition_check` 业务 App Mock。
- Business App 当前可以消费标准 `inference_result`，输出：
  - `AppDecision`
  - `GatewayMessage`
  - 业务寄存器

说明：

- 当前 `Gateway / Business App / Modbus` 在仓库内主要是 mock / 业务闭环验证实现。
- 接真实 PLC、真实产线信号和真实业务规则后的长期稳定性，仍需继续真机联调。

## 5. 标准模型包规范

当前模型包固定为：

```text
models/<model_name>/
├── model.rknn
└── model.yaml
```

约束：

- Collector 只扫描 `models_root` 下一级目录。
- `model.yaml` 是唯一元信息来源。
- 不再使用 `manifest.json / labels.txt` 作为主路径。
- 当前不会自动识别同目录下额外的 `model2.rknn`。

## 6. 编译 C++ Runtime

```bash
cd /opt/visionops_v3

cmake -S . -B build-rknn \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_ENABLE_RGA=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DVISIONOPS_RGA_INCLUDE_DIR=/usr/include \
  -DVISIONOPS_RGA_LIBRARY=/usr/lib/librga.so

cmake --build build-rknn -j4
```

说明：

- 当前 target 名称仍为 `visionops_runtime_mock`。
- 但当 `--backend rknn` 时，会走真实 RKNN 路径。
- 后续可以考虑把 target 重命名为更贴近生产含义的名称，但当前接口与脚本仍以现名为准。

## 7. 启动 Runtime

```bash
MODEL_DIR=/opt/visionops_v3/models/test_rknn_model

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
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
  --device-id lb3576-001
```

## 8. 启动 Collector Web

```bash
source /opt/visionops/venv/bin/activate

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 \
  --port 18091 \
  --runtime-url http://127.0.0.1:28081 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
  --device-id lb3576-dev
```

说明：

- 浏览器访问：`http://127.0.0.1:18091/`
- `runtime-url`：Collector 代理的 Runtime 地址。
- `gateway-url`：Collector 读取 Gateway 状态与寄存器的地址。
- `business-app-url`：Collector 读取业务 App 状态、寄存器和业务结果的地址。

## 9. 常用验证命令

### Runtime

```bash
curl -s http://127.0.0.1:28081/health | python3 -m json.tool
curl -s http://127.0.0.1:28081/api/runtime/status | python3 -m json.tool
curl -s -X POST http://127.0.0.1:28081/api/runtime/start_preview | python3 -m json.tool
curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg -o /tmp/runtime_snapshot.jpg
curl -s -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
curl -s http://127.0.0.1:28081/api/runtime/latest_result | python3 -m json.tool
```

### Collector Web

```bash
curl -s http://127.0.0.1:18091/health | python3 -m json.tool
curl -s http://127.0.0.1:18091/api/collector/status | python3 -m json.tool
curl -s http://127.0.0.1:18091/api/models | python3 -m json.tool
curl -s -X POST http://127.0.0.1:18091/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"package_dir":"test_rknn_model"}' | python3 -m json.tool
```

### Orbbec Bridge

```bash
curl -s http://127.0.0.1:18182/stream/profiles | python3 -m json.tool
curl -s http://127.0.0.1:18182/stream/status | python3 -m json.tool
```

### 生产模式与网络

```bash
curl -s http://127.0.0.1:18091/api/gateway/status | python3 -m json.tool
curl -s http://127.0.0.1:18091/api/app/status | python3 -m json.tool
curl -s http://127.0.0.1:18091/api/settings/vision_box | python3 -m json.tool
ip -j addr show eth0
ip -j addr show eth1
```

## 10. Web 操作流程

1. 启动 Camera Bridge / SDK Bridge。
2. 启动 Runtime。
3. 启动 Collector Web。
4. 打开 Web 页面。
5. 在设置页配置相机、算法、视觉盒子。
6. 在模型验证页选择模型并测试实时检测。
7. 进入生产模式。
8. 查看生产模式实时检测画面。
9. 通过状态区查看 Gateway / Business App / Modbus 状态。
10. 返回工厂模式时输入 `admin / admin`。

## 11. 配置文件与持久化

- `configs/`
  - 仅存放示例配置和分层设计说明。
  - 当前不能视为现场运行时的唯一真实配置源。
- `/opt/visionops_v3/config/vision_box_settings.json`
  - 视觉盒子设置当前默认持久化文件。
- `edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env`
  - Orbbec Bridge 运行参数来源。
- 浏览器 `localStorage`
  - 保存前端刷新间隔、部分 UI 状态与本地页面设置。
- 会写入后端文件的配置
  - Orbbec Bridge 设置
  - 视觉盒子设置
  - 算法阈值会写回当前模型的 `model.yaml`
- 会立即生效的配置
  - 双网口 `eth0 / eth1` 配置保存后会立即调用 `ip` 命令应用

## 12. 测试与开发

- 无硬件测试：
  - `mock backend`
  - `mock frame source`
  - `mock model package`
  - 大部分 `unit / integration / smoke test`
- 依赖 RKNN / 相机 / 板端环境的验证：
  - 真实 RKNN 推理
  - RGA 路径
  - HP60C / Orbbec 真机取流
  - 真实业务 App / Gateway / PLC 联调

常用测试命令：

```bash
python -m pytest tests/unit
python -m pytest tests/integration
cmake -S . -B build
cmake --build build -j4
bash edge/runtime_cpp/tests/smoke_test.sh
bash apps/collector_web/tests/smoke_test.sh
bash edge/gateway_adapter/tests/smoke_test.sh
```

`mock backend` 的用途：

- 无 RKNN SDK、无相机、无板端时验证接口契约与前后端联通。
- 不代表生产默认路径。

## 13. 部署与同步

- `edge/deploy/push.sh`
  - 用于通过 `ssh + rsync` 直接同步边缘端代码到 `/opt/visionops_v3`
  - 默认同步：
    - `apps/collector_web`
    - `edge`
    - `interfaces`
    - `configs`
    - `deploy`
    - `tools`
    - 根目录 `README.md / AGENTS.md / CMakeLists.txt / .gitignore`
  - 默认不同步：
    - `build`
    - `tests`
    - `models`
    - `training`
    - 缓存、模型、压缩包、日志

当前仍以手动启动为主。`systemd` 服务化目录已预留，但尚未整理成完整的统一安装与升级流程。

## 14. 当前限制与后续 TODO

- `systemd` 完整服务化仍待统一整理。
- 真实现场长期稳定性仍需继续验证。
- Business App 接真实检测结果后的产线联调仍未完成。
- 模型热切换长期稳定性仍需在真机连续验证。
- 双 Runtime / 双业务并行部署仍需继续验证。
- RGA、HP60C Bridge、Orbbec Bridge 的真实性能收益仍需在真机实测确认。

## 补充说明

- `interfaces/schemas / interfaces/examples / interfaces/protocols` 是模块间契约，请勿随意删除。
- `edge/runtime_cpp/examples/mock_model_package`、`mock backend`、`test_image / mock frame source` 仍保留，用于无硬件测试与开发验证。
- 历史过程性交接文档已归档到 `docs/archive/handoff/`；当前理解系统请优先阅读本 README 和 `docs/handoff/current_state_2026_06_30.md`。

## 15. 服务端 MVP：数据、训练、模型包与设备分发

v3 边缘端初版完成后，服务端重新定位为：

```text
数据中心 + 训练中心 + 模型包中心 + 设备分发中心
```

服务端不替代边缘端 Collector Web，也不执行边缘端实时推理。边缘端主链路仍然保持：

```text
Camera Bridge / SDK Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
```

### 15.1 当前服务端已实现能力

当前 MVP 位于 `apps/server_api`，已支持：

- `GET /api/server/health`：服务健康检查。
- `GET /api/server/incoming-packages`：扫描 incoming 目录下尚未处理的 tar.gz。
- `POST /api/server/batches/process-incoming`：处理选中的 tar.gz，解压为数据批次 batch。
- `POST /api/server/batches/upload`：保留兼容入口，支持简单直传包。
- `GET /api/server/batches`：查看 batch 列表。
- `POST /api/server/batches/{batch_id}/accept`：接受批次。
- `POST /api/server/batches/{batch_id}/reject`：拒绝批次。
- `POST /api/server/datasets/build`：从 accepted batch 构建 dataset 清单。
- `POST /api/server/training/jobs`：创建训练任务。
- `GET /api/server/training/jobs/{job_id}/logs`：查看训练日志。
- `GET /api/server/model-packages`：查看模型包列表。
- `POST /api/server/model-packages/{model_id}/publish`：发布模型包到同步目录。
- `GET /api/server/devices` / `POST /api/server/devices`：设备注册表。
- `POST /api/server/devices/{device_id}/assign-model`：给设备分配目标模型。

说明：当前 training job 是 mock runner，用于验证任务编排和 v3 标准模型包生成契约；不执行真实 GPU 训练、ONNX 导出或 RKNN 转换。

### 15.2 启动服务端

```bash
cd /opt/visionops_v3

python3 -m apps.server_api.backend.main \
  --host 0.0.0.0 \
  --port 18100 \
  --data-root /opt/visionops_v3/server_data \
  --incoming-root /opt/visionops_v3/server_data/incoming \
  --publish-root /opt/visionops_v3/server_data/published_models
```

浏览器访问：

```text
http://127.0.0.1:18100/
```

### 15.3 服务端推荐工作流

```text
边缘端采集并导出 tar.gz
  -> tar.gz 复制或 Syncthing 同步到 server_data/incoming
  -> 服务端扫描 incoming 并处理选中上传包，生成 batch
  -> 第二步查看 manifest，确认任务类型，accept / reject
  -> 从 extracted/accepted batch 构建 dataset
  -> 创建 training job
  -> mock runner 生成 v3 model package
  -> publish 到 Syncthing 共享目录
  -> 边缘端 Collector Web 扫描 model.rknn + model.yaml
```

### 15.4 v3 标准模型包

服务端完整模型包：

```text
server_data/model_packages/<model_id>/
├── model.rknn
├── model.yaml
├── package.json
├── metrics.json
├── train_config.yaml.json
├── export_report.json
└── logs/
```

发布到边缘端时只复制：

```text
models/<model_name>/
├── model.rknn
└── model.yaml
```

`model.yaml` 是 Collector Web 扫描模型包的唯一元信息来源。

### 15.5 常用验证命令

```bash
curl http://127.0.0.1:18100/api/server/health | python3 -m json.tool

curl http://127.0.0.1:18100/api/server/incoming-packages | python3 -m json.tool

curl -X POST http://127.0.0.1:18100/api/server/batches/process-incoming \
  -H 'Content-Type: application/json' \
  -d '{"packages":["rk3576-001_package-test_20260707_085333.tar.gz"]}' | python3 -m json.tool

curl http://127.0.0.1:18100/api/server/batches | python3 -m json.tool

curl -X POST http://127.0.0.1:18100/api/server/batches/<batch_id>/accept \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"detection"}'

curl -X POST http://127.0.0.1:18100/api/server/datasets/build \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"detection","batch_ids":["<batch_id>"]}' | python3 -m json.tool

curl -X POST http://127.0.0.1:18100/api/server/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"<dataset_id>","task_type":"detection","epochs":50,"batch_size":16,"imgsz":640}' \
  | python3 -m json.tool

curl http://127.0.0.1:18100/api/server/model-packages | python3 -m json.tool

curl -X POST http://127.0.0.1:18100/api/server/model-packages/<model_id>/publish \
  -H 'Content-Type: application/json' \
  -d '{"publish_root":"/tmp/visionops_publish"}'
```

更多说明见：

- `docs/server/README.md`
- `docs/server/api.md`
- `docs/server/model_package_spec.md`
- `docs/server/workflow.md`

### 15.6 当前限制

- 真实训练流水线尚未接入。
- MLflow 目前只保留配置字段，未创建真实 run。
- ONNX 导出和 RKNN 转换为 stage 占位。
- 标注审核器尚未迁移到 v3 服务端。
- 设备分发当前只维护注册表和目标模型，尚未远程控制边缘端 Collector。
