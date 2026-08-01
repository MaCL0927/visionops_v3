# VisionOps v3 端到端视觉 AI 平台

VisionOps v3 是面向 `RK3576 / LB3576 / RK3588` 工业视觉盒子的视觉 AI 平台，覆盖服务端数据管理、标注审核、训练导出、模型发布、边缘端 RKNN 推理、Web 管理、生产业务算法以及 PLC / 机器人 / 上位机通信。

当前代码、配置和 Git 历史是事实来源。旧版 v2 只能作为功能参考，不能恢复 v2 的 Python RKNN 生产主链路，也不能把边缘端、服务端和产线业务逻辑重新混在一起。

## 1. 总体架构

```text
服务端:
  数据上传 -> 标注/审核 -> 数据集 -> 训练 -> ONNX/RKNN 导出
    -> v3 模型包发布 -> 同步到边缘端 models/

边缘端:
  Camera Bridge / SDK Bridge
    -> C++ RKNN Runtime
    -> Collector Web
    -> Production App / Gateway / Modbus / Robot Client
    -> PLC / 机器人调度系统 / 上位机
```

核心边界：

- C++ Runtime 负责取帧、预处理、RKNN 推理、后处理、标准 `inference_result`、快照和模型切换。
- Collector Web 负责边缘端配置、展示、状态聚合、采集上传、模型验证和生产页面；不直接连接相机、不加载模型、不解析 RKNN 原始 tensor。
- Camera Bridge 负责 HP60C、Orbbec 336L 等相机 SDK/取流差异。
- Production App 负责具体产线业务判断，例如纸箱、洗衣液、泡沫圆环抓取。
- Gateway / Modbus / Robot Client 负责通信协议和寄存器映射。
- Server API 负责数据、标注、训练、模型包、设备分发管理；不做边缘端实时推理。

## 2. 顶层目录

```text
apps/
  collector_web/       边缘端 Web 与后端代理
  server_api/          服务端 API、Web 控制台、标注器
edge/
  runtime_cpp/         C++ RKNN Runtime，含 mock/real backend、RGA、帧源
  camera_bridge/       HP60C 与 Orbbec 336L Bridge
  modbus_adapter/      通用 Modbus TCP/Holding Register 基础库
  gateway_adapter/     Gateway 消息基础结构
production/
  carton_line/         纸隔板、纸筒、取筒产线
  carton_palletizing/  纸箱码垛与抓取点任务
  detergent_grasp/     洗衣液抓取任务
  foam_ring_grasp/     泡沫圆环 RGB-D 抓取几何任务
training/
  pipeline/            preprocess/train/evaluate/export/convert/package stages
interfaces/            JSON Schema、协议和跨模块示例
configs/               通用 example 配置
models/                本地模型目录，权重和模型包默认不进 Git
server_data/           服务端运行数据目录，仅保留 .gitkeep
tools/                 配置、接口、存储和性能诊断工具
tests/                 当前有效单元测试与集成测试
docs/                  架构、服务端、迁移和任务说明
scripts/               通用启动、环境和清理脚本
```

## 3. 支持任务类型

平台层支持以下模型任务类型：

- `detection`
- `obb`
- `segmentation`
- `classification`

生产任务可以在 `production/<line_id>/tasks/<task_id>/` 中把标准 `inference_result` 转换为现场业务结果、机器人协议或 Modbus 寄存器。业务规则不得写入通用 Runtime。

## 4. 服务端到生产部署流程

典型闭环：

1. 边缘端 Collector Web 采集图片、RGB-D 或数据包。
2. Server API 接收上传包并登记 batch。
3. 人工审核、标注或使用 SAM 辅助生成 segmentation 多边形。
4. 从 accepted batches 构建 dataset。
5. 创建 training job，执行 `preprocess -> train -> evaluate -> export_onnx -> convert_rknn -> package_v3_model`。
6. 生成 v3 模型包，包含 `model.rknn` 和 `model.yaml`。
7. publish 到同步目录或复制到边缘端 `models/`。
8. Collector Web 在模型验证页扫描模型目录并请求 Runtime 切换模型。
9. 生产模式中 Runtime 输出标准结果，Production App / Gateway / Modbus 继续处理。

服务端详细说明见：

- [docs/server/README.md](docs/server/README.md)
- [docs/server/api.md](docs/server/api.md)
- [docs/server/workflow.md](docs/server/workflow.md)
- [docs/server/model_package_spec.md](docs/server/model_package_spec.md)

## 5. 模型目录规范

预训练权重统一放在：

```text
models/pretrained/
```

其中 `yolo26n.pt` 是 Ultralytics AMP 检查可能使用的预训练模型，规范位置是：

```text
models/pretrained/yolo26n.pt
```

仓库根目录不应出现 `yolo26n.pt`。训练 pipeline 会从 repo root 解析 `models/pretrained/yolo26n.pt`，并在 job work 目录暴露给 AMP 检查逻辑，避免第三方库在仓库根目录重新生成该文件。

发布到边缘端的 Runtime 模型包采用：

```text
models/<task_or_model_name>/
├── model.rknn
└── model.yaml
```

`model.yaml` 是边缘端模型扫描、模型切换、类别、输入尺寸、后处理阈值和任务类型的主要元信息来源。真实 `.pt`、`.onnx`、`.rknn` 和采集数据默认不进入 Git。

## 6. 常用启动命令

创建边缘端 Python 环境：

```bash
cd /opt/visionops_v3
sudo bash scripts/setup_edge_env.sh
```

编译 C++ Runtime（RKNN + RGA）：

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

启动 Runtime：

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

二进制名称 `visionops_runtime_mock` 为兼容历史部署脚本保留；当 `--backend rknn` 时运行真实 RKNN 路径。

启动 Collector Web：

```bash
source /opt/visionops_v3/venv/bin/activate

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 \
  --port 18091 \
  --runtime-url http://127.0.0.1:28081 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
  --models-root /opt/visionops_v3/models \
  --device-id lb3576-dev
```

启动 Server API：

```bash
source /opt/visionops_v3/venv/bin/activate
python3 -m apps.server_api.backend.main --host 0.0.0.0 --port 18100
```

## 7. 常用检查命令

Runtime：

```bash
curl -s http://127.0.0.1:28081/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:28081/api/runtime/start_preview | python3 -m json.tool
curl -s http://127.0.0.1:28081/api/runtime/status | python3 -m json.tool
curl -s -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
curl -s http://127.0.0.1:28081/api/runtime/snapshot.jpg -o /tmp/runtime_snapshot.jpg
```

Collector Web：

```bash
curl -s http://127.0.0.1:18091/health | python3 -m json.tool
curl -s http://127.0.0.1:18091/api/models | python3 -m json.tool
curl -s -X POST http://127.0.0.1:18091/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"package_dir":"test_rknn_model"}' | python3 -m json.tool
```

Server API：

```bash
curl -s http://127.0.0.1:18100/api/server/health | python3 -m json.tool
curl -s http://127.0.0.1:18100/api/server/batches | python3 -m json.tool
curl -s http://127.0.0.1:18100/api/server/training/jobs | python3 -m json.tool
curl -s http://127.0.0.1:18100/api/server/model-packages | python3 -m json.tool
```

## 8. Camera Bridge

当前包含：

- HP60C / HP60CN Angstrong SDK Bridge：`edge/camera_bridge/hp60c_bridge/`
- Orbbec Gemini 336L Bridge：`edge/camera_bridge/orbbec336l_bridge/`

相机选择与双相机说明见：

- [docs/HP60C_ORBBEC_DUAL_CAMERA_INTEGRATION.md](docs/HP60C_ORBBEC_DUAL_CAMERA_INTEGRATION.md)
- [edge/camera_bridge/hp60c_bridge/README.md](edge/camera_bridge/hp60c_bridge/README.md)
- [edge/camera_bridge/orbbec336l_bridge/README.md](edge/camera_bridge/orbbec336l_bridge/README.md)

硬件取流、深度采样、SDK 重连和 watchdog 行为必须在 RK 板和真实相机上验证。

## 9. Production 任务索引

当前主要 production 任务：

- `production/carton_line/`：纸隔板、纸筒、取筒视觉与统一 Robot Gateway。
- `production/carton_palletizing/`：纸箱码垛、first-layer placement、box grasp vision。
- `production/detergent_grasp/`：洗衣液检测与抓取。
- `production/foam_ring_grasp/`：泡沫圆环 RGB-D 几何、轴线、抓取点、碰撞和 M35.4 有向定长 3D 轴杆诊断。

泡沫圆环任务入口：

- [production/foam_ring_grasp/README.md](production/foam_ring_grasp/README.md)
- [docs/M35.4_DIRECTED_FIXED_LENGTH_AXIS_ROD.md](docs/M35.4_DIRECTED_FIXED_LENGTH_AXIS_ROD.md)

## 10. 测试与开发

PC 上可运行的基础检查：

```bash
python3 -m pytest tests/unit
python3 -m pytest tests/integration
git diff --check
```

训练 pipeline 相关快速测试：

```bash
python3 -m pytest tests/unit/test_server_api_services.py tests/unit/test_sam_annotation_service.py
```

泡沫圆环几何相关测试：

```bash
python3 -m pytest \
  tests/unit/test_foam_ring_rim_pinch_geometry.py \
  tests/unit/test_foam_ring_axis_visualization.py
```

硬件相关测试需要 RK3576/LB3576、真实相机、真实 RKNN、机器人或 Modbus/PLC 环境时，可以跳过，但必须在变更报告中说明。

## 11. 部署与同步

通用代码同步脚本：

```bash
bash edge/deploy/push.sh --host <edge-ip> --user <ssh-user>
```

该脚本用于把代码同步到边缘端固定目录，默认不应同步 `models/`、`server_data/`、训练输出、缓存、日志和真实数据。模型同步应走发布目录、Syncthing、rsync 白名单或现场部署流程。

产线 systemd 与启动入口优先放在对应 `production/<line_id>/deploy/` 和 `production/<line_id>/scripts/` 下。

## 12. 仓库卫生

不进入 Git：

- `__pycache__/`、`*.pyc`、`.pytest_cache/`
- CMake build 目录和编译产物
- `.env`、密钥、设备私有配置
- `.pt`、`.onnx`、`.rknn`
- 数据集、采集图片、视频、日志、诊断包
- `server_data/` 运行数据
- `runs/`、`mlruns/`、训练中间产物和发布制品
- 压缩包和一次性调试输出

保留：

- `*.env.example`
- `*.example.yaml`
- `interfaces/examples`
- 测试 fixture
- 小型模型元数据示例
- 部署脚本和 systemd 文件

本次清理记录见 [docs/REPOSITORY_CLEANUP_REPORT.md](docs/REPOSITORY_CLEANUP_REPORT.md)。
