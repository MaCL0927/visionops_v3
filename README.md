# VisionOps v3 端到端视觉 AI 平台

## 1. 项目定位

VisionOps v3 面向 `RK3576 / LB3576 / RK3588` 工业视觉盒子，覆盖数据采集、服务端标注与训练、模型转换发布、边缘端 RKNN 推理、Web 管理以及 PLC / 上位机通信。

当前生产边缘主链路固定为：

```text
Camera Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Production Line Gateway / Modbus-TCP or task TCP client
  -> PLC / 机器人调度系统 / 上位机
```

平台代码与现场业务代码严格分开：

- `apps/`、`edge/`、`training/` 提供可复用平台能力。
- `production/` 保存具体产线方案、任务算法、现场配置和部署文件。
- 新增现场任务不得继续散落到 `edge/`、`configs/`、根目录 `scripts/` 等多个位置。

## 2. 当前目录结构

```text
visionops_v3/
├── apps/                    # Collector Web 与 Server API
├── edge/                    # 通用边缘能力：相机、Runtime、Modbus、Gateway 基础工具
├── production/              # 实际产线方案，按产线和任务组织
│   └── carton_line/         # 纸隔板 + 纸筒产线
├── training/                # 训练、导出、RKNN 转换与模型打包
├── interfaces/              # JSON Schema、协议与示例
├── configs/                 # 通用平台示例配置，不放现场专用配置
├── scripts/                 # 通用服务启动脚本
├── tools/                   # 开发、校验和诊断工具
├── tests/                   # 当前有效的自动化测试
├── docs/                    # 当前架构、服务端说明和迁移原则
├── models/                  # 本地模型包目录，不进入 Git
└── server_data/             # 服务端运行数据，不进入 Git
```

## 3. 核心模块边界

### `apps/collector_web`

负责边缘端 Web、状态聚合、配置管理、模型切换、采集上传和生产画面。它不直接读取相机、不加载 RKNN 模型，也不执行现场业务判断。

### `apps/server_api`

负责服务端上传包接收、标注审核、数据集构建、训练任务、模型包发布和设备部署。

### `edge/runtime_cpp`

负责真实生产推理：取帧、预处理、RKNN、后处理、标准 `inference_result`、快照与模型切换。

### `edge/camera_bridge`

封装厂商相机 SDK 和取流差异。当前包含 Orbbec Gemini 336L Bridge。

### `edge/modbus_adapter`

提供通用 Holding Register Bank 和最小 Modbus-TCP Server。具体寄存器定义由生产方案传入，不在通用适配层硬编码。

### `production`

保存现场方案。当前 `production/carton_line/` 内包含：

- 隔板 5×8 小方格结构检测；
- 纸筒站立/倒伏与 RGB-Depth 高度判断；
- 双机械手坐标转换；
- 统一 Robot Protocol Gateway；
- 三套 Runtime、三套 Collector、一个 Modbus-TCP 服务和一个机器人调度 TCP Client；
- 单一产线配置文件和 systemd 部署文件。

完整说明见：

```text
production/carton_line/README.md
```

## 4. 模型包规范

边缘模型包固定为：

```text
models/<task>/<model_version>/
├── model.rknn
└── model.yaml
```

生产默认目录：

```text
models/carton_partition_check/current/
models/carton_tube_check/current/
models/tube_pick_vision/current/
```

`model.yaml` 是模型任务类型、类别、输入尺寸和模型标识的唯一元信息来源。

## 5. 编译 C++ Runtime

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

当前二进制名称仍为：

```text
build-rknn/edge/runtime_cpp/visionops_runtime_mock
```

名称保留是为了兼容已有部署脚本；当使用 `--backend rknn` 时实际运行真实 RKNN 路径。

## 6. 通用单实例启动

```bash
./scripts/start_runtime.sh /opt/visionops_v3/models/<model_dir>
./scripts/start_collector.sh
./scripts/start_server.sh
```

这些脚本用于单实例调试。纸隔板/纸筒生产线应使用 `production/carton_line/scripts/` 下的专用启动脚本。

## 7. 纸隔板与纸筒生产线启动

```bash
cd /opt/visionops_v3

./production/carton_line/scripts/start_runtime.sh partition
./production/carton_line/scripts/start_runtime.sh tube
./production/carton_line/scripts/start_runtime.sh pick
./production/carton_line/scripts/start_gateway.sh
./production/carton_line/scripts/start_tcp_pick.sh
./production/carton_line/scripts/start_collector.sh partition
./production/carton_line/scripts/start_collector.sh tube
./production/carton_line/scripts/start_collector.sh pick
```

唯一主配置：

```text
production/carton_line/config/line.yaml
```

安装 systemd：

```bash
sudo bash production/carton_line/deploy/install_services.sh
```

## 8. 通用验证

```bash
curl -s http://127.0.0.1:28081/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:28081/api/runtime/infer_once | python3 -m json.tool
curl -s http://127.0.0.1:18091/health | python3 -m json.tool
curl -s http://127.0.0.1:19090/health | python3 -m json.tool
```

运行当前自动化测试：

```bash
python3 -m pytest tests/unit tests/integration
```

硬件、真实 RKNN、真实相机和 PLC 结果仍必须在 LB3576 上单独验收，不能以 x86 Mock 测试代替。

## 9. 仓库卫生

以下内容不进入 Git：

- `__pycache__`、pytest 缓存、构建目录；
- `.env`、密钥和设备私有配置；
- `.pt`、`.onnx`、`.rknn`；
- 数据集、采集图片、视频、日志和诊断结果；
- `server_data`、训练输出和发布制品；
- 压缩包及一次性调试文件。

实际设备配置放在 `/etc/visionops_v3/`，仓库只保留 `*.env.example` 和可审查的 YAML 模板。
