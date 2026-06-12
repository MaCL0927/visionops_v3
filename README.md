# VisionOps v3

VisionOps v3 是面向工业视觉场景重新设计的端到端视觉 AI 软件平台。项目以 RK3588、LB3576 等 Rockchip 边缘设备为主要部署目标，覆盖数据采集、标注审核、训练评估、模型导出、设备部署、边缘推理、现场配置以及 Gateway/Modbus/PLC 通信。

v3 不复制 v2 的目录和实现，而是从清晰的职责边界、稳定的接口契约和可维护的部署单元出发重新建设。当前仓库仅建立架构与文档骨架，不包含业务代码、模型或数据。

## 设计目标

- 建立服务端、训练端和边缘端之间清晰、可测试的边界。
- 将 C++ RKNN Runtime 作为生产推理唯一主链路。
- 使用 Python 承担 Web、配置、训练、导出、部署编排和诊断工作。
- 通过显式协议与数据结构连接模块，避免共享内部实现。
- 同时支持 RK3588 与 LB3576，并隔离设备差异。
- 配置可追踪、可校验、可生成，生产运行参数可审计。
- 从 v2 按能力迁移，而不是按文件迁移。

## 核心链路

### 模型生产链路

```text
边缘数据采集
  -> 服务端接收与数据管理
  -> 标注与审核
  -> 训练与评估
  -> ONNX 导出
  -> RKNN 转换与验证
  -> 版本化模型包
  -> 边缘设备部署
```

### 边缘生产链路

```text
Camera Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Gateway / Modbus
  -> PLC 或上层业务系统
```

Camera Bridge 负责稳定取流与帧交付；C++ RKNN Runtime 负责预处理、NPU 推理和后处理；Collector Web 负责配置、状态、诊断和结果展示；Gateway/Modbus 负责将标准化结果转换为现场通信协议。

## 模块组成

```text
visionops_v3/
├── apps/                    # Python 应用：服务端 API 与边缘 Web
│   ├── server_api/
│   └── collector_web/
├── training/                # Python 训练、评估、导出与转换编排
│   ├── pipeline/
│   └── export/
├── edge/                    # 边缘生产运行组件
│   ├── camera_bridge/       # 相机接入与帧传输
│   ├── rknn_runtime/        # C++ RKNN 生产推理主链路
│   ├── gateway_adapter/     # Gateway 协议适配
│   ├── modbus_adapter/      # Modbus/PLC 协议适配
│   └── common/              # 仅限边缘端稳定公共能力
├── interfaces/              # 跨进程、跨语言接口与协议定义
│   ├── schemas/
│   └── protocols/
├── configs/                 # 人工配置模板与生成配置目录
│   ├── edge/
│   ├── task/
│   ├── app/
│   └── runtime/
├── deploy/                  # systemd、部署脚本和发布包定义
├── docs/                    # 架构、迁移与历史说明
├── tests/                   # 单元测试与集成测试
└── tools/                   # 开发、校验和诊断工具
```

## 架构文档

- [整体架构](docs/architecture/overview.md)
- [边缘端运行时](docs/architecture/edge_runtime.md)
- [配置设计](docs/architecture/config_design.md)
- [从 v2 迁移](docs/migration/from_v2.md)

## 当前阶段

当前仅完成基础仓库结构与架构约束。后续业务能力应先定义接口和验收标准，再以小模块逐步实现，不以复制 v2 代码作为起点。

## 仓库约束

仓库禁止提交真实环境变量、模型、数据集和压缩制品，包括 `.env`、`.pt`、`.onnx`、`.rknn`、采集图片、视频和部署压缩包。详细协作规则见 [AGENTS.md](AGENTS.md)。
