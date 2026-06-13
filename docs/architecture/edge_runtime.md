# 边缘端运行时架构

## 1. 主链路

VisionOps v3 的边缘生产主链路固定为：

```text
Camera Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Gateway / Modbus
  -> PLC 或上层系统
```

这里的箭头表示数据或控制关系，不要求所有组件串行转发同一份数据。实时推理结果可以由 Runtime 同时发布给 Collector Web 和通信适配器，避免 Web 成为生产链路瓶颈。

## 2. Camera Bridge

Camera Bridge 是相机与推理运行时之间的唯一生产接入层，职责包括：

- 管理 RTSP、厂商 SDK 或设备媒体接口。
- 处理断线重连、超时、帧率限制和时间戳。
- 输出约定的像素格式、尺寸和帧描述信息。
- 在平台允许时采用零拷贝或低拷贝传输。
- 暴露健康状态、实际帧率、丢帧数和最近错误。

Camera Bridge 不负责模型预处理、推理、业务判断或 PLC 通信。相机厂商差异应封装在该模块内部。

建议优先考虑 Unix Domain Socket、共享内存或 DMA-BUF 等本机传输方案。具体选择应基于设备能力验证，协议定义放入 `interfaces/protocols/`。

## 3. C++ RKNN Runtime

C++ RKNN Runtime 是生产推理核心，职责包括：

- 校验并加载版本化模型包。
- 管理 RKNN Context、NPU 核心和模型生命周期。
- 完成 resize、颜色转换、归一化等预处理。
- 执行 RKNN 推理并解析输出张量。
- 完成 detection、classification、OBB、segmentation 或组合任务后处理。
- 生成与模型实现无关的标准化推理结果。
- 记录端到端耗时、队列等待、推理耗时和错误计数。
- 支持健康检查、优雅停止和可回滚的模型切换。

Runtime 不直接读写 Web 表单，不直接映射 PLC 寄存器，也不依赖 Python 解释器。模型特定后处理可以插件化，但必须遵守统一 Runtime 接口。

## 4. Collector Web

Collector Web 默认由 Python Web 服务和静态前端组成，负责：

- 设备与组件状态展示。
- 相机、任务、应用和通信配置管理。
- 模型列表、模型切换和部署结果展示。
- 单帧验证、结果可视化和生产监控。
- 日志摘要、诊断包生成和服务控制入口。
- 采集任务管理以及数据上传编排。

Collector Web 通过受控接口与 Camera Bridge、Runtime 和通信适配器交互。它不得加载 RKNN 模型作为生产推理路径，也不得通过修改其他服务内部文件来控制服务。

M4 阶段进一步固定以下边界：

- Collector Web 只做配置、状态、诊断、低频预览和 Runtime HTTP 代理。
- Collector Web 不做生产推理，不加载模型，不调用 RKNN 或 NPU。
- Collector Web 不直接连接相机；相机接入只能属于 Camera Bridge。
- 浏览器前端只访问 Collector 同源接口，不直接访问 Runtime 端口。
- Collector 聚合状态在 Runtime 不可达时仍可用，以便现场判断是管理面故障还是 Runtime 故障。
- Runtime 的业务状态码由代理保留，例如尚无结果时的 404 不转换为 Collector 500。

`edge/runtime_cpp/` 中的 Runtime Mock 是后续真实 C++ RKNN Runtime 的接口替身。它只用于契约、Collector 和 Gateway 集成开发，不能作为生产推理能力或性能基准。

## 5. Gateway Adapter

Gateway Adapter 将标准化 `InferenceResult` 转换为上层 Gateway 所需消息，负责：

- 消息字段映射和协议版本管理。
- 重试、超时、限流和断线策略。
- 请求关联、幂等键和端到端时间戳。
- 连接状态与上传指标。

Gateway 不可用时不得阻塞推理线程。缓存必须有容量和过期策略，默认不能无限积压图片或结果。

## 6. Modbus Adapter

Modbus Adapter 负责将业务结果映射到 Modbus TCP/RTU 寄存器或线圈，并与 PLC 交互。映射关系属于应用配置，不应写死在 Runtime 中。

适配器需要明确：

- 客户端或服务端角色。
- 寄存器地址、数据类型、字节序和缩放方式。
- 心跳、结果有效位、序号和应答机制。
- 通信超时、重连和故障安全值。
- 多帧结果覆盖、排队或聚合策略。

PLC 通信失败应独立告警，不得导致相机和推理服务崩溃。

## 7. 控制流与数据流

数据流以 Frame ID 为关联主键：Camera Bridge 产生 Frame ID，Runtime 在结果中原样携带，Gateway/Modbus 和 Web 使用该标识关联时间、图像与结果。

控制流包括配置应用、模型切换、服务重启和诊断请求。控制命令必须鉴权、记录操作者或来源，并返回明确状态，不使用“写文件后等待服务自行发现”的隐式控制方式。

## 8. 启动与依赖

建议的 systemd 依赖顺序为：

```text
基础网络与设备节点
  -> Camera Bridge
  -> C++ RKNN Runtime
  -> Gateway/Modbus Adapter
  -> Collector Web
```

组件应允许依赖服务暂时不可用，并自行重连。启动顺序只是优化，不是正确性的唯一保障。

## 9. 可观测性

所有组件至少提供：版本、运行时长、健康状态、最近错误、吞吐、延迟和资源使用情况。日志使用统一时间基准，并携带 `component`、`device_id`、`frame_id`、`model_version` 等可用字段。

诊断工具可以由 Python 编排，但收集过程不能中断生产服务，也不能默认打包模型、原始图片或密钥。
