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

M8 对 `edge/runtime_cpp/` 完成第一期结构拆分，但不接入真实 RKNN、RGA 或相机：

- `main.cpp` 只处理 CLI、信号和服务组装，是薄启动入口。
- `RuntimeApp` 编排健康状态、预览、单次推理、最近结果和快照能力。
- `RuntimeState` 通过互斥锁维护运行模式、序号、计数器和最近 Frame/Result。
- `HttpServer` 只负责 POSIX socket、HTTP 解析、路由和 JSON/JPEG 响应。
- `RknnRunnerMock` 与任务后处理模块生成标准 Mock `inference_result`，不调用 NPU。
- `StreamWorkerMock` 与 `SnapshotProvider` 提供 Mock Frame 和内置 JPEG，不连接设备。

该拆分保持 M3 HTTP API 完全兼容。M9.1 已在 `RuntimeApp` 初始化阶段加入轻量模型包元数据读取：manifest 提供模型包、平台和相对文件信息，YAML 提供任务、类别、输入尺寸和后处理阈值。解析结果统一进入 `loaded_model`，并供 Mock `inference_result.model` 复用。

M9.1 不打开 manifest 中的 `.rknn` 文件，不创建 RKNN Context，也不改变 Mock 推理路径。配置缺失时 Runtime 以 `degraded` 状态继续提供诊断接口。

M9.2 在 `rknn_runner` 边界加入统一 Runner 接口：

- 默认 `VISIONOPS_ENABLE_RKNN=OFF`，x86 和普通 CI 只构建 Mock Runner，不要求 SDK。
- RK3576/RK3588 部署构建可指定 RKNN 头文件与 Runtime 库，条件编译 `RknnRunnerReal`。
- Real Runner 管理模型二进制、RKNN Context、输入设置、执行、原始输出获取和资源释放。
- 未编译 RKNN 却选择 `--backend rknn` 时，Unavailable Runner 让服务保持可诊断，并明确报告降级原因。
- Runtime status 暴露 backend、Runner 加载状态、SDK 编译状态和错误；原始 tensor 不直接发送给 Gateway 或 Web。

M9.2 尚未迁移完整 YOLO detection、OBB 或 segmentation decode，`infer_once` 仍用标准 Mock 后处理维持接口契约，并通过 `debug` 标记 Real Runner 调用与原始输出数量。完整后处理将在 M9.3 接入；M10 再将真实相机取流放入 `stream_worker` 边界。设备逻辑不得重新堆入 `main.cpp` 或 `HttpServer`。

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

`edge/runtime_cpp/` 中的 Runtime Mock 是后续真实 C++ RKNN Runtime 的接口替身。它只用于契约、Collector 和 Gateway 集成开发，不能作为生产推理能力或性能基准。M8 改变的是内部职责组织，不改变这个定位。

M7 将 Collector Web 实际功能固定为三个工作区：Capture 通过 Collector 代理读取 Runtime snapshot；Validate 通过 Collector 触发 `infer_once` 并仅对标准结果做 bbox/OBB 可视化；Production 聚合 Runtime、Gateway 和 Business App 状态及寄存器。

M7.1 不改变这些数据边界，但将交互外壳回归旧版现场使用习惯：默认进入校验，顶部导航提供校验、采集上传、模型验证、设置和生产模式。校验和采集采用左侧步骤与大图像工作区，模型验证采用模型状态侧栏，生产模式使用状态、摘要和寄存器卡片。

三个页面都只访问 Collector 同源 API。Web 不直接取相机、不加载模型、不执行业务判断；下游不可达时页面显示 `unreachable`，Collector 本身保持可诊断。

因此 Web 结构与实际设备实现解耦：将 Runtime Mock 替换为真实 C++ RKNN Runtime，或将 Gateway/Business App Mock 替换为现场服务时，Collector 同源 API 契约保持稳定，前端总体导航和页面不需重写。

## 5. Gateway Adapter

Gateway Adapter 将标准化 `InferenceResult` 转换为上层 Gateway 所需消息，负责：

- 消息字段映射和协议版本管理。
- 重试、超时、限流和断线策略。
- 请求关联、幂等键和端到端时间戳。
- 连接状态与上传指标。

Gateway 不可用时不得阻塞推理线程。缓存必须有容量和过期策略，默认不能无限积压图片或结果。

M5 的 `edge/gateway_adapter/` 是 Gateway Mock，它从 Collector Web 或 Runtime Mock 拉取标准 `inference_result`，生成 `gateway_message`，再更新内存 Holding Registers。该闭环只用于契约和联调，不连接真实 PLC，也不表示生产网络的实时性与故障安全性。

Gateway 必须保持模型无关：只消费标准结果中的决策、detections、几何和耗时字段，不读取 RKNN 原始张量。`carton_tube_check`、`carton_partition_check` 等业务应在 Gateway app 层扩展专用 register map。

M6 将该扩展点具体化为业务 App Mock 层：

```text
standard inference_result
  -> business rules
  -> AppDecision
  -> GatewayMessage
  -> app-specific Holding Registers
```

`carton_tube_check` 使用 `100..119`，`carton_partition_check` 使用 `200..219`。两者复用 M5 Modbus Adapter，不重复实现 Modbus 协议栈。业务 App 可用 `file` Mock 输入独立调试，也可从 Collector 或 Runtime 读取相同的 `/api/runtime/latest_result`。未来切换真实 upstream 时不改业务决策和寄存器协议。

Collector Web 可聚合展示 AppDecision 和寄存器状态，但不执行纸筒或隔板判断。C++ Runtime 只产生标准推理结果，不包含现场业务阈值、PLC 寄存器或最终工位语义。

## 6. Modbus Adapter

Modbus Adapter 负责将业务结果映射到 Modbus TCP/RTU 寄存器或线圈，并与 PLC 交互。映射关系属于应用配置，不应写死在 Runtime 中。

适配器需要明确：

- 客户端或服务端角色。
- 寄存器地址、数据类型、字节序和缩放方式。
- 心跳、结果有效位、序号和应答机制。
- 通信超时、重连和故障安全值。
- 多帧结果覆盖、排队或聚合策略。

PLC 通信失败应独立告警，不得导致相机和推理服务崩溃。

M5 的 Modbus TCP Mock 默认监听 `1502`，不使用需特权的 `502`。它只支持 Holding Registers 的 FC03、FC06 和 FC16，寄存器只表达心跳、序号、业务决策、几何和耗时摘要，不传输图片或大块 JSON。

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
