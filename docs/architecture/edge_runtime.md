# 边缘端运行时架构

## 1. 组件职责

### Camera Bridge

封装 RTSP、USB 或厂商 SDK，提供快照、视频流、深度图、profile 和健康状态。它不加载模型，也不执行业务判断。

### C++ RKNN Runtime

负责：

- 读取 Camera Bridge 帧；
- 预处理与 RGA 加速；
- RKNN Context 和 NPU 推理；
- Detection、Classification、OBB、Segmentation 后处理；
- 标准 `inference_result`；
- 模型包扫描、加载与切换；
- `health/status/infer_once/latest_result/snapshot.jpg` API。

Runtime 不包含 PLC 地址、纸筒高度阈值、隔板模板或机器人坐标。

### Collector Web

负责配置、状态、预览、采集上传、模型验证和生产画面。浏览器仅访问 Collector 同源 API，不直接连接 Runtime、Gateway 或相机端口。

### Modbus Adapter

提供通用寄存器 Bank 和 TCP 协议处理。寄存器定义由具体生产方案提供。

### Production Gateway

位于 `production/<line_id>/gateway/`，负责：

- 接收 PLC 触发；
- 主动调用相应 Runtime 的 `infer_once`；
- 调用任务算法；
- 写回结果与坐标寄存器；
- 保存受控调试结果；
- 对 Collector 暴露业务兼容接口。

## 2. 多任务方式

同一设备上的不同生产任务采用：

```text
每个任务一个 Runtime
每个任务一个 Collector
一条产线一个统一 Gateway / Modbus Server
```

这样模型、`latest_result` 和 Web 状态互不覆盖，而 PLC 寄存器仍由一个服务统一维护。

## 3. 配置边界

通用 Runtime 和 Collector 可使用命令行或通用示例配置。现场任务的端口、模型目录、业务参数、深度阈值和坐标标定统一放入：

```text
production/<line_id>/config/line.yaml
```

设备实际生效配置安装到 `/etc/visionops_v3/`。仓库不提交真实 `.env`。
