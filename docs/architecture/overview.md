# VisionOps v3 整体架构

## 1. 分层原则

VisionOps v3 将平台能力和现场方案分开：

```text
平台控制面：Server API / Collector Web / 配置与部署
平台数据面：Camera Bridge / C++ RKNN Runtime / Modbus 基础库
现场方案层：production/<line_id>/tasks + gateway + config + deploy
```

平台层不得包含某一条产线的类别名、算法阈值、PLC 地址或仿射参数；现场方案不得复制 Runtime、Collector、相机 Bridge 或 Modbus 协议栈。

## 2. 数据链路

```text
服务端数据闭环
上传包 -> 标注审核 -> 数据集 -> 训练 -> ONNX -> RKNN -> 模型包 -> 设备部署

边缘生产闭环
Camera Bridge -> C++ RKNN Runtime -> 标准 inference_result
                    ├-> Collector Web
                    └-> Production Gateway -> Modbus-TCP -> PLC
```

Collector Web 是管理和观察入口，不位于 PLC 触发的关键路径。Gateway 应直接调用 Runtime，避免 Web 故障影响生产结果。

## 3. 目录所有权

- `apps/`：Web 与服务端控制面。
- `edge/`：可被多个产线复用的边缘基础能力。
- `production/`：按产线组织的业务算法、寄存器语义、配置和部署。
- `interfaces/`：跨进程契约和 Schema。
- `training/`：训练、导出和模型包生成。
- `configs/`：通用平台示例配置；现场主配置放在对应产线目录。

## 4. 故障边界

Camera Bridge、Runtime、Gateway 和 Collector 以独立进程运行：

- Collector 重启不得中断 PLC 检测；
- Gateway 不可用不得导致 Runtime 崩溃；
- 相机失败应返回可诊断错误，不得让 Web 进程退出；
- 模型切换失败必须保留上一可用模型；
- 生产任务之间使用独立 Runtime 端口和模型目录。

## 5. 新产线接入

新产线优先复用现有平台组件，并新增：

```text
production/<line_id>/
├── config/line.yaml
├── tasks/<task_id>/
├── gateway/
├── scripts/
└── deploy/
```

只有真正通用、至少被两条产线复用的能力，才允许从 `production/` 上移到 `edge/` 或 `apps/`。
