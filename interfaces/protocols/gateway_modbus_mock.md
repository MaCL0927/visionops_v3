# Gateway / Modbus Mock 契约

## 1. 用途与限制

本契约用于 M5 的本机集成验证：

```text
standard inference_result
  -> Gateway Mock
  -> gateway_message
  -> HoldingRegisterBank
  -> Modbus TCP Mock
```

M5 不连接真实 PLC、串口或 Modbus RTU，不代表生产安全、时序或实时性结论。Gateway 消费标准 `inference_result`，不依赖 RKNN 张量或模型内部结构。

## 2. 上游契约

Gateway 对 Collector 和 Runtime 使用同一路径：

```http
GET /api/runtime/latest_result
```

- `200`：返回可映射的 `inference_result`。
- `404`：上游可达，但尚无最新结果。Gateway 保持存活。
- 连接失败：Gateway 记录 `unreachable`，保持 HTTP 和 Modbus Mock 可用。

## 3. Gateway HTTP 调试接口

### `GET /health`

返回 Gateway 进程自身状态、`device_id`、`app_id`、上游 URL 和 Modbus 端口。上游不可达不会使该接口失败。

### `GET /api/gateway/status`

返回 `gateway`、`upstream`、最新标识、最新 `gateway_message`、寄存器快照和计数器。

### `POST /api/gateway/poll_once`

立即拉取一次上游结果。成功时返回新的 `gateway_message`；尚无结果返回 `404`；上游不可达返回 `502` JSON 错误。

### `GET /api/gateway/latest_message`

返回最近一次成功生成的 `gateway_message`，尚无消息时返回 `404` JSON 错误。

### `GET /api/gateway/registers`

返回 Holding Registers 快照，每项包含 `address`、`name`、`value`、`type`、`scale` 和 `description`。

## 4. 默认 Holding Register Map

| 地址 | 名称 | 说明 |
| ---: | --- | --- |
| 0 | `heartbeat` | 每次新消息翻转 |
| 1 | `sequence` | Gateway 消息序号 |
| 2 | `status_code` | 上游状态摘要 |
| 3 | `final_code` | 最终业务决策码 |
| 4 | `ok` | 布尔结果，0/1 |
| 5 | `reason_code` | reason 稳定摘要 |
| 6 | `error_code` | error code 稳定摘要 |
| 7 | `object_count` | 检测目标数 |
| 8 | `score_x1000` | 最高分乘 1000 |
| 9-10 | `center_x`, `center_y` | 最高分目标中心 |
| 11-14 | `bbox_x1`..`bbox_y2` | 最高分目标包围框 |
| 15-16 | `inference_ms`, `total_ms` | 推理与总耗时整数摘要 |
| 17-18 | `frame_id_low`, `result_id_low` | ID 的 16 位稳定摘要 |
| 19 | `reserved` | 保留 |

寄存器只表达紧凑业务结果，不用于传输图片、mask、模型张量或大块 JSON。

## 5. Modbus TCP Mock

- 默认端口为 `1502`，不使用 `502`。
- 支持 FC03、FC06 和 FC16。
- 不支持的 function code 返回 exception `0x01`。
- 非法地址返回 exception `0x02`。
- 非法数量或 PDU 返回 exception `0x03`。

## 6. 后续扩展

`generic_mock` 的默认映射只用于契约验证。`carton_tube_check`、`carton_partition_check` 等真实业务应在 Gateway app 层定义专用 register map、握手、超时、字节序和故障安全值。不允许把 v2 Modbus 服务原样复制到 v3。
