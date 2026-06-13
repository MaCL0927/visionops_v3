# Gateway Adapter Mock

`edge/gateway_adapter/` 是 M5 阶段的 Gateway Mock，用于验证从标准 `inference_result` 到 `gateway_message` 再到 Holding Registers 的最小闭环。它只是接口替身，不连接真实 PLC、真实串口或生产 Modbus 网络。

## 责任边界

- 从 Collector Web 或 Runtime Mock 读取 `/api/runtime/latest_result`。
- 只依赖 `interfaces/` 定义的标准推理结果，不解析模型原始张量。
- 生成 `gateway_message` 并更新默认寄存器映射。
- 提供 HTTP 调试接口、上游状态和轻量计数器。
- 上游不可达或尚无结果时保持进程存活。

Gateway 不传输图片、模型文件或大块 JSON。真实业务如 `carton_tube_check` 和 `carton_partition_check` 应在 Gateway app 层扩展专用决策逻辑与 register map，不应把规则写入 C++ Runtime。

## 启动

```bash
python -m edge.gateway_adapter.gateway_mock_service \
  --host 0.0.0.0 \
  --port 19090 \
  --upstream-url http://127.0.0.1:8090 \
  --upstream-kind collector \
  --modbus-host 0.0.0.0 \
  --modbus-port 1502
```

调试接口：

- `GET /health`
- `GET /api/gateway/status`
- `POST /api/gateway/poll_once`
- `GET /api/gateway/latest_message`
- `GET /api/gateway/registers`

`--upstream-kind runtime` 可用于绕过 Collector 直接验证 Runtime Mock，两种模式均读取 `/api/runtime/latest_result`。

## 验证

```bash
python -m pytest tests/unit/test_gateway_mapping.py tests/integration/test_gateway_modbus_mock.py
bash edge/gateway_adapter/tests/smoke_test.sh
```

M5 不使用 v2 Modbus 服务源码。未来接入真实 PLC 时必须重新定义应用寄存器契约、超时、重连和故障安全值。

## M6 业务 App Mock

`apps/` 在通用 Gateway 之上增加可独立测试的业务规则层，当前包含：

- `carton_tube_check`：纸筒类别、置信度、多目标、ROI、中心偏差和尺寸规则。
- `carton_partition_check`：隔板 cell 数量、置信度和 defect 类别规则。

业务 App 消费标准 `inference_result`，输出统一 `AppDecision`、`GatewayMessage` 和专用 Holding Registers。业务规则不进入 C++ Runtime，Collector Web 只展示状态与结果。M6 不连接真实设备，也不把 v2 的两个业务服务原样复制到 v3。
