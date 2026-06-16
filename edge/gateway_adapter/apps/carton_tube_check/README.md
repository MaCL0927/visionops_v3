# carton_tube_check 业务 App

本目录实现纸筒检测业务层。M11 起它既可以使用本地 mock case，也可以读取真实 Runtime/Collector 的 `inference_result`，输出：

```text
inference_result -> AppDecision -> GatewayMessage -> Holding Registers
```

业务规则位于 Gateway app 层，不进入 C++ Runtime；Collector Web 只代理状态、结果和手动触发，不直接做纸筒判断。

## 真实 Runtime 闭环

假设 v3 Runtime 已在 `18081` 输出真实 `tube` 检测结果，Collector Web 运行在 `8091`：

```bash
python -m edge.gateway_adapter.apps.carton_tube_check.service \
  --host 0.0.0.0 \
  --port 19110 \
  --upstream-kind collector \
  --upstream-url http://127.0.0.1:8091 \
  --config configs/app/carton_tube_check.real.example.yaml \
  --device-id lb3576-dev \
  --poll-interval-ms 500
```

手动触发一次业务判断：

```bash
curl -X POST http://127.0.0.1:19110/api/app/evaluate_once | python3 -m json.tool
curl http://127.0.0.1:19110/api/app/latest_decision | python3 -m json.tool
curl http://127.0.0.1:19110/api/app/registers | python3 -m json.tool
```

如果检测到 `tube` 且满足阈值、ROI、尺寸规则，输出 `final_code=0 / OK`；无目标输出 `final_code=2 / NO_TARGET`；多个目标是否允许由 `allow_multi_target` / `max_target_count` 控制。

## Mock 测试

```bash
python -m edge.gateway_adapter.apps.carton_tube_check.service \
  --host 0.0.0.0 --port 19110 \
  --upstream-kind file --mock-case ok
```

## 寄存器

默认业务寄存器为 `100..119`，可选 Modbus TCP 端口为 `1510`，只有显式传入 `--enable-modbus` 才启动。`offset_x_signed` 和 `offset_y_signed` 按 int16 补码编码到 uint16，例如 `-6` 编码为 `65530`。

本实现没有复制 v2 服务。v2 的触发、深度和现场寄存器经验只作为业务边界参考，v3 保持标准 AppDecision / GatewayMessage / registers 契约。
