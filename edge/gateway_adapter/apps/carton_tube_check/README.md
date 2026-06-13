# carton_tube_check Mock

本目录是 M6 纸筒检测业务 Mock，只验证标准 `inference_result -> AppDecision -> GatewayMessage -> Holding Registers` 链路，不连接真实相机、RKNN/NPU、深度图或 PLC。

业务规则位于 Gateway app 层，不进入 C++ Runtime；Collector Web 只展示状态和结果，不执行纸筒判断。未来接真实业务时，将 `--upstream-kind` 从 `file` 切换为 `collector` 或 `runtime` 即可，不改 AppDecision 和寄存器契约。

```bash
python -m edge.gateway_adapter.apps.carton_tube_check.service \
  --host 0.0.0.0 --port 19110 \
  --upstream-kind file --mock-case ok
```

默认寄存器为 `100..119`，可选 Modbus TCP Mock 端口为 `1510`，只有显式传入 `--enable-modbus` 才启动。`offset_x_signed` 和 `offset_y_signed` 按 int16 补码编码到 uint16，例如 `-6` 编码为 `65530`。

本实现未复制 v2 `carton_tube_check` 服务。v2 的深度、触发和现场寄存器经验需在真实需求确认后逐项重新建模。
