# Modbus TCP Mock

`edge/modbus_adapter/` 提供 M5 阶段的内存 Holding Register Bank、最小 Modbus TCP Server 和测试客户端。该实现不连接真实 PLC，不支持 Modbus RTU，不用于生产现场。

## 协议范围

- 默认监听 `0.0.0.0:1502`，避免使用需特权的标准端口 `502`。
- 支持 FC03 Read Holding Registers。
- 支持 FC06 Write Single Register。
- 支持 FC16 Write Multiple Registers。
- 接受任意 Unit ID，数据只存在于进程内存。
- 非法 function code、地址或参数返回 Modbus exception。

寄存器值严格限制为 `0..65535`。浮点业务值必须根据 register definition 的 `scale` 转换为整数后写入，不直接把浮点数放入单个 Holding Register。

## 测试客户端

```bash
python -m edge.modbus_adapter.modbus_test_client \
  --host 127.0.0.1 \
  --port 1502 \
  --read-start 0 \
  --read-count 20 \
  --print-registers
```

默认 register map 只表达心跳、序号、决策、几何、耗时和 ID 摘要，不传输图片或大块 JSON。专用现场映射应由 Gateway app 层以独立契约扩展。
