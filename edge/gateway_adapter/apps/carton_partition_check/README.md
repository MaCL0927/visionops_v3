# carton_partition_check Mock

本目录是 M6 纸箱隔板/蜂窝格结构检测业务 Mock。它从标准 `inference_result` 统计 cell 与 defect，输出 AppDecision、GatewayMessage 和业务寄存器，不连接真实相机、模型、PLC 或 Modbus RTU。

业务判断只在 Gateway app 层运行，C++ Runtime 保持模型无关，Collector Web 不执行结构判断。真实 Runtime/Collector 只需继续提供 `/api/runtime/latest_result`，不需修改业务输出协议。

```bash
python -m edge.gateway_adapter.apps.carton_partition_check.service \
  --host 0.0.0.0 --port 19120 \
  --upstream-kind file --mock-case defect
```

默认寄存器为 `200..219`，可选 Modbus TCP Mock 端口为 `1520`，默认不启动。M6 只实现数量、置信度和 defect 类别规则；网格模板、角度、仿射和槽位匹配属于后续经真实数据验证后的业务扩展。不允许把 v2 服务原样复制进 v3。
