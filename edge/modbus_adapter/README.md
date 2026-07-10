# Modbus Adapter

该目录提供可复用的：

- 线程安全 Holding Register Bank；
- Modbus-TCP FC03 / FC06 / FC16 Server；
- 独立测试客户端。

通用适配层不提供默认寄存器表。每条生产线必须显式传入自己的 `RegisterDefinition`，避免不同任务共享隐式地址。

当前生产使用方：

```text
production/carton_line/gateway/register_bank.py
production/carton_line/gateway/service.py
```
