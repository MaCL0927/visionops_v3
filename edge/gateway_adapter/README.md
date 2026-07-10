# Gateway Adapter

该目录只保留可被不同产线复用的 Gateway 消息工具，不包含具体现场任务、端口、寄存器表或启动服务。

具体产线 Gateway 位于：

```text
production/<line_id>/gateway/
```

当前纸隔板/纸筒产线实现：

```text
production/carton_line/gateway/
```

业务算法、PLC 触发语义和寄存器映射不得重新放回通用 `edge/` 目录。
