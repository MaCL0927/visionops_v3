# carton_partition_check 业务 App

本目录实现纸箱隔板 / 蜂窝格结构检测业务层。M11 起它可以读取真实 Runtime/Collector 的 `inference_result`，统计正常 cell 与 defect，并输出 AppDecision、GatewayMessage 和业务寄存器。

业务规则只在 Gateway app 层运行，C++ Runtime 保持模型无关，Collector Web 不执行结构判断。

## 真实 Runtime 闭环

隔板模型通常需要独立 Runtime/Collector，避免和纸筒模型抢占同一个模型配置。示例：Runtime `18082`，Collector `8092`，业务 App `19120`：

```bash
python -m edge.gateway_adapter.apps.carton_partition_check.service \
  --host 0.0.0.0 \
  --port 19120 \
  --upstream-kind collector \
  --upstream-url http://127.0.0.1:8092 \
  --config configs/app/carton_partition_check.real.example.yaml \
  --device-id lb3576-dev \
  --poll-interval-ms 500
```

手动触发一次业务判断：

```bash
curl -X POST http://127.0.0.1:19120/api/app/evaluate_once | python3 -m json.tool
curl http://127.0.0.1:19120/api/app/latest_decision | python3 -m json.tool
curl http://127.0.0.1:19120/api/app/registers | python3 -m json.tool
```

## 规则说明

当前 M11 版本包含：

- cell / defect 类别过滤；
- 置信度阈值；
- `expected_cell_count` / `min_cell_count` / `max_cell_count`；
- 可选 `expected_rows` / `expected_cols` 轻量网格角度指标；
- defect 优先级高于数量判断。

v2 的模板校准、槽位编号、仿射/倾斜精细指标没有原样复制，后续可在本模块继续扩展模板文件和槽位匹配逻辑。

## Mock 测试

```bash
python -m edge.gateway_adapter.apps.carton_partition_check.service \
  --host 0.0.0.0 --port 19120 \
  --upstream-kind file --mock-case defect
```

默认寄存器为 `200..219`，可选 Modbus TCP 端口为 `1520`，默认不启动。
