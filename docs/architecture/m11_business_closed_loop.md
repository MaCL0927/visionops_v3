# M11 真实业务闭环与双 Runtime 拓扑

M11 的目标是把真实 Runtime/RKNN 输出接入 `carton_tube_check` 和 `carton_partition_check` 两个业务 App。业务 App 仍然只消费标准 `inference_result`，不直接接相机、RKNN、Web 或 PLC。

## 单业务链路

```text
HP60C Bridge / V4L2
  -> C++ Runtime
  -> Collector Web
  -> Business App
  -> AppDecision
  -> GatewayMessage
  -> Business Registers
```

业务 App 可通过 `--upstream-kind collector` 读取 Collector 的 `/api/runtime/latest_result`，也可以通过 `--upstream-kind runtime` 直接读取 Runtime。

## 同一块 3576 上同时运行两个任务

纸筒和隔板通常需要不同模型，因此推荐两套 Runtime + Collector + Business App 使用不同端口并行运行：

| 任务 | Runtime | Collector | Business App | App Modbus |
|---|---:|---:|---:|---:|
| carton_tube_check | 18081 | 8091 | 19110 | 1510 |
| carton_partition_check | 18082 | 8092 | 19120 | 1520 |

这样两个任务互不抢模型、互不覆盖 latest_result，也不会争用业务寄存器地址。

## 纸筒业务启动示例

```bash
MODEL_DIR=/opt/visionops_v3/models/tube_model
./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18181 \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 --port 18081 --device-id lb3576-dev

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 --port 8091 \
  --runtime-url http://127.0.0.1:18081 \
  --business-app-url http://127.0.0.1:19110 \
  --device-id lb3576-dev

python3 -m edge.gateway_adapter.apps.carton_tube_check.service \
  --host 0.0.0.0 --port 19110 \
  --upstream-kind collector --upstream-url http://127.0.0.1:8091 \
  --config configs/app/carton_tube_check.real.example.yaml \
  --device-id lb3576-dev
```

## 隔板业务启动示例

```bash
MODEL_DIR=/opt/visionops_v3/models/partition_model
./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18181 \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 --port 18082 --device-id lb3576-dev

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 --port 8092 \
  --runtime-url http://127.0.0.1:18082 \
  --business-app-url http://127.0.0.1:19120 \
  --device-id lb3576-dev

python3 -m edge.gateway_adapter.apps.carton_partition_check.service \
  --host 0.0.0.0 --port 19120 \
  --upstream-kind collector --upstream-url http://127.0.0.1:8092 \
  --config configs/app/carton_partition_check.real.example.yaml \
  --device-id lb3576-dev
```

## Collector Web 手动触发业务判断

M11 中 Collector 新增了业务 App 代理接口：

```text
POST /api/app/evaluate_once
GET  /api/app/latest_decision
GET  /api/app/latest_gateway_message
GET  /api/app/registers
```

生产模式页面中的“执行业务判断”按钮会调用 `/api/app/evaluate_once`，然后刷新业务决策和寄存器。
