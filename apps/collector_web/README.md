# VisionOps Collector Web

Collector Web 是边缘设备上的轻量管理入口。M4 仅实现标准库 Python 后端、极简静态页面和 C++ Runtime Mock 的 HTTP 代理，不接入真实相机、RKNN、NPU、模型或现场通信设备。

## 职责边界

Collector Web 负责：

- 展示 Collector 自身与 Runtime 状态。
- 低频调用 Runtime 预览、单次推理和最新结果接口。
- 代理 Runtime JPEG 快照。
- 后续承载配置、诊断与受控服务操作。

Collector Web 明确不负责：

- 加载模型或解析模型输出张量。
- 调用 RKNN、NPU 或实现任何推理逻辑。
- 直接连接相机、HP60C SDK、V4L2 或 RTSP。
- 成为 Gateway/Modbus 的生产实时数据通道。
- 让浏览器直接访问 Runtime 端口。

生产推理始终属于 C++ Runtime。当前 Runtime Mock 是后续真实 RKNN Runtime 的接口替身，只用于开发与契约验证。

## 启动

先启动 Runtime Mock：

```bash
cmake -S . -B build
cmake --build build -j4
./build/edge/runtime_cpp/visionops_runtime_mock \
  --host 127.0.0.1 \
  --port 18080
```

再从仓库根目录启动 Collector：

```bash
python -m apps.collector_web.backend.main \
  --host 0.0.0.0 \
  --port 8090 \
  --runtime-url http://127.0.0.1:18080 \
  --device-id example-edge-001 \
  --component collector_web
```

浏览器访问：

```text
http://127.0.0.1:8090/
```

命令行参数：

```text
--host
--port
--runtime-url
--device-id
--component
--help
```

## Collector API

```text
GET  /health
GET  /api/collector/status
GET  /api/runtime/status
POST /api/runtime/start_preview
POST /api/runtime/stop_preview
POST /api/runtime/infer_once
GET  /api/runtime/latest_result
GET  /api/runtime/snapshot.jpg
```

`/health` 只表示 Collector 自身健康，不等同于 Runtime 健康。

`/api/collector/status` 聚合 Collector 和 Runtime 状态。Runtime 不可达时该接口仍返回 HTTP 200，并使用 `runtime.health: "unreachable"` 表达依赖故障。

Runtime 代理接口保留上游 HTTP 状态码。例如尚无最新结果时，Runtime 的 404 会原样通过 Collector 返回。JSON 响应保持 `application/json`，快照保持 `image/jpeg`。

前端只访问上述 Collector 同源接口，不包含 Runtime 地址。

## 实现说明

- 服务端使用 `ThreadingHTTPServer`。
- Runtime 客户端使用 `urllib.request`。
- 不新增第三方 Python 依赖。
- Runtime 请求超时默认为 2 秒。
- 请求体限制为 1 MiB，Runtime 响应限制为 4 MiB。
- 静态文件只从仓库内固定路径提供，不接受用户指定文件路径。

## 测试

```bash
python -m pytest tests/integration/test_collector_web_proxy.py
bash apps/collector_web/tests/smoke_test.sh
```

冒烟测试使用本机 `127.0.0.1:18080` 和 `127.0.0.1:8090`，自动构建、启动、调用并停止 Runtime Mock 与 Collector。
