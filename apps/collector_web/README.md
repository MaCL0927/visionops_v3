# VisionOps Collector Web

Collector Web 是边缘设备上的轻量管理入口。M7 按 v3 进程边界重建了采集上传、模型验证和生产运行三个核心页面，开始承载实际 Web 功能，但仍不越过 Collector 后端直接操作相机、模型或业务规则。

## 职责边界

Collector Web 负责：

- 展示 Collector 自身与 Runtime 状态。
- 低频调用 Runtime 预览、单次推理和最新结果接口。
- 代理 Runtime JPEG 快照。
- 聚合 Gateway 和 Business App 状态与寄存器快照。
- 后续承载配置、诊断与受控服务操作。

Collector Web 明确不负责：

- 加载模型或解析模型输出张量。
- 调用 RKNN、NPU 或实现任何推理逻辑。
- 直接连接相机、HP60C SDK、V4L2 或 RTSP。
- 成为 Gateway/Modbus 的生产实时数据通道。
- 让浏览器直接访问 Runtime 端口。
- 在 Collector Web 中实现纸筒、隔板或其他业务决策。

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
  --config configs/app/collector.example.yaml \
  --host 0.0.0.0 \
  --port 8090 \
  --runtime-url http://127.0.0.1:18080 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
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
--gateway-url
--business-app-url
--snapshot-refresh-interval-ms
--status-refresh-interval-ms
--config
--device-id
--component
--help
```

## Collector API

```text
GET  /health
GET  /api/collector/status
GET  /api/collector/config
GET  /api/runtime/status
POST /api/runtime/start_preview
POST /api/runtime/stop_preview
POST /api/runtime/infer_once
GET  /api/runtime/latest_result
GET  /api/runtime/snapshot.jpg
GET  /api/gateway/status
GET  /api/gateway/registers
GET  /api/app/status
GET  /api/app/registers
```

`/health` 只表示 Collector 自身健康，不等同于 Runtime 健康。

`/api/collector/status` 聚合 Collector 和 Runtime 状态。Runtime 不可达时该接口仍返回 HTTP 200，并使用 `runtime.health: "unreachable"` 表达依赖故障。

Runtime 代理接口保留上游 HTTP 状态码。例如尚无最新结果时，Runtime 的 404 会原样通过 Collector 返回。JSON 响应保持 `application/json`，快照保持 `image/jpeg`。

Gateway 和 Business App 的 status 接口在下游不可达时返回稳定 `unreachable` JSON，使生产页仍可诊断 Collector 自身。

前端只访问上述 Collector 同源接口，不包含 Runtime、Gateway 或 Business App 的直连地址。

## 三个核心页面

- **Capture**：显示、刷新和本地下载 Runtime JPEG 快照；采集包导出按钮仅作为后续入口。
- **Validate**：调用 `infer_once`，显示标准结果和四段耗时，在快照上绘制 detection bbox 与 OBB points。
- **Production**：聚合 Collector、Runtime、Gateway、Business App 状态和寄存器快照。

前端使用原生 ES modules，按 `api`、`state`、`pages`、`render` 拆分。v2 的顶部页签和工作区布局仅作为功能参考，未复制其巨型 `app.js`。

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
