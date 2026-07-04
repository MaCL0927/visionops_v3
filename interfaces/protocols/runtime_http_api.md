# C++ Runtime HTTP API 契约

## 1. 目的与范围

本文定义 VisionOps v3 后续 C++ Runtime Mock 和生产 Runtime 共同遵守的最小 HTTP API。当前阶段只定义契约，不实现服务，也不接入真实相机、RKNN、Collector Web 或 Gateway。

Runtime 的生产职责是接收 Camera Bridge 帧、执行 C++ RKNN 推理并发布标准化结果。Collector Web 使用本 API 进行健康检查、状态展示、预览控制和单次验证；Gateway/Modbus 适配器主要消费 `inference_result`，不解析模型原始张量。

## 2. 通用约定

- API 版本：`v1`，消息体使用 `schema_version: "1.0"`。
- JSON 响应类型：`application/json; charset=utf-8`。
- 图片响应类型：`image/jpeg`。
- 时间：Unix epoch 毫秒，字段名为 `timestamp_ms`。
- 调用方可通过请求头 `X-Trace-Id` 传入 `trace_id`；未传时 Runtime 生成。
- 涉及具体图像帧的请求和响应必须包含 `frame_id`。
- 错误响应包含 `status: "error"` 和 `error`，其中含 `code`、`message`、`detail`、`recoverable`。
- Runtime 必须设置合理的请求体限制，不接受 Base64 图片作为 JSON 字段。
- `POST` 控制接口应具备幂等语义：重复开始或停止请求返回当前稳定状态，而不是制造额外会话。

通用错误示例：

```json
{
  "schema_version": "1.0",
  "message_type": "runtime_error",
  "device_id": "example-edge-001",
  "component": "rknn_runtime",
  "timestamp_ms": 1760000000999,
  "trace_id": "trace-request-0001",
  "source": "http_api",
  "status": "error",
  "error": {
    "code": "CAMERA_NOT_READY",
    "message": "相机当前不可用",
    "detail": null,
    "recoverable": true
  }
}
```

## 3. GET /health

### 用途

用于 systemd、部署工具和 Collector Web 判断 Runtime 进程是否存活以及核心依赖是否可用。该接口应快速返回，不执行推理。

### 请求参数

无查询参数、无请求体。`X-Trace-Id` 可选。不需要 `frame_id`。

### 成功响应

HTTP `200 OK`：

```json
{
  "schema_version": "1.0",
  "message_type": "runtime_health",
  "device_id": "example-edge-001",
  "component": "rknn_runtime",
  "timestamp_ms": 1760000001000,
  "trace_id": "trace-health-0001",
  "source": "runtime:primary",
  "status": "ok",
  "health": "ok",
  "ready": true,
  "version": "0.1.0"
}
```

### 错误状态

- HTTP `503 Service Unavailable`：进程存在，但模型、相机桥接或内部队列导致 Runtime 尚未就绪。
- HTTP `500 Internal Server Error`：无法生成健康状态。

### 调用关系

Collector Web 可每 5 到 10 秒轮询；部署工具在服务启动后调用。Gateway 不应以高频轮询该接口替代结果通道。

## 4. GET /api/runtime/status

### 用途

返回当前 Runtime 状态快照，响应结构遵守 `runtime_status.schema.json`。

### 请求参数

无请求体。`X-Trace-Id` 可选。不需要 `frame_id`。

### 成功响应

HTTP `200 OK`，示例见 `interfaces/examples/runtime_status.example.json`。主要包含运行模式、模型、相机连接、FPS、延迟、计数器和资源使用率。

### 错误状态

- HTTP `500 Internal Server Error`：状态聚合失败。
- HTTP `503 Service Unavailable`：Runtime 正在启动或维护，仍应尽可能返回 `health: "degraded"` 的状态结构。

### 调用关系

Collector Web 用于状态页和诊断页。Gateway 可低频读取状态用于设备健康上报，但业务结果应来自标准化推理结果。

## 5. POST /api/runtime/start_preview

### 用途

请求 Runtime 进入预览模式。预览模式允许持续接收帧并更新快照，但是否执行推理由请求参数决定。

### 请求参数

JSON 请求体：

```json
{
  "schema_version": "1.0",
  "message_type": "start_preview_request",
  "trace_id": "trace-preview-0001",
  "inference_enabled": false,
  "snapshot_fps": 2.0
}
```

- `trace_id`：必需，也可由 `X-Trace-Id` 提供。
- `inference_enabled`：可选，默认 `false`。
- `snapshot_fps`：可选，由 Runtime 限制在安全范围。
- 不需要 `frame_id`，因为该请求控制后续帧流。

### 成功响应

HTTP `200 OK`：

```json
{
  "schema_version": "1.0",
  "message_type": "runtime_command_result",
  "device_id": "example-edge-001",
  "component": "rknn_runtime",
  "timestamp_ms": 1760000001100,
  "trace_id": "trace-preview-0001",
  "source": "http_api",
  "status": "ok",
  "command": "start_preview",
  "mode": "preview",
  "accepted": true
}
```

### 错误状态

- HTTP `400 Bad Request`：参数格式或范围错误。
- HTTP `409 Conflict`：Runtime 处于 maintenance 等禁止预览的模式。
- HTTP `503 Service Unavailable`：Camera Bridge 不可用。

### 调用关系

Collector Web 在用户打开预览时调用。Gateway 不调用该接口。

## 6. POST /api/runtime/stop_preview

### 用途

停止预览快照更新，使 Runtime 返回 `idle` 或原先的生产检测模式。该操作不得卸载生产模型，除非后续配置明确要求。

### 请求参数

JSON 请求体：

```json
{
  "schema_version": "1.0",
  "message_type": "stop_preview_request",
  "trace_id": "trace-preview-0002"
}
```

需要 `trace_id`，不需要 `frame_id`。

### 成功响应

HTTP `200 OK`，返回 `message_type: "runtime_command_result"`、`command: "stop_preview"`、`accepted: true` 和最终 `mode`。

### 错误状态

- HTTP `400 Bad Request`：请求结构错误。
- HTTP `409 Conflict`：当前状态不允许切换。
- HTTP `500 Internal Server Error`：停止过程失败。

### 调用关系

Collector Web 在关闭预览或页面会话超时时调用。Gateway 不调用该接口。

## 7. POST /api/runtime/infer_once

### 用途

请求 Runtime 对 Camera Bridge 的下一可用帧或指定缓存帧执行一次推理，用于 Collector Web 单帧验证。请求不上传图片内容。

### 请求参数

JSON 请求体：

```json
{
  "schema_version": "1.0",
  "message_type": "infer_once_request",
  "trace_id": "trace-infer-0001",
  "frame_id": "frame-example-000001",
  "timeout_ms": 3000
}
```

- `trace_id`：必需。
- `frame_id`：可选；传入时指定 Runtime 已知的缓存帧，不传时使用下一可用帧。
- `timeout_ms`：可选，必须受服务端上限约束。
- 成功响应必须包含最终实际使用的 `frame_id`。

### 成功响应

HTTP `200 OK`，响应遵守 `inference_result.schema.json`。Detection 示例：

```json
{
  "schema_version": "1.0",
  "message_type": "inference_result",
  "device_id": "example-edge-001",
  "component": "rknn_runtime",
  "timestamp_ms": 1760000001200,
  "trace_id": "trace-infer-0001",
  "frame_id": "frame-example-000001",
  "source": "runtime:infer_once",
  "status": "ok",
  "result_id": "result-example-000001",
  "task_type": "detection",
  "model": {
    "model_id": "model-example",
    "model_name": "example-detector",
    "model_version": "1.0.0",
    "backend": "mock",
    "input_size": {"width": 640, "height": 640}
  },
  "image": {"width": 1920, "height": 1080},
  "timing": {
    "preprocess_ms": 2.0,
    "inference_ms": 12.0,
    "postprocess_ms": 2.0,
    "total_ms": 16.0
  },
  "detections": []
}
```

### 错误状态

- HTTP `400 Bad Request`：请求字段非法。
- HTTP `404 Not Found`：指定 `frame_id` 已不存在。
- HTTP `409 Conflict`：Runtime 正在切换模型或维护。
- HTTP `503 Service Unavailable`：相机或模型不可用。
- HTTP `504 Gateway Timeout`：在限制时间内未取得帧或结果。

### 调用关系

Collector Web 用于“拍照检测”或选中缓存帧验证。Gateway 默认不主动调用；如需拉取结果，应读取 latest result 或使用后续定义的推送通道。

## 8. POST /api/runtime/switch_model

### 用途

请求 Runtime 按标准模型包目录切换当前加载模型。Collector Web 负责扫描和校验模型包，Runtime 负责真正创建新的 Runner、加载 RKNN 并在成功后替换旧模型。

### 请求参数

JSON 请求体：

```json
{
  "model_dir": "/opt/visionops_v3/models/carton_tube_check"
}
```

- `model_dir`：必需，表示一个标准模型包目录。
- Runtime 需要在该目录内读取固定文件 `model.rknn` 和 `model.yaml`；`model.yaml` 是唯一模型元信息来源。
- 不需要 `frame_id`；`trace_id` 可由请求头传入。

### 成功响应

HTTP `200 OK`，返回最新 `runtime_status`，其中 `loaded_model` 必须已经切换为新模型。

### 错误状态

- HTTP `400 Bad Request`：请求体缺少 `model_dir` 或 JSON 非法。
- HTTP `409 Conflict`：Runtime 当前正处于不允许切换模型的状态。
- HTTP `500 Internal Server Error`：新模型加载失败。

错误时必须保留旧模型，不允许让 Runtime 进入无模型状态。

### 调用关系

浏览器不直接调用该接口。Collector Web 先通过 `/api/models` 扫描并校验模型包，再调用本接口。Business App、Gateway 和 Modbus 不调用该接口。

## 9. GET /api/runtime/latest_result

### 用途

返回最近一次成功或可诊断的标准化推理结果。

### 请求参数

可选查询参数：

- `after_result_id`：仅当最新结果不同于该 ID 时返回；M1 Mock 可先忽略长轮询能力。
- `X-Trace-Id`：可选。

不要求请求方提供 `frame_id`，但成功响应中的推理结果必须包含 `frame_id`。

### 成功响应

- HTTP `200 OK`：响应遵守 `inference_result.schema.json`。
- HTTP `204 No Content`：Runtime 尚无推理结果，或没有比 `after_result_id` 更新的结果。

### 错误状态

- HTTP `400 Bad Request`：查询参数非法。
- HTTP `500 Internal Server Error`：结果读取失败。

### 调用关系

Collector Web 可低频轮询用于结果展示。Gateway/Modbus Mock 可轮询该接口验证集成，但生产环境优先采用受控推送或本机消息通道，避免高频 HTTP 轮询。

## 10. GET /api/runtime/snapshot.jpg

### 用途

返回最近预览帧或最近推理帧的 JPEG 快照，供 Collector Web 展示。图片不进入任何 JSON 接口。

### 请求参数

可选查询参数：

- `frame_id`：请求指定缓存帧；省略时返回最新快照。
- `overlay`：`none` 或 `result`，默认 `result`。
- `quality`：JPEG 质量建议值，Runtime 可限制范围。
- `X-Trace-Id`：可选。

指定 `frame_id` 时，响应头必须回传同一值。

### 成功响应

HTTP `200 OK`，响应体为 JPEG 字节，至少包含以下响应头：

```text
Content-Type: image/jpeg
Cache-Control: no-store
X-Frame-Id: frame-example-000001
X-Trace-Id: trace-snapshot-0001
X-Timestamp-Ms: 1760000001300
```

### 错误状态

- HTTP `400 Bad Request`：查询参数非法。
- HTTP `404 Not Found`：指定帧不存在或当前没有快照。
- HTTP `503 Service Unavailable`：Camera Bridge 或预览链路不可用。

错误时返回 JSON 通用错误结构，不返回伪造图片。

### 调用关系

Collector Web 使用该接口显示预览和结果叠加图。Gateway/Modbus 不调用该接口，也不得将快照自动嵌入业务消息。

## 11. 并发、缓存与安全边界

- 控制接口与推理数据面必须隔离，慢速 Web 客户端不能阻塞相机和 NPU 线程。
- 最新结果与快照采用有界缓存；过期 `frame_id` 返回 `404`。
- Runtime 不接受任意文件路径、模型路径或 shell 命令作为本 API 参数。
- 模型切换、服务重启和生产参数变更不属于本文最小 API，后续需单独定义鉴权和审计契约。
- Mock 与生产实现必须返回相同字段结构，但 Mock 使用 `backend: "mock"` 明确标识。
