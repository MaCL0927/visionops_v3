# VisionOps Collector Web

Collector Web 是边缘端的管理、展示和代理入口，当前已经用于 `3576` 真机联调，但它不是生产推理进程。

在当前主链路中：

```text
Camera Bridge / HP60C Bridge
  -> C++ RKNN Runtime
  -> Collector Web
  -> Business App / Gateway / Modbus
```

Collector Web 的职责是：

- 聚合 Collector / Runtime / Gateway / Business App 状态。
- 代理 Runtime 的 `status`、`infer_once`、`latest_result`、`snapshot.jpg`。
- 扫描 `models_root` 下的标准模型包目录，并通过 Runtime 触发模型切换。
- 提供边缘端 Web 页面：校验、采集上传、模型验证、设置、生产模式。
- 承载低频操作、状态展示和调试入口。

当前前端界面已经按工厂现场大屏 / 触屏使用习惯做过一轮放大和重排，视觉上参考了 `visionops_v2/edge/collector` 的卡片密度、按钮尺寸和页面分区方式，但底层接口、代理链路和模块边界保持 v3 设计，不回退到 v2 的旧后端逻辑。

Collector Web 明确不负责：

- 直接连接相机、读取 `/dev/videoX`、调用 HP60C SDK。
- 加载模型、调用 RKNN / NPU。
- 解析 RKNN 原始 tensor。
- 实现纸筒、隔板等业务判断。

## 与 Runtime / Gateway / Business App 的关系

- 浏览器只访问 Collector 同源接口。
- Collector 后端再去访问 Runtime / Gateway / Business App。
- 前端不直接访问 `18080 / 19090 / 19110` 这类下游端口。
- 生产推理仍然由 C++ Runtime 负责，业务判断由 Business App 负责。

## 启动

3576 现场常见启动方式：

```bash
source /opt/visionops/venv/bin/activate

python3 -m apps.collector_web.backend.main \
  --host 0.0.0.0 \
  --port 18091 \
  --runtime-url http://127.0.0.1:28081 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
  --models-root /opt/visionops_v3/models \
  --device-id lb3576-dev
```

本地开发示例：

```bash
python -m apps.collector_web.backend.main \
  --config configs/app/collector.example.yaml \
  --host 0.0.0.0 \
  --port 8090 \
  --runtime-url http://127.0.0.1:18080 \
  --gateway-url http://127.0.0.1:19090 \
  --business-app-url http://127.0.0.1:19110 \
  --models-root ./models \
  --device-id example-edge-001 \
  --component collector_web
```

`--models-root` 未显式传入时，Collector 会优先使用仓库根目录下的 `./models`，若不存在则回退为 `/opt/visionops_v3/models`。

浏览器访问：

```text
http://127.0.0.1:8090/
```

## 页面与预览行为

页面顶部保持旧版 VisionOps 使用习惯：

- 校验
- 采集上传
- 模型验证
- 设置
- 切换生产模式

其中快照与预览都来自 Runtime 代理，不直接访问相机。

当前前端在页面初始化后会自动调用：

```text
POST /api/runtime/start_preview
```

这样可以让 Runtime 进入 preview 状态，持续刷新 `snapshot.jpg`。如果该调用失败，页面不会阻塞，但实时预览会退化，需要检查 Runtime 与帧源状态。

当前前端还支持两个浏览器侧节流设置：

- `preview_refresh_interval_ms`：控制校验页、采集页对 `/api/runtime/snapshot.jpg` 的刷新间隔，默认 `200ms`
- `inference_interval_ms`：控制模型验证页“实时检测”循环触发 `/api/runtime/infer_once` 的间隔，默认 `500ms`

这两个设置通过设置页保存到浏览器 `localStorage`，刷新页面后仍会生效，但不会写入 Runtime 或 Collector 的后端配置文件。

## 模型扫描与切换

“模型验证”页面当前支持：

- 扫描 `models_root` 下的一级模型包目录
- 展示模型名称、版本、任务类型、平台、输入尺寸、类别数量和模型大小
- 标识当前 Runtime 正在使用的模型
- 点击切换到目标模型

Collector 本身不加载 `.rknn`。它只负责：

1. 扫描 `models_root`
2. 校验模型包是否为标准目录
3. 将选中的 `model_dir` 发送给 Runtime

真正的模型加载和替换由 C++ Runtime 完成。

当前标准模型包规则：

- 一个目录只表示一个模型包
- 必须包含 `model.rknn` 和 `model.yaml`
- `model.yaml` 是唯一元信息来源
- 不再读取 `manifest.json` / `labels.txt`
- 当前不自动识别同目录中的额外 `model2.rknn`

## Collector API

```text
GET  /health
GET  /api/collector/status
GET  /api/collector/config
POST /api/collector/config
GET  /api/runtime/status
POST /api/runtime/start_preview
POST /api/runtime/stop_preview
POST /api/runtime/infer_once
GET  /api/runtime/latest_result
GET  /api/runtime/snapshot.jpg
GET  /api/models
POST /api/models/switch
GET  /api/gateway/status
GET  /api/gateway/registers
GET  /api/app/status
GET  /api/app/registers
```

说明：

- `/health` 只表示 Collector 自身健康。
- `/api/collector/status` 会聚合下游状态；下游不可达时仍返回稳定 JSON。
- `snapshot.jpg` 只是 Runtime 快照代理，不是 Web 自己取图。
- `/api/models` 返回模型扫描结果和当前 Runtime `loaded_model`。
- `/api/models/switch` 只允许切换到 Collector 已扫描并验证通过的模型包，不能传任意绝对路径。

## 当前限制

- Collector 不做生产推理。
- Collector 不直接保存真实模型或现场私密配置。
- 设置页当前以代理配置与前端临时配置为主，不写 `.env`。
- M13 起，预览刷新与自动推理间隔会持久化到浏览器本地存储。
- 真实采集保存、采集包导出和上传仍是后续工作。

## 验证

```bash
python -m pytest tests/integration/test_collector_web_proxy.py
bash apps/collector_web/tests/smoke_test.sh
```

`Runtime / Gateway / Business App` 的真实联调仍需在 3576 真机上验证。

## M14 设置界面

Collector Web 的设置弹窗已拆分为“相机设置 / 视觉盒子设置 / 算法设置”三页。当前保存方式为浏览器 localStorage，用于前端刷新间隔和模型验证页可视化开关；真实后端配置保存接口后续接入。

算法设置页中的可视化开关会立即影响模型验证页 overlay，例如关闭 OBB 外接水平框、保留 OBB 旋转框，或控制 segmentation bbox / mask polygon 显示。

## M14.1 SDK Bridge 相机设置页

设置中心的相机页已改为 SDK Bridge 通用配置入口，面向 HP60C 与 Orbbec Gemini 336L 两类 SDK + HTTP Bridge。固定的服务 URL、快照路径和深度图路径不再作为用户可编辑项展示；页面只保留相机型号、画面帧率、RGB/Depth profile、JPEG 质量、翻转、RGB 顺序、深度单位等现场配置入口。当前所有设置仍保存到浏览器 localStorage，后续再接入后端配置 API 写入对应 bridge env 并重启服务。

## M14 Orbbec 336L 设置 API

Collector Web 提供 Orbbec 336L SDK Bridge 设置接口：

```bash
curl -s http://127.0.0.1:18091/api/settings/sdk_bridge/orbbec336l | python3 -m json.tool
```

保存设置示例：

```bash
curl -s -X POST http://127.0.0.1:18091/api/settings/sdk_bridge/orbbec336l \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_model":"orbbec336l",
    "rgb_profile":"orbbec:1280x720@30",
    "depth_profile":"orbbec:1280x720@30",
    "display_fps":10,
    "camera_jpeg_quality":95,
    "flip_vertical":"false",
    "flip_horizontal":"false",
    "depth_unit":"mm",
    "orbbec_serial":""
  }' | python3 -m json.tool
```

该接口会写入 Orbbec Bridge env 并重启 `visionops-orbbec336l-bridge.service`。如果 Collector 不是 root 运行，需要配置受限 sudo 权限。
