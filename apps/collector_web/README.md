# VisionOps Collector Web

> 总体项目入口请优先阅读仓库根目录 `README.md`。本文档只补充 Collector Web 的局部职责、接口和设置说明。

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
source /opt/visionops_v3/venv/bin/activate

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

当前前端保留预览刷新节流，并使用一个统一的精确推理 FPS：

- `preview_refresh_interval_ms`：控制校验页、采集页快照刷新；
- `production_inference_fps`：设置页中的“生产 / 验证推理 FPS（统一）”。对于支持后台生产者的 App，该值会原样提交到 `/api/app/inference_settings`；模型验证定时器也由同一精确 FPS 换算。

`inference_interval_ms` 仅是浏览器定时器的派生值，不再作为生产推理 FPS 的独立数据源。旧 localStorage 只有 `inference_interval_ms` 时会一次性迁移为精确 FPS。

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
- 浏览器侧刷新间隔、部分页面状态仍保存在 `localStorage`。
- Orbbec Bridge 设置会写入对应 bridge env 并触发服务重启。
- 视觉盒子设置会写入 `/opt/visionops_v3/config/vision_box_settings.json`。
- 算法阈值会写回当前模型目录下的 `model.yaml`。
- M32.1 已完成边缘端采集 ROI；服务端继承 ROI 元数据和 Runtime 输入 ROI 仍属于后续阶段。

## 验证

```bash
python -m pytest tests/integration/test_collector_web_proxy.py
bash apps/collector_web/tests/smoke_test.sh
```

`Runtime / Gateway / Business App` 的真实联调仍需在 3576 真机上验证。

## 设置界面

Collector Web 的设置弹窗已拆分为“相机设置 / 视觉盒子设置 / 算法设置”三页。当前保存方式为浏览器 localStorage，用于前端刷新间隔和模型验证页可视化开关；真实后端配置保存接口后续接入。

算法设置页中的可视化开关会立即影响模型验证页 overlay，例如关闭 OBB 外接水平框、保留 OBB 旋转框，或控制 segmentation bbox / mask polygon 显示。

## SDK Bridge 相机设置页

设置中心的相机页已改为 SDK Bridge 通用配置入口，面向 HP60C 与 Orbbec Gemini 336L 两类 SDK + HTTP Bridge。固定的服务 URL、快照路径和深度图路径不再作为用户可编辑项展示；页面只保留相机型号、画面帧率、RGB/Depth profile、JPEG 质量、翻转、RGB 顺序、深度单位等现场配置入口。当前所有设置仍保存到浏览器 localStorage，后续再接入后端配置 API 写入对应 bridge env 并重启服务。

## Orbbec 336L 设置 API

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

## 视觉盒子设置

Collector Web 增加视觉盒子设置 API：

```bash
curl -s http://127.0.0.1:18091/api/settings/vision_box | python3 -m json.tool
```

配置默认保存到 `/opt/visionops_v3/config/vision_box_settings.json`，可通过 `VISIONOPS_VISION_BOX_SETTINGS_FILE` 覆盖。Runtime/Gateway/Business App URL、Device ID、目录和端口为启动参数/部署固定值，页面只展示；可写字段包括默认启动模式、状态刷新 FPS、磁盘告警阈值和服务端上传配置。

## 采集上传 API

Collector Web 提供边缘端数据集采集、打包与上传接口：

- `GET /api/dataset/capture_roi`：读取采集 ROI、当前采集图片数量和批次锁定状态。
- `POST /api/dataset/capture_roi`：保存或关闭采集 ROI。
- `POST /api/dataset/images/capture`：保存当前 Runtime 快照到 `/opt/visionops_v3/data/images`；启用采集 ROI 时，后端先裁剪再保存。
- `GET /api/dataset/images?offset=0&limit=24`：分页读取图片。
- `GET /api/dataset/images/<filename>/content`：预览图片。
- `DELETE /api/dataset/images/<filename>`：删除图片。
- `POST /api/dataset/packages/create`：打包为 `/opt/visionops_v3/data/upload_packages/*.tar.gz`。
- `POST /api/dataset/upload`：使用视觉盒子设置里的 SSH 配置打包并上传。

使用 SSH 密码上传时，运行环境需要安装 `paramiko` 或系统 `sshpass`。

### M32.1 采集 ROI

采集上传页新增蓝色“采集 ROI”。它只影响采集链路，与模型验证页已有的黄色“结果 ROI”相互独立：

```text
采集 ROI：手动拍照、定时采图、本地下载和上传包中的图片会被实际裁剪
结果 ROI：模型仍输入完整图，只在 Runtime 后处理阶段过滤检测结果
```

采集 ROI 默认保存到：

```text
/opt/visionops_v3/data/capture_roi.json
```

可以通过环境变量覆盖：

```bash
export VISIONOPS_CAPTURE_ROI_CONFIG_FILE=/custom/path/capture_roi.json
```

读取状态：

```bash
curl -s http://127.0.0.1:18091/api/dataset/capture_roi \
  | python3 -m json.tool
```

启用 ROI 示例：

```bash
curl -s -X POST http://127.0.0.1:18091/api/dataset/capture_roi \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled": true,
    "source_resolution": {"width": 1280, "height": 720},
    "normalized_xyxy": [0.25, 0.20, 0.75, 0.80]
  }' | python3 -m json.tool
```

关闭 ROI：

```bash
curl -s -X POST http://127.0.0.1:18091/api/dataset/capture_roi \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled": false,
    "source_resolution": {"width": 1280, "height": 720}
  }' | python3 -m json.tool
```

为避免同一个数据包混入不同裁剪区域，只要采集目录中仍有图片，系统就锁定当前 ROI。此时修改或关闭 ROI 会返回：

```text
HTTP 409 / CAPTURE_ROI_BATCH_LOCKED
```

Web 页面会让用户确认是否清空现有图片。API 调用方也可以明确提交：

```json
{
  "enabled": true,
  "source_resolution": {"width": 1280, "height": 720},
  "normalized_xyxy": [0.20, 0.15, 0.80, 0.85],
  "clear_existing_images": true
}
```

该操作会删除当前采集目录内的全部图片，再应用新 ROI。

启用 ROI 后，手动拍照与定时采图都调用同一个后端保存函数。后端只在采集时对 Runtime JPEG/PNG 快照执行一次解码、裁剪和重新编码，不改变生产推理链路。

每个采集压缩包根目录会同时包含：

```text
manifest.json
capture_manifest.json
images/
```

`capture_manifest.json` 示例：

```json
{
  "schema_version": "1.0",
  "message_type": "capture_manifest",
  "images_are_cropped": true,
  "coordinate_space": "runtime_snapshot",
  "capture_roi": {
    "enabled": true,
    "source_resolution": {"width": 1280, "height": 720},
    "normalized_xyxy": [0.25, 0.20, 0.75, 0.80],
    "pixel_xyxy": [320, 144, 960, 576],
    "crop_resolution": {"width": 640, "height": 432}
  },
  "image_count": 100
}
```

M32.1 只完成边缘端采集链路。服务端接收后继续传递 ROI 到训练任务和模型 `model.yaml`，以及 Runtime 使用 RGA 将 ROI 实际作为模型输入，属于后续阶段。

## 定时采图

“采集上传 / 拍照采集”页面提供“定时采图”按钮。设置间隔后，Collector 在后台周期性读取 Runtime 的 `/api/runtime/snapshot.jpg`，并使用与手动拍照相同的保存目录和命名规则。

接口：

```text
GET  /api/dataset/timed_capture
POST /api/dataset/timed_capture
```

启动示例：

```bash
curl -s -X POST http://127.0.0.1:18093/api/dataset/timed_capture \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"interval_seconds":10}' | python3 -m json.tool
```

停止示例：

```bash
curl -s -X POST http://127.0.0.1:18093/api/dataset/timed_capture \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false}' | python3 -m json.tool
```

定时任务属于当前 Collector 进程；Collector 重启后默认停止，需要重新启用。

## 检测结果 ROI

模型验证页提供“绘制 ROI”按钮。ROI 使用归一化坐标保存到对应 Runtime 的独立配置文件。模型输入仍为完整相机画面，ROI 只在 C++ Runtime 完成 detection / OBB / segmentation 后处理后过滤输出，判断规则为目标中心点位于 ROI 内。

Collector 代理接口：

```text
GET  /api/runtime/roi
POST /api/runtime/roi
```

因此 Collector Web、生产 Gateway、Modbus 和 Tube Pick TCP 读取到的都是同一份 ROI 过滤后的 Runtime 结果。

## 第一阶段生产 FPS 调度

浏览器定时器最低间隔由固定 100 ms 改为 16 ms，因此 15/20/30 FPS 选项不再被
静默压回 10 FPS。模型验证、生产画面、快照和状态循环均采用“工作耗时 + 剩余等待
= 目标周期”的调度方式。

当 `production_inference_source: app` 时，生产页面优先读取：

```text
GET /api/app/latest_decision
```

浏览器作为观察者复用后台生产结果，不再和机器人后台线程同时调用
`evaluate_once`、竞争同一个 Runtime/NPU。没有后台生产者的旧业务应用会自动回退到
`POST /api/app/evaluate_once`；“手动检测”仍会强制执行一次推理。

支持后台连续推理的业务应用可实现：

```text
GET  /api/app/inference_settings
POST /api/app/inference_settings  {"detection_fps":10}
```

算法设置中的“生产 / 验证推理 FPS（统一）”会把用户填写的数值直接提交给后台生产者，
不会先换算成整数毫秒再反算，因此输入 `15` 时 App 收到的就是 `15.0`，不会变成
`14.925...`。生产画面显示 App 返回的 `configured_fps` 与 `actual_fps`，不再把浏览器
轮询频率误标为生产推理设定。WebSocket 没有独立推送 Hz，每个完成的生产结果都会立即推送。

## M32.4 输入 ROI 可视化

模型验证页和生产模式会在完整快照上绘制青色 `input_roi`，并保留黄色结果过滤 ROI。两页均可打开“模型输入预览”，在浏览器端复现 crop + letterbox，同时展示 RGA、ROI 解析、crop_resize 和 NPU 推理耗时。该功能不触发额外相机取图或 Runtime 推理。完整说明见 `docs/M32.4_WEB_INPUT_ROI_VISUALIZATION_PHASE4.md`。

## M32.4.1 model.yaml 设置保存修复

算法设置不再通过 PyYAML 重写整个 `model.yaml`，而是只原位修改
`postprocess` 中的阈值标量，保持输入尺寸、input ROI 内联数组、版本字符串、
注释和字段顺序不变。classification、detection、segmentation、OBB 四类任务均已覆盖。
Runtime 同时兼容历史 PyYAML 生成的 `pixel_xyxy` / `normalized_xyxy` 块列表。
完整说明见 `docs/M32.4.1_MODEL_YAML_SETTINGS_FORMAT_FIX.md`。

## M32.4.2 通用 Runtime 推理 FPS 解耦

使用 `scripts/start_runtime.sh + scripts/start_collector.sh` 时，旧实时循环会串行等待
`infer_once`、快照 JPEG、浏览器解码和 Canvas 绘制，导致 33 ms 的模型也可能只有约
5～6 FPS。M32.4.2 将推理循环与画面刷新循环分离：推理由
`production_inference_fps` 控制，画面由 `display_fps` 控制，二者互不等待。

生产页面显示“实际推理 FPS / 设定推理 FPS · 画面 FPS”；模型验证页也分别显示推理和
画面 FPS。通用 Runtime 模式的统一推理 FPS 会持久化到
`config/vision_box_settings.json`，不再只依赖浏览器 localStorage。完整说明见
`docs/M32.4.2_GENERIC_RUNTIME_FPS_DECOUPLING.md`。

## M32.8 同步 RGB-D 采集

拍照采集页新增“同步保存深度”开关，默认关闭。开启后：

- 手动拍照以及新启动的定时采图保存一组同名记录；
- RGB：`data/images/<capture_id>.jpg`；
- 深度：`data/depth/<capture_id>.png`，16 位 `16UC1`，单位毫米；
- 元数据：`data/meta/<capture_id>.json`，包含帧时间戳、RGB/Depth sequence、相机内参、ROI、有效深度比例和同步方式；
- Orbbec Bridge 通过 POSIX shared memory 发布同一 SDK FrameSet 的 RGB 与 D2C 深度。Collector 仅接受 `timestamp_epoch_ms` 完全相同且 sequence 读取稳定的帧对；无法验证同步时本次采集失败，不会退化为相邻帧拼接；
- 启用采集 ROI 后，RGB 与深度按相同 normalized ROI 裁剪，meta 内同时保存原始内参与裁剪后的有效内参；
- 删除 RGB 记录时会同步删除对应 depth/meta；
- `tar.gz` 包含 `images/`、`depth/`、`meta/`、`manifest.json` 与 `capture_manifest.json`。仅 RGB 的旧数据仍保持兼容。

环境变量可覆盖共享内存路径：

```bash
VISIONOPS_CAPTURE_SHARED_RGB_PATH=/dev/shm/visionops_orbbec336l_rgb
VISIONOPS_CAPTURE_SHARED_DEPTH_PATH=/dev/shm/visionops_orbbec336l_depth
```
