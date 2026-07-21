# Carton Palletizing（独立纸箱托盘多层堆垛任务）

该目录是一条独立现场方案，不属于 `production/carton_line/`。当前版本实现第一层 OBB 占位检测，以及第 2、3、4 层和任意更多层的 RGB-D 堆垛状态机。

## 工作流程

1. 使用 OBB 模型检测并锁定托盘，类别固定为 `0=box`、`1=tray`；
2. 以托盘短边构建居中的正方形垛型区域，生成 P1～P4 四块横竖交错摆放掩膜；
3. 第一层根据纸箱 OBB 与 slot 的多边形 IoU、中心距离和朝向确认占位；
4. 第一层放满后采集多帧稳定的 D2C 对齐深度图，作为第二层基准；
5. 第二层及以上逐 slot 计算 `上一层基准深度 - 当前深度`；高度差和覆盖率达到阈值后，确认该位置新增一箱并隐藏掩膜；
6. 当前层放满后重新采集深度基准，自动进入下一层；
7. 奇数层使用模板 A，偶数层使用横竖错开的模板 B；第 3 层重新使用 A，依次交替；
8. `max_layers` 默认为 4，可改成任意正整数；设为 `0` 时支持持续增加层数，直到人工 reset。

## 模型和相机要求

模型必须是 OBB 模型：

```text
0 = box（纸箱）
1 = tray（托盘）
```

模型目录：

```text
/opt/visionops_v3/models/carton_palletizing/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 应声明：

```yaml
task: obb
class_names: [box, tray]
```

第 2 层及以上依赖 Orbbec 336L Bridge 的 D2C 对齐深度：

```text
http://127.0.0.1:18182/stream/depth.png
```

深度 PNG 必须是 `uint16` 毫米值，并与 RGB 画面空间对齐。

## 多层参数

主要配置位于 `production/carton_palletizing/config/line.yaml`：

> ROI 仅由 VisionOps Web/Runtime 配置控制；机器人 `config.detect_region` 会被忽略，机器人不能修改 ROI。

```yaml
task:
  algorithm:
    layering:
      max_layers: 4                  # 任意正整数；0=不限层数
      auto_advance: true
      baseline_capture_frames: 3
      baseline_settle_frames: 5
      baseline_stability_mm: 15.0
      next_layer_geometry: layer_template
      use_previous_detected_boxes: false

    template:
      layer_strategy: odd_even
      default_template: odd
      slot_order: [P3, P1, P2, P4]
      templates:
        odd:   # A：第1/3/5...层
          template_id: A
          slots: [...]
        even:  # B：第2/4/6...层
          template_id: B
          slots: [...]

    depth:
      min_depth_mm: 100
      max_depth_mm: 5000
      slot_roi_shrink_ratio: 0.12
      min_valid_ratio: 0.45
      baseline_min_valid_ratio: 0.55
      min_height_delta_mm: 80.0
      max_height_delta_mm: 600.0
      min_coverage_ratio: 0.55
      height_percentile: 50.0
      occupied_confirm_frames: 3
      occupied_stability_mm: 20.0
```

`min_height_delta_mm=80` 和 `max_height_delta_mm=600` 只是初始值。现场应测量一层纸箱顶面相对上一层的实际深度减小量，再设定合理范围。

## 状态

主要状态包括：

```text
WAIT_TRAY
LAYER_N_FILLING
LAYER_N_WAIT_DEPTH
LAYER_N_CAPTURING_BASELINE
LAYER_N_COMPLETE
STACK_COMPLETE
```

输出同时提供：

- `layer`：当前层；
- `max_layers`：配置层数，0 表示不限；
- `completed_layers`：已经完成的层；
- `layer_complete`：当前层是否完成；
- `stack_complete`：整垛是否达到最大层数；
- `next_slot_id` / `next_slot_key`：下一位置；
- 每个 slot 的 `depth.height_delta_mm`、`coverage_ratio` 和 `valid_ratio`。

## 手动启动

```bash
cd /opt/visionops_v3
./production/carton_palletizing/scripts/start_runtime.sh
./production/carton_palletizing/scripts/start_app.sh
./production/carton_palletizing/scripts/start_collector.sh
```

默认端口：

| 服务 | 地址 |
|---|---|
| RKNN Runtime | `127.0.0.1:28084` |
| 多层堆垛业务应用 | `127.0.0.1:19210` |
| Robot WebSocket | `0.0.0.0:9001/vision` |
| Collector Web | `0.0.0.0:18094` |
| 336L MJPEG | `:18182/stream.mjpeg` |

机器人使用 trigger 模式：M29.2 中 `pallet_place_target`（兼容任务号 `1`）返回托盘或当前最上层 1～4 个纸箱的实测 OBB 位姿；`held_box_pose`（兼容任务号 `2`）返回手持纸箱 OBB 中心、相机坐标和角度。数字、数字字符串和原任务名均可触发，响应 `trigger_task_id` 原样回显。初期/后期只需切换 `task.communication.held_box_selection.mode=nearest_depth|outside_tray`，详细协议见 `tasks/first_layer_placement/PROTOCOL.md`。

该 trigger 任务默认设置 `websocket.status_enabled: false`，不会在模拟机器人终端持续刷
`status`。需要状态心跳时再启用；模拟客户端使用 `--show-status` 才显示状态消息。

生产界面只显示当前层未占用的掩膜；绿色是当前建议位置，黄色是后续空位。层完成时显示深度基准采集进度，达到 `max_layers` 后显示“堆垛完成”。

## 调试接口

```bash
curl http://127.0.0.1:28084/api/runtime/status | python3 -m json.tool
curl http://127.0.0.1:19210/health | python3 -m json.tool

curl -X POST http://127.0.0.1:19210/api/app/evaluate_once \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool

curl http://127.0.0.1:19210/api/app/latest_decision | python3 -m json.tool

# 更换托盘或开始新一垛时，清空全部层数、托盘锁定、深度基准和占位状态
curl -X POST http://127.0.0.1:19210/api/app/reset \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
```

## 开机自启

```bash
cd /opt/visionops_v3
sudo bash production/carton_palletizing/deploy/install_services.sh
```

已有 `/etc/visionops_v3/carton_palletizing.yaml` 时，安装脚本不会覆盖它。需要手动合并新的
`layering.next_layer_geometry`、`template.templates.odd/even` 和
`communication.websocket.status_enabled/status_on_connect` 配置。

---

## 机器人眼睛视角纸箱抓取点任务

M25.3 新增第二个独立子任务：

```text
production/carton_palletizing/tasks/box_grasp_vision/
```

该任务使用 336L 倾斜俯视 RGB-D 画面和 segmentation 模型，不使用 OBB 近似箱体边缘。它从 mask 外轮廓计算透视四边形，固定输出：

- 左上、右上、右下、左下四角；
- 箱体中心；
- 左右两条边的中点（机器人抓取点）；
- 以上七个点的像素坐标、深度和相机三维坐标；
- 原始外轮廓点。

模型目录：

```text
/opt/visionops_v3/models/carton_box_grasp/current/
├── model.rknn
└── model.yaml
```

手动启动：

```bash
./production/carton_palletizing/scripts/start_box_grasp_runtime.sh
./production/carton_palletizing/scripts/start_box_grasp_app.sh
./production/carton_palletizing/scripts/start_box_grasp_collector.sh
```

默认接口：

| 服务 | 地址 |
|---|---|
| Segmentation Runtime | `127.0.0.1:28085` |
| HTTP App | `127.0.0.1:19211` |
| Robot WebSocket | `0.0.0.0:9001/vision` |
| Collector Web | `0.0.0.0:18095` |
| 336L MJPEG | `:18182/stream.mjpeg` |

安装独立 systemd 服务：

```bash
sudo bash production/carton_palletizing/deploy/install_box_grasp_services.sh
```

机器人报文中 `items[]` 的每一项表示一个抓取点。每个纸箱输出两个同 ID 的抓取点，字段统一为 `id/class_id/confidence/position_camera/center_px`，与 `tube_pick_vision` 的单抓取点结构一致。

机器人协议与完整字段见：

```text
production/carton_palletizing/tasks/box_grasp_vision/PROTOCOL.md
```

### Box grasp 第一阶段 FPS 优化

`box_grasp_vision` 的后台 worker 是生产模式唯一推理生产者。机器人 WebSocket 和
Collector 生产页面都读取 `latest_decision`，避免两条链路重复提交 NPU 推理。

后台频率接口：

```bash
curl -s http://127.0.0.1:19211/api/app/inference_settings | python3 -m json.tool

curl -s -X POST http://127.0.0.1:19211/api/app/inference_settings \
  -H 'Content-Type: application/json' \
  -d '{"detection_fps":10}' | python3 -m json.tool
```

设置以 schema `2.0` 持久化到 `/opt/visionops_v3/config/box_grasp_inference_settings.json`，
字段为 `production_inference_fps`。`box_grasp.websocket` 不再配置 `detection_hz`：后台生产者
只有一个目标 FPS，WebSocket 和 Collector 都消费同一结果流，并对每个完成结果立即推送。
旧 schema `1.0` 的 5 Hz 文件会被忽略，Web 启动时会用统一推理 FPS 完成一次迁移。

默认关闭 `debug.save_every_trigger`，避免每次推理创建保存线程并写入 RGB、Depth、JSON 和
Overlay。需要现场取证时再临时开启。

该任务启动 Runtime 时会使用：

```text
--max-detections 1
--mask-max-points 64
```

它只影响输出候选和 mask polygon 后处理，不改变模型输入或 NPU 网络结构。

### Box grasp 第三阶段 FPS 优化

Orbbec 336L Bridge 会通过 `/visionops_orbbec336l_rgb` POSIX 共享内存发布 RGB888
双缓冲帧，box-grasp Runtime 默认使用 `shared_memory` 帧源，删除逐帧 JPEG HTTP
下载和 OpenCV 解码。RKNN Runner 同时复用输入引用和预分配输出 buffer，降低
segmentation 大输出带来的动态分配、复制和时延抖动。

兼容策略：共享内存异常时可回退 `/stream/snapshot.jpg`；旧 RKNN driver 不支持预分配
float 输出时自动回退动态 `rknn_outputs_get/release`。详细验证方式见
`tasks/box_grasp_vision/README.md`。

### Box grasp 优化6：本地 Raw HTTP 与深度共享内存

`box_grasp_vision` 对 localhost Runtime/Bridge 请求默认使用轻量 socket HTTP client，
直接按 `Content-Length` 读取响应，失败时自动回退 urllib。App 状态会分别记录
`runtime_connect_ms`、`runtime_send_ms`、`runtime_headers_wait_ms`、
`runtime_body_read_ms` 和 `runtime_transport`。

```yaml
box_grasp:
  ipc:
    raw_http_enabled: true
    raw_http_fallback_urllib: true
```

Orbbec 场景下，深度路径优先使用 `/visionops_orbbec336l_depth` POSIX 共享内存。
App 在双缓冲活动帧上直接采样 7 个 ROI，并用 Header 中的有效内参进行针孔反投影；
共享内存失效、过期或校准未就绪时自动回退 Bridge 的
`/api/coordinate/sample_deproject`。HP60C 不启用此路径。

```yaml
camera_bridge:
  shared_depth_enabled: true
  shared_depth_name: /visionops_orbbec336l_depth
  shared_depth_fallback_http: true
```

`GET /api/app/status` 新增最近 100 个结果的 `latency_ms.p50/p95` 和
`app_timing_stats.<field>.p50/p95`，避免仅凭单帧抖动判断优化效果。

可观测性检查：

```bash
curl -s http://127.0.0.1:19211/api/app/status | jq '{
  latency_ms,
  app_timing_stats,
  ipc,
  last_app_timing
}'
```

正常生产路径中，`ipc.runtime.last_transport=raw_socket`，有目标时
`last_app_timing.depth_transport=posix_shared_memory`。`ipc.camera_bridge.shared_depth`
同时给出 mmap 状态、重试计数和最近错误；若回退到 HTTP，可以从
`ipc.camera_bridge.http.last_transport` 和 `last_raw_error` 继续定位。

共享深度路径使用有效内参的针孔公式做本地反投影，目的是删除逐帧 HTTP 往返。首次
真机部署必须抽取画面中心和四周若干点，与 Bridge SDK
`/api/coordinate/sample_deproject` 的结果进行误差对比；如果镜头畸变或边缘误差不满足
机器人精度要求，可暂时设置 `shared_depth_enabled: false` 回退 SDK 反投影。
