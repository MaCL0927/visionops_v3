# VisionOps carton_palletizing 触发式机器人对接协议（M29.2）

> 适用任务：`production/carton_palletizing/tasks/first_layer_placement`  
> 传输：WebSocket JSON + 独立 MJPEG 视频  
> 连接：视觉盒为 Server，机器人调度端为 Client  
> 地址：`ws://<视觉盒IP>:9001/vision`

本任务遵循机器人侧《外部视觉盒子通信协议》v2.0-draft 的消息名称、字段层级、心跳和 trigger 关联规则。模型为 OBB，返回目标均包含 `angle_deg`。

## 1. 两个 trigger 任务

### 1.0 任务号兼容规则

为兼容机器人侧更简洁的任务编号，两个任务同时接受原字符串任务名和数字别名：

| 业务 | 原任务名 | 数字别名 |
|---|---|---:|
| 获取托盘或垛顶纸箱信息 | `pallet_place_target` | `1` |
| 获取机器人手持纸箱信息 | `held_box_pose` | `2` |

以下写法均有效：

```json
{"type":"trigger","task_id":"pallet_place_target"}
{"type":"trigger","task_id":"1"}
{"type":"trigger","task_id":1}
```

```json
{"type":"trigger","task_id":"held_box_pose"}
{"type":"trigger","task_id":"2"}
{"type":"trigger","task_id":2}
```

响应中的 `trigger_task_id` **原样回显请求值**，便于机器人直接关联：

- 请求 `task_id: 1`，响应 `trigger_task_id: 1`；
- 请求 `task_id: "1"`，响应 `trigger_task_id: "1"`；
- 请求原任务名，响应仍返回原任务名。

数字别名可在 `task.communication.trigger_tasks` 中配置；即使旧 `/etc` 配置没有别名字段，代码默认仍兼容 `1/2`。

### 1.1 托盘/垛顶目标观测

机器人发送：

```json
{"type":"trigger","task_id":"pallet_place_target"}
```

M29 中，视觉盒**不再规划摆放位置，不再向机器人返回层号、slot_id 或下一位置**。视觉只返回当前检测到的放置支撑目标：

- 没有检测到纸箱：返回置信度最高的一个托盘；
- 检测到纸箱：按中心深度分层，只返回离相机最近的最上层纸箱；
- 同一最上层返回 1～4 个纸箱；
- 机器人自行决定使用哪个目标以及如何摆放。

仅有托盘时：

```json
{
  "type": "detection",
  "frame_id": 101,
  "timestamp": 1700000000.123,
  "trigger_task_id": "pallet_place_target",
  "items": [
    {
      "id": 0,
      "class_id": 1,
      "confidence": 0.98,
      "position_camera": [-5.2, 41.3, 1280.0],
      "angle_deg": 0.0,
      "center_px": [320.0, 245.0],
      "type": null
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

存在两层纸箱、最上层有两个纸箱时：

```json
{
  "type": "detection",
  "frame_id": 102,
  "timestamp": 1700000001.123,
  "trigger_task_id": "pallet_place_target",
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.96,
      "position_camera": [-118.4, -62.1, 735.0],
      "angle_deg": 5.5,
      "center_px": [266.0, 176.0],
      "type": null
    },
    {
      "id": 1,
      "class_id": 0,
      "confidence": 0.94,
      "position_camera": [106.8, -58.7, 752.0],
      "angle_deg": -3.0,
      "center_px": [372.0, 180.0],
      "type": null
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

`items[]` 中不再包含：

```text
layer
slot_id
target_kind
internal_sample_count
```

### 1.2 机器人手持纸箱

机器人发送：

```json
{"type":"trigger","task_id":"held_box_pose"}
```

手持纸箱判断逻辑保持不变：

```json
{
  "type": "detection",
  "frame_id": 103,
  "timestamp": 1700000002.123,
  "trigger_task_id": "held_box_pose",
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.95,
      "position_camera": [65.2, -141.8, 820.0],
      "angle_deg": -18.0,
      "center_px": [516.0, 170.0],
      "type": null,
      "source_detection_id": "det-7"
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE",
  "result_state": "HELD_BOX_READY"
}
```

## 2. 最上层纸箱选择

视觉对位于托盘/垛型区域中的每个纸箱 OBB 中心读取 D2C 对齐深度，按深度从小到大排序。离相机越近，深度越小。

从最近纸箱开始构建最上层深度簇；出现以下任一情况时，后续目标视为下一层并停止加入：

```text
与前一个纸箱的深度差 > layer_gap_mm
或
与最近纸箱的总深度跨度 > max_top_layer_span_mm
```

默认配置：

```yaml
task:
  communication:
    surface_target_selection:
      max_items: 4
      layer_gap_mm: 80.0
      max_top_layer_span_mm: 140.0
      filter_to_tray_region: true
      tray_expand_ratio: 0.08
      min_tray_overlap_ratio: 0.05
      allow_boxes_without_tray_reference: true
      sort_order: image_yx
      depth:
        roi_radius_px: 6
        percentile: 50.0
        min_valid_pixels: 3
        min_depth_mm: 100
        max_depth_mm: 5000
```

同层纸箱因相机倾角导致深度差较大时，可适当增大 `layer_gap_mm` 和 `max_top_layer_span_mm`；不能增大到接近真实纸箱高度，否则会把下一层纸箱也纳入。


## 3. ROI 控制权

纸箱码垛任务的检测 ROI **只能由 VisionOps Web 界面设置**。Web 设置会写入 Runtime 的
`runtime.roi_config_path`，Runtime 在生成 `detections[]` 前完成 ROI 过滤。

机器人连接后仍可按通用协议发送：

```json
{
  "type": "config",
  "detect_region": [100, 10, 540, 470],
  "confidence_threshold": 0.5
}
```

但本任务对 `detect_region` 仅做兼容接收和诊断记录，**永久不应用**，因此机器人不能覆盖
Web ROI，也不会因 640×480 与 1280×720 的像素区域不一致而把有效托盘/纸箱过滤掉。
`confidence_threshold` 默认仍允许动态调整；可在 `remote_config.allow_confidence_threshold`
中关闭。

调试结果会包含：

```json
{
  "roi_control_source": "visionops_web_runtime",
  "robot_detect_region_applied": false,
  "last_ignored_robot_detect_region": [100.0, 10.0, 540.0, 470.0]
}
```

## 4. 放置观测不再推进视觉摆放状态

`pallet_place_target` 每次 trigger 只执行：

```text
一次 OBB 推理
→ 一次深度图读取
→ 托盘/垛顶目标筛选
→ 一帧 detection 返回
```

它不会再执行：

```text
slot 占位确认
层号推进
奇偶层模板切换
下一摆放位置规划
深度基准采集
```

原多层摆放算法仍保留给生产界面可视化和兼容调试，但机器人通信不依赖这些状态。

## 5. 初期/后期手持箱配置

手持纸箱任务仍通过配置切换，无需修改代码。

初期固定机器人：

```yaml
task:
  communication:
    held_box_selection:
      mode: nearest_depth
```

后期机器人可移动：

```yaml
task:
  communication:
    held_box_selection:
      mode: outside_tray
```

## 6. 坐标与角度

- `center_px=[u,v]`：OBB 中心像素坐标；
- `position_camera=[x,y,z]`：336L 彩色相机坐标系，单位毫米；
- X 向右、Y 向下、Z 向前；
- 深度在中心附近小区域取中位数；
- `angle_deg`：OBB 长轴方向，归一化到 `[-90,90]`；
- `class_id=0` 表示纸箱，`class_id=1` 表示托盘。

## 7. 无目标和故障

正常但未检测到托盘或纸箱：

```json
{
  "type": "detection",
  "trigger_task_id": "pallet_place_target",
  "items": [],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

相机/深度不可用：`fault_code=3101`。推理、OBB、深度点或反投影异常：`fault_code=3201`。

## 8. 心跳与视频

- 支持 WebSocket 原生 Ping/Pong；
- 支持应用层 `{"type":"ping"}` / `{"type":"pong"}`；
- trigger 模式默认关闭周期 `status`；
- 视频使用 `http://<视觉盒IP>:18182/stream.mjpeg`，与 JSON 软同步。

## 9. 调试

```bash
curl -s -X POST http://127.0.0.1:19210/api/app/trigger \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"pallet_place_target"}' | python3 -m json.tool
```

重点检查：

```text
robot_message.items
surface_target_selection.diagnostics
```

模拟机器人：

```bash
python3 -m production.carton_palletizing.tasks.first_layer_placement.mock_robot_client \
  --url ws://127.0.0.1:9001/vision
```
