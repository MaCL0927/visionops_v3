# plastic_bag_grasp 机器人通信协议（M40.1）

适用任务：`plastic_bag_grasp_vision`  
传输：WebSocket JSON（底层 TCP）  
地址：`ws://<视觉盒IP>:9001/vision`

本任务沿用 `tube_pick_vision`、`detergent_grasp` 的 VisionOps 统一抓取点协议，不增加新的机器人侧解析格式。

## 1. 目标与抓取点定义

模型类别：

| class_id | class_name | 含义 |
|---:|---|---|
| 0 | `plastic_bag` | 整个透明塑料包装目标 |

机器人抓取点固定定义为：

```text
center_px = detection bbox 的中心点
```

因此塑料袋打结头部偏向左、右、上、下，都不会改变本任务的抓取点定义。

Runtime/模型使用 640x640 输入或 ROI 不影响协议：`center_px` 必须是 **Runtime 恢复到当前原始 RGB 帧后的像素坐标**。源图可能为 640x480，也可能为 1280x720；机器人应同时读取顶层 `image.width/height`，不要把像素尺寸写死。

## 2. items[] 核心字段

与既有统一协议保持一致：

```json
{
  "id": 0,
  "class_id": 0,
  "confidence": 0.95,
  "position_camera": [12.5, -34.2, 1260.0],
  "center_px": [818.5, 469.5]
}
```

字段含义：

- `id`：本帧目标序号；当前任务默认最多 1 个目标，因此通常为 0。
- `class_id`：`plastic_bag=0`。
- `confidence`：检测置信度。
- `center_px=[u,v]`：目标检测框中心，单位 pixel，坐标属于本帧 RGB 图像。
- `position_camera=[X,Y,Z]`：D2C 深度有效时的彩色相机坐标，单位 mm；深度无效时固定 `[0,0,0]`。

**RGB detection 有效但透明薄膜导致 depth 无效时，不会删除该目标。** 如果机器人侧使用自己完成的像素标定，直接使用 `center_px` 即可。

## 3. 单次 trigger（推荐机器人生产模式）

机器人发送：

```json
{
  "type": "trigger",
  "task_id": "plastic_bag_grasp",
  "request_id": 1001
}
```

也兼容既有 control trigger：

```json
{
  "type": "control",
  "command": "trigger",
  "request_id": 1001
}
```

服务首先返回 ACK，随后返回同一个 `request_id` 对应的 detection。机器人**必须按 `request_id` 配对结果**。

示例 detection：

```json
{
  "type": "detection",
  "request_id": 1001,
  "trigger_task_id": "plastic_bag_grasp",
  "frame_id": 48231,
  "timestamp": 1787100000.123,
  "task_id": "plastic_bag_grasp",
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.95,
      "position_camera": [21.4, -18.7, 842.6],
      "center_px": [818.5, 469.5]
    }
  ],
  "image": {
    "width": 1280,
    "height": 720
  },
  "coordinate_frame": "color_camera",
  "coordinate_unit": "mm",
  "fault_code": 0,
  "fault_type": "NONE"
}
```

没有检测到目标不是系统故障：

```json
{
  "type": "detection",
  "request_id": 1002,
  "task_id": "plastic_bag_grasp",
  "items": [],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

机器人应把 `items.length == 0` 解释为“当前帧没有可抓目标”，而不是通信故障。

## 4. 连续模式

沿用既有命令：

```json
{"type":"control","command":"start","request_id":1}
{"type":"control","command":"stop","request_id":2}
```

`auto_start=true` 时连接后可以直接收到连续 detection。若机器人只想按节拍抓取，建议连接成功后先发 `stop`，之后每个机器人周期发送一次带唯一 `request_id` 的 `trigger`。

## 5. FPS 与结果可靠性

本任务不设置历史 `push_hz=5` / `detection_hz=5` 之类独立推送限流：

- 推理线程按照 `production_inference_fps` 上限持续请求 Runtime，默认 30 FPS；
- WebSocket 每完成一个有效后处理结果就推送一次，因此连续推送频率跟随真实推理 FPS；
- 连续帧采用容量 1 的 latest-only 队列，旧连续帧允许覆盖；
- 显式 `trigger/request_id` 结果进入可靠路径，**不能被连续帧覆盖**；
- App→Runtime 使用已经验证的 raw local HTTP fast path，不默认走 `urllib.request`；
- JSON 解码放到后处理线程；
- Orbbec 深度优先共享内存 + 小 ROI 采样，不传整张 depth PNG；
- Runtime 推荐 `VISIONOPS_RUNTIME_HTTP_WORKERS=1`，避免此前多 worker 下出现的堆内存破坏问题。

查询当前 App FPS 设置：

```bash
curl -s http://127.0.0.1:19214/api/app/inference_settings | python3 -m json.tool
```

恢复 30 FPS 上限：

```bash
curl -s -X POST http://127.0.0.1:19214/api/app/inference_settings \
  -H 'Content-Type: application/json' \
  -d '{"detection_fps":30}' | python3 -m json.tool
```

## 6. 故障码

| fault_code | fault_type | 含义 |
|---:|---|---|
| 0 | `NONE` | 通信和推理链路正常；`items=[]` 仍可属于此状态 |
| 3101 | `CAMERA_DISCONNECTED` | 相机/Bridge 不可用或 RGB 帧异常 |
| 3201 | `VISION_INFERENCE_ERROR` | Runtime 请求、detection 解析或后处理异常 |

## 7. 机器人最小解析规则

机器人侧最少只需要：

1. 发带唯一 `request_id` 的 trigger；
2. 等同一 `request_id` 的 `type=detection`；
3. `fault_code != 0`：本周期失败；
4. `items=[]`：没有目标；
5. 否则取 `items[0].center_px`；
6. 若机器人选择使用相机三维坐标，只有 `position_camera != [0,0,0]` 时才使用 XYZ。
