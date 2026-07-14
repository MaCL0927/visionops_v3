# tube_pick_vision 新增 `lying` 类别对接说明

版本：v1.2  
适用任务：`tube_pick_vision`  
视觉盒 WebSocket：`ws://192.168.213.137:9001/vision`  
原始视频：`http://192.168.213.137:18182/stream.mjpeg`

## 1. 变更目的

纸筒检测模型新增倒伏纸筒类别 `lying`。该类别代表产线异常对象。视觉盒只负责检测并按现有
`detection.items` 格式返回结果，不在视觉侧生成独立告警状态；机器人系统收到该类别后自行决定
告警、停止抓取、人工介入等动作。

## 2. 类别映射

| class_id | 类别 | 含义 | 机器人侧建议 |
|---:|---|---|---|
| 0 | product | 正常直立纸筒产品 | 按原有抓取逻辑处理 |
| 1 | separator | 大隔板 | 按原有隔板逻辑处理 |
| 2 | lying | 倒伏纸筒，异常对象 | 触发机器人侧异常告警逻辑 |

## 3. WebSocket 返回格式

`lying` 与原有两类共用完全相同的数据结构：

```json
{
  "type": "detection",
  "frame_id": 2051,
  "timestamp": 1783936622.161,
  "items": [
    {
      "id": 0,
      "class_id": 2,
      "confidence": 0.9431,
      "position_camera": [-82.4, 41.7, 920.0],
      "center_px": [278.0, 258.0]
    }
  ],
  "image": {
    "width": 640,
    "height": 480
  },
  "coordinate_frame": "color_camera",
  "coordinate_unit": "mm",
  "video_url": "http://192.168.213.137:18182/stream.mjpeg",
  "video_sync": "soft",
  "latency_ms": 360.2
}
```

字段没有新增：

- `class_id=2`：倒伏纸筒；
- `confidence`：检测置信度；
- `position_camera=[X,Y,Z]`：彩色相机坐标系三维坐标，单位毫米；
- `center_px=[cx,cy]`：640×480 图像中的检测框中心；
- 深度无效时 `position_camera=[0,0,0]`，但 `class_id=2` 仍表示检测到了倒伏纸筒。

## 4. 机器人侧判断建议

机器人侧不应等待额外的 `alarm`、`abnormal` 或 `lying_detected` 字段，直接遍历 `items`：

```python
lying_items = [item for item in message.get("items", []) if item.get("class_id") == 2]
if lying_items:
    raise_lying_alarm(lying_items)
```

建议区分两个概念：

1. **类别检测有效**：`class_id == 2` 且 `confidence` 达到双方约定阈值；
2. **三维坐标有效**：`position_camera[2] > 0`。

即使深度无效，只要 `class_id=2` 存在，也应保留异常告警；是否允许机器人继续动作由机器人侧安全策略决定。

## 5. 与 trigger/request_id 的关系

机器人发送：

```json
{"type":"control","command":"trigger","request_id":"pick-1001"}
```

视觉盒返回的对应 `detection` 会原样携带：

```json
{
  "type":"detection",
  "request_id":"pick-1001",
  "frame_id":2051,
  "items":[
    {
      "id":0,
      "class_id":2,
      "confidence":0.9431,
      "position_camera":[-82.4,41.7,920.0],
      "center_px":[278.0,258.0]
    }
  ]
}
```

机器人应以 `request_id` 匹配本次触发结果，再检查其中是否存在 `class_id=2`。

## 6. 视频显示

MJPEG 仍是软同步原始视频。机器人前端可在最新画面上使用 `center_px` 绘制异常目标标记，推荐
将 `class_id=2` 显示为红色框或红色中心点。视频仅用于显示和标定观察，业务判断以 WebSocket
`detection` 为准。

## 7. ROI 影响

ROI 由 VisionOps 边缘 Web 设置。模型仍对完整图像推理，但 Runtime 在输出前按检测框中心过滤。
因此，只有中心位于 ROI 内的 `lying` 才会发送给机器人。机器人侧无需下发 ROI。

## 8. 验收项目

1. 画面中无倒伏纸筒时，不应出现 `class_id=2`；
2. 放入倒伏纸筒后，`items` 中出现 `class_id=2`；
3. `center_px` 能在 640×480 MJPEG 画面中正确高亮目标；
4. 深度有效时 `position_camera[2] > 0`；
5. 深度无效时返回 `[0,0,0]`，机器人仍能根据 `class_id=2` 告警；
6. trigger 模式下返回的 `request_id` 与请求一致；
7. ROI 外的倒伏纸筒不进入 WebSocket 结果。
