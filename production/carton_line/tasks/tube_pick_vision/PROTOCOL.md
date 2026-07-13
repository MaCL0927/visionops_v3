# 外部推理盒子 WebSocket 契约（tube_pick_vision）

## 1. 网络

- 视觉盒：WebSocket Server；
- 机器人后端：WebSocket Client；
- 地址：`ws://<box-ip>:9001/vision`；
- JSON 使用 WebSocket 文本帧；
- 心跳使用 WebSocket 原生 Ping/Pong；
- 原始视频：`http://<box-ip>:18182/stream.mjpeg`，与检测结果软同步。

## 2. 检测结果

连续模式按配置频率推送：

```json
{
  "type": "detection",
  "frame_id": 1024,
  "timestamp": 1783905059.684,
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.92,
      "position_camera": [12.5, -34.2, 1260.0],
      "center_px": [320.0, 240.0]
    },
    {
      "id": 1,
      "class_id": 1,
      "confidence": 0.90,
      "position_camera": [15.0, 20.0, 1310.0],
      "center_px": [350.0, 260.0]
    }
  ],
  "image": {"width": 640, "height": 480},
  "coordinate_frame": "color_camera",
  "coordinate_unit": "mm",
  "video_url": "http://192.168.2.211:18182/stream.mjpeg",
  "video_sync": "soft",
  "latency_ms": 58.3
}
```

字段：

- `frame_id`：盒子服务进程内递增序号，用于排序；MJPEG 软同步不提供严格逐帧对应；
- `timestamp`：本次采集/推理开始时的 Unix 秒；
- `class_id=0`：产品；`class_id=1`：隔板；
- `position_camera`：彩色相机光心坐标系，X 向右、Y 向下、Z 向前，单位毫米；
- `center_px`：640×480 RGB 图像中的检测框中心；
- 深度无效：`position_camera=[0,0,0]`；
- 不返回 `angle_deg`，当前模型为普通 detection，机器人端按无角度处理。

内部推理异常时连接保持，盒子返回：

```json
{
  "type": "detection",
  "frame_id": 1025,
  "timestamp": 1783905060.100,
  "items": [],
  "error": {"code": "UpstreamError", "message": "..."}
}
```

## 3. 控制

```json
{"type":"control","command":"start","request_id":101}
{"type":"control","command":"stop","request_id":102}
{"type":"control","command":"trigger","request_id":103}
```

- `start`：开启连续推理；
- `stop`：暂停连续推理，连接保持；
- `trigger`：请求立即执行一次推理；
- `trigger.request_id` 必填，允许整数或非空字符串；
- trigger 对应的 detection 原样返回 `request_id`；
- 连续模式 detection 不带 `request_id`。

接收命令后先返回排队确认：

```json
{
  "type":"ack",
  "request_type":"control",
  "request_id":103,
  "command":"trigger",
  "success":true,
  "queued":true
}
```

随后返回：

```json
{
  "type":"detection",
  "request_id":103,
  "frame_id":1026,
  "timestamp":1783905060.500,
  "items":[]
}
```

## 4. 状态

连接建立后立即发送，并按配置周期推送：

```json
{
  "type":"status",
  "online":true,
  "fps":10.0,
  "model":"tube_pick_vision",
  "camera_connected":true,
  "latency_ms":58.3,
  "continuous_enabled":true,
  "clients":1,
  "video_url":"http://192.168.2.211:18182/stream.mjpeg",
  "error":null
}
```

## 5. ROI 与阈值

机器人侧不下发 ROI 或阈值。ROI 由 VisionOps 边缘 Web 设置：

```text
整图输入模型 → Runtime 后处理 → 目标中心位于 ROI 内才保留 → WebSocket 输出
```

因此 Web 模型验证、WebSocket、其他 Runtime 调用获得相同的 ROI 过滤结果。

## 6. 三维坐标

视觉服务从 D2C 对齐 16UC1 深度图采样中心邻域深度，然后调用 336L Bridge：

```text
POST /api/coordinate/deproject
{"points":[[u,v,depth_mm], ...]}
```

Bridge 内部使用 Orbbec SDK `CoordinateTransformHelper::calibration2dTo3d()`，输出彩色相机坐标系毫米值。视觉系统不执行手眼标定和机器人坐标转换。
