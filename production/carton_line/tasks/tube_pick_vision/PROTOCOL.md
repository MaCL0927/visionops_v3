# tube_pick_vision WebSocket 对接协议

版本：v1.3（简化故障码）

## 1. 网络与角色

- 视觉盒：WebSocket Server；
- 机器人后端：WebSocket Client；
- WebSocket：`ws://<盒子IP>:9001/vision`；
- 原始视频：`http://<盒子IP>:18182/stream.mjpeg`；
- JSON 使用 WebSocket 文本帧；
- 心跳使用 WebSocket 原生 Ping/Pong；
- MJPEG 与检测结果采用软同步，只用于显示和标定观察。

盒子 IP 及 `video_url` 以现场 `/etc/visionops_v3/carton_line.yaml` 为准。

## 2. 类别定义

| class_id | 含义 | 处理方式 |
|---:|---|---|
| 0 | 正常直立纸筒产品 | 返回检测结果和相机三维坐标 |
| 1 | 大隔板 | 返回检测结果和中心点相机三维坐标 |
| 2 | 倒伏纸筒 `lying` | 按普通目标返回；机器人侧负责告警 |

视觉盒使用普通 detection 模型，不返回旋转角度。

## 3. 正常检测消息

连续模式按配置频率推送；`trigger` 模式会额外原样返回 `request_id`。

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
      "class_id": 2,
      "confidence": 0.94,
      "position_camera": [-82.4, 41.7, 920.0],
      "center_px": [278.0, 258.0]
    }
  ],
  "image": {"width": 640, "height": 480},
  "coordinate_frame": "color_camera",
  "coordinate_unit": "mm",
  "video_url": "http://<盒子IP>:18182/stream.mjpeg",
  "video_sync": "soft",
  "latency_ms": 58.3,
  "fault_code": 0,
  "fault_type": "NONE",
  "source": {
    "runtime_frame_id": "frame-hp60c-00000102",
    "runtime_result_id": "result-rknn-00000102"
  }
}
```

字段说明：

- `frame_id`：视觉服务进程内递增序号；
- `timestamp`：本次推理所用图像的采集时间或推理开始时间，Unix 秒；
- `position_camera`：彩色相机坐标系 `[X,Y,Z]`，X 向右、Y 向下、Z 向前，单位 mm；
- `center_px`：640×480 RGB 图像中的检测框中心；
- 深度无效时：`position_camera=[0,0,0]`，但二维检测仍有效；
- `fault_code=0`、`fault_type=NONE`：视觉服务正常。

## 4. 故障消息

机器人 WebSocket 只暴露两个稳定故障字段，不发送 Bridge 内部重连状态、SDK 错误文本、告警确认状态或 Modbus 预留结构。

### 4.1 相机不可用

RGB 或 Depth 无有效新帧、相机拔出、Bridge 正在重连或 Bridge 无法访问，均统一映射为：

```json
{
  "type": "detection",
  "request_id": "pick-1008",
  "frame_id": 1025,
  "timestamp": 1783905060.100,
  "items": [],
  "latency_ms": 42.1,
  "fault_code": 3101,
  "fault_type": "CAMERA_DISCONNECTED"
}
```

异常期间不会继续发送拔线前的旧检测结果。

### 4.2 推理服务异常

相机正常，但 Runtime 推理、结果解析、深度处理或三维反投影发生未归类异常时：

```json
{
  "type": "detection",
  "request_id": "pick-1009",
  "frame_id": 1026,
  "timestamp": 1783905060.500,
  "items": [],
  "latency_ms": 51.2,
  "fault_code": 3201,
  "fault_type": "VISION_INFERENCE_ERROR"
}
```

## 5. 故障码表

| fault_code | fault_type | 含义 | 机器人建议 |
|---:|---|---|---|
| 0 | `NONE` | 正常 | 正常消费 `items` |
| 3101 | `CAMERA_DISCONNECTED` | RGB/Depth 相机不可用或正在恢复 | 暂停抓取并报警，等待恢复 |
| 3201 | `VISION_INFERENCE_ERROR` | 相机正常但视觉推理链路失败 | 暂停本次动作并报警/重试 |

机器人只需判断：

```python
fault_code = int(message.get("fault_code", 0))
if fault_code != 0:
    stop_pick_and_raise_alarm(fault_code, message.get("fault_type"))
```

未来接入 PLC Modbus-TCP 时，可直接把 `fault_code` 写入约定故障寄存器。当前版本尚未实现寄存器通信。

## 6. 状态消息

建立连接后立即发送，之后按配置周期发送：

正常：

```json
{
  "type": "status",
  "online": true,
  "fps": 3.2,
  "model": "detection-tube-pick",
  "camera_connected": true,
  "fault_code": 0,
  "fault_type": "NONE",
  "latency_ms": 58.3,
  "continuous_enabled": true,
  "clients": 1,
  "video_url": "http://<盒子IP>:18182/stream.mjpeg"
}
```

相机异常：

```json
{
  "type": "status",
  "online": true,
  "fps": 3.2,
  "model": "detection-tube-pick",
  "camera_connected": false,
  "fault_code": 3101,
  "fault_type": "CAMERA_DISCONNECTED",
  "latency_ms": 42.1,
  "continuous_enabled": true,
  "clients": 1,
  "video_url": "http://<盒子IP>:18182/stream.mjpeg"
}
```

`online=true` 表示 WebSocket 服务在线，不代表相机一定正常。机器人应以 `fault_code` 为故障判断主字段。

## 7. 控制命令

```json
{"type":"control","command":"start","request_id":101}
{"type":"control","command":"stop","request_id":102}
{"type":"control","command":"trigger","request_id":103}
```

- `start`：开启连续推理；
- `stop`：暂停连续推理，连接保持；
- `trigger`：立即请求一次检测；
- `trigger.request_id` 必填，允许整数或非空字符串；
- 对应 detection 原样返回 `request_id`；
- 连续模式 detection 不带 `request_id`。

接收 trigger 后先返回：

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

## 8. ROI、视频和三维坐标

- ROI 只在 VisionOps Web 设置，机器人不下发 ROI；
- 模型仍对完整 640×480 图像推理，Runtime 在输出阶段按目标中心过滤 ROI；
- RGB 与 D2C Depth 固定为 640×480；
- 三维反投影由 Orbbec SDK 完成；
- 机器人负责手眼标定和相机坐标到机器人坐标的转换；
- MJPEG 断线后机器人视频客户端应自动重新连接；
- MJPEG 为软同步，不应使用视频帧替代 detection 中的三维坐标。

## 9. 视觉盒本地诊断

复杂的相机状态、帧年龄、重连次数和 SDK 错误信息仍保留在视觉盒本地 HTTP 接口，供维护人员排查：

```bash
curl -s http://127.0.0.1:19130/api/app/status | python3 -m json.tool
curl -s http://127.0.0.1:18182/health | python3 -m json.tool
```

这些诊断字段不属于机器人 WebSocket 对接契约。
