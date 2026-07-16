# tube_pick_vision 机器人故障码对接说明

版本：v1.3

## 接口地址

```text
WebSocket: ws://<盒子IP>:9001/vision
MJPEG:    http://<盒子IP>:18182/stream.mjpeg
```

实际盒子 IP 以现场配置为准。

## 对接变化

原有正常检测字段和 `start/stop/trigger/request_id` 流程不变。新增并固定以下两个扁平字段：

```json
"fault_code": 0,
"fault_type": "NONE"
```

机器人无需解析 `alarm`、`error`、`camera_state`、重连次数或 SDK 错误文本。

## 故障码

| fault_code | fault_type | 说明 |
|---:|---|---|
| 0 | `NONE` | 视觉正常 |
| 3101 | `CAMERA_DISCONNECTED` | RGB/Depth 相机不可用或正在自动恢复 |
| 3201 | `VISION_INFERENCE_ERROR` | 推理、深度处理或三维反投影异常 |

## 正常检测

```json
{
  "type": "detection",
  "request_id": "pick-001",
  "frame_id": 100,
  "timestamp": 1784160836.11,
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.93,
      "position_camera": [-225.3, -247.9, 856.0],
      "center_px": [224.9, 136.6]
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

## 相机断线

```json
{
  "type": "detection",
  "request_id": "pick-002",
  "frame_id": 101,
  "timestamp": 1784160837.20,
  "items": [],
  "fault_code": 3101,
  "fault_type": "CAMERA_DISCONNECTED"
}
```

机器人建议立即暂停抓取并报警。视觉盒会自行重连；恢复后重新返回 `fault_code=0`。

## 推理异常

```json
{
  "type": "detection",
  "request_id": "pick-003",
  "frame_id": 102,
  "timestamp": 1784160838.20,
  "items": [],
  "fault_code": 3201,
  "fault_type": "VISION_INFERENCE_ERROR"
}
```

## 机器人参考逻辑

```python
def handle_vision_message(message: dict) -> None:
    fault_code = int(message.get("fault_code", 0))
    fault_type = str(message.get("fault_type", "NONE"))

    if fault_code != 0:
        robot.stop_pick()
        robot.raise_alarm(fault_code, fault_type)
        return

    if message.get("type") != "detection":
        return

    lying = [
        item for item in message.get("items", [])
        if int(item.get("class_id", -1)) == 2
    ]
    if lying:
        robot.raise_lying_alarm(lying)
        return

    robot.consume_detection_items(message.get("items", []))
```

## 注意事项

1. `online=true` 只表示 WebSocket 服务在线；故障判断以 `fault_code` 为准。
2. 相机异常时 `items=[]`，视觉盒不会重复发送拔线前的旧结果。
3. `class_id=2` 仍表示倒伏纸筒，由机器人执行工艺异常告警。
4. MJPEG 仅用于显示和标定观察，采用软同步；断流后客户端需要自动重连。
5. 后续 PLC Modbus-TCP 可直接复用相同数值故障码，当前尚未实现寄存器写入。
