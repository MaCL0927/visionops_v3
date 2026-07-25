# 洗衣液抓取机器人协议（M30.2）

适用任务：`detergent_grasp_vision`  
传输：WebSocket JSON（底层 TCP）+ 独立 MJPEG 视频流  
检测地址：`ws://<视觉盒IP>:9001/vision`  
视频地址：`http://<视觉盒IP>:18181/stream.mjpeg`（端口随 active_camera 自动切换）

本任务沿用此前 VisionOps 外部视觉盒协议，不另起一套裸 TCP 分帧方式。原有顶层字段和 `items[]` 核心字段均保留。

## 1. 模型类别

默认配置按当前模型约定：

| class_id | 默认名称 | 语义 |
|---:|---|---|
| 0 | `big` | 大瓶洗衣液 |
| 1 | `head` | 瓶体抓取点 |
| 2 | `box` | 目标纸箱 |
| 3 | `small` | 小瓶洗衣液 |

类别名优先于 class_id。模型重新训练、类别顺序变化后，只需修改 `config/line.yaml` 的 `algorithm.classes`。

## 2. 抓取点匹配

视觉先解析全部 OBB，再把 `head` 中心分配给大瓶或小瓶：

1. 优先匹配中心位于扩大后瓶体 OBB 内的 `head`；
2. 如果不在 OBB 内，允许按瓶体对角线归一化距离匹配；
3. 一个 `head` 只能属于一个瓶体；
4. 默认 `require_grasp_point=true`，没有匹配到抓取点的瓶体不会发给机器人；
5. 纸箱不需要 `head`，直接返回纸箱 OBB 中心和角度。

## 3. detection 示例

```json
{
  "type": "detection",
  "frame_id": 1024,
  "timestamp": 1784900000.123,
  "task_id": "detergent_grasp",
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.83,
      "position_camera": [0.0, 0.0, 0.0],
      "angle_deg": 90.0,
      "center_px": [130.0, 122.5],
      "type": null,
      "target_type": "big_bottle",
      "class_name": "big",
      "object_center_px": [130.0, 180.0],
      "grasp_point_px": [130.0, 122.5],
      "grasp_confidence": 0.91,
      "obb_points": [[80.0, 100.0], [180.0, 100.0], [180.0, 260.0], [80.0, 260.0]],
      "source_detection_id": "det-0",
      "grasp_source_detection_id": "det-1"
    },
    {
      "id": 1,
      "class_id": 3,
      "confidence": 0.72,
      "position_camera": [0.0, 0.0, 0.0],
      "angle_deg": 33.7,
      "center_px": [220.0, 150.0],
      "type": null,
      "target_type": "small_bottle",
      "class_name": "small",
      "object_center_px": [230.0, 162.5],
      "grasp_point_px": [220.0, 150.0],
      "grasp_confidence": 0.89,
      "obb_points": [[220.0, 110.0], [280.0, 150.0], [240.0, 215.0], [180.0, 175.0]],
      "source_detection_id": "det-4",
      "grasp_source_detection_id": "det-5"
    },
    {
      "id": 2,
      "class_id": 2,
      "confidence": 0.93,
      "position_camera": [0.0, 0.0, 0.0],
      "angle_deg": 2.6,
      "center_px": [455.0, 210.0],
      "type": null,
      "target_type": "box",
      "class_name": "box",
      "object_center_px": [455.0, 210.0],
      "grasp_point_px": null,
      "obb_points": [[350.0, 90.0], [570.0, 100.0], [560.0, 330.0], [340.0, 320.0]],
      "source_detection_id": "det-3"
    }
  ],
  "image": {"width": 640, "height": 480},
  "coordinate_frame": "image",
  "coordinate_unit": "pixel",
  "video_url": "http://<视觉盒IP>:18181/stream.mjpeg",
  "video_sync": "soft",
  "latency_ms": 52.3,
  "fault_code": 0,
  "fault_type": "NONE",
  "source": {
    "runtime_frame_id": "frame-hp60c-00000102",
    "runtime_result_id": "result-rknn-00000102"
  }
}
```

## 4. 字段兼容说明

未修改的既有字段：

```text
顶层：type / frame_id / timestamp / items / request_id /
      fault_code / fault_type / video_url
items：id / class_id / confidence / position_camera /
       angle_deg / center_px / type
```

本任务新增字段：

| 字段 | 说明 |
|---|---|
| `target_type` | `big_bottle`、`small_bottle` 或 `box` |
| `class_name` | Runtime 模型类别名 |
| `object_center_px` | 瓶体或纸箱的 OBB 中心 |
| `grasp_point_px` | 瓶体匹配后的 `head` 中心；纸箱为 null |
| `grasp_confidence` | `head` 检测置信度，仅瓶体存在 |
| `obb_points` | 目标 OBB 四角像素坐标 |
| `source_detection_id` | Runtime 原目标 ID，便于日志追踪 |
| `grasp_source_detection_id` | 匹配的 `head` 原目标 ID |

关键约定：

- 对大/小瓶，`center_px` 继续表示机器人实际使用的抓取点，与此前抓取协议一致；
- 对纸箱，`center_px` 表示纸箱 OBB 中心；
- **字段名 `angle_deg` 未修改。** 对瓶体，它现在表示消除 180° 歧义后的把手方向，范围为 `[-180, 180)`；对纸箱没有可用于判断正反朝向的附加目标，因此仍表示无向 OBB 长轴角度，数值自然落在 `[-90, 90]`；
- 当前任务只有 RGB 图像，没有深度反投影，因此保留 `position_camera` 字段但固定为 `[0,0,0]`。机器人应使用像素手眼标定；后续加入深度或平面标定时无需改字段名。

## 4.1 瓶体 360° 角度的计算与定义

瓶体 OBB 本身只能给出一条无向长轴，所以 `-88°` 与 `92°` 对旋转矩形而言是同一条轴。M30.2 使用已经匹配到瓶体的 `head` 抓取点来确定正反方向：

1. 取得瓶体 OBB 中心 `object_center_px` 和无向长轴；
2. 沿长轴从瓶体中心构造两个相反端点；
3. 分别计算 `head` 中心到两个端点的距离；
4. **距离 `head` 更远的端点定义为洗衣液把手方向**；
5. 将该有向向量转换成 `[-180, 180)` 的 `angle_deg`。

角度采用图像坐标系：

```text
                    -90°（图像上方）
                           ↑
 ±180°（图像左方） ← 0°（图像右方）
                           ↓
                    +90°（图像下方）
```

也就是说，`angle_deg` 是“从图像水平向右的 +x 轴，旋转到瓶体把手方向”的夹角。由于图像 y 轴向下，**正角度为顺时针**。它是俯拍图像平面内的投影角，不是相机三维坐标系相对于真实水平面的俯仰角。机器人若采用逆时针为正，通常需要结合手眼标定的零位偏移进行符号和零点转换。

## 5. 控制和触发

连续模式：

```json
{"type":"control","command":"start","request_id":1}
{"type":"control","command":"stop","request_id":2}
```

单次触发，兼容两种既有写法：

```json
{"type":"control","command":"trigger","request_id":3}
```

```json
{"type":"trigger","task_id":"detergent_grasp","request_id":4}
```

也接受 `task_id: "detergent_pick"`、`task_id: "1"` 和 `task_id: 1`。`trigger_task_id` 在 detection 中原样回显。

## 6. FPS

App 只有一个生产推理 FPS：

- Runtime 推理完成一帧，就向 WebSocket 推送这一帧；
- 不存在独立的 `push_hz` 或固定 5 Hz 配置；
- `configured_fps` 是上限，算力不足时 `actual_fps` 自然降低；
- Web 生产画面和机器人消息消费同一份 App 结果。

查询和修改：

```bash
curl -s http://127.0.0.1:19212/api/app/inference_settings | python3 -m json.tool
curl -s -X POST http://127.0.0.1:19212/api/app/inference_settings \
  -H 'Content-Type: application/json' -d '{"detection_fps":15}' | python3 -m json.tool
```

## 7. 故障码

| fault_code | fault_type | 含义 |
|---:|---|---|
| 0 | `NONE` | 正常 |
| 3101 | `CAMERA_DISCONNECTED` | Bridge 不可访问、RGB 帧过旧或相机断开 |
| 3201 | `VISION_INFERENCE_ERROR` | Runtime、OBB 解析或抓取点匹配链路异常 |
