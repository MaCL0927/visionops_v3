# VisionOps 抓取点 WebSocket 协议（box_grasp_vision）

版本：v1.3  
适用任务：`box_grasp_vision`  
传输方式：WebSocket JSON + MJPEG 视频流

## 1. 统一语义

机器人侧接收的 `items[]` 不再表示“检测目标列表”，而表示“抓取点列表”。

每一个抓取点固定使用同一组字段：

```json
{
  "id": 0,
  "class_id": 0,
  "confidence": 0.92,
  "position_camera": [12.5, -34.2, 1260.0],
  "center_px": [320.0, 240.0]
}
```

字段含义：

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `id` | int | — | 抓取点所属目标的 ID；同一目标的多个抓取点使用相同 ID |
| `class_id` | int | — | 目标类别 ID |
| `confidence` | float | — | 目标检测置信度 `[0,1]`；同一目标的多个点使用相同置信度 |
| `position_camera` | `[x,y,z]` | mm | 当前抓取点在彩色相机坐标系中的三维坐标 |
| `center_px` | `[u,v]` | pixel | 当前抓取点的像素坐标；此处不是纸箱整体中心 |

`box_grasp_vision` 每个纸箱有两个抓取点，因此同一个 `id` 在 `items[]` 中出现两次。算法先计算纸箱左右边中点，再将两点分别向纸箱中心内缩；两项不使用 `left/right` 字段区分。机器人根据相同 `id` 分组，再根据像素坐标或自身坐标规则区分两点。

视觉盒输出顺序保持确定性：同一纸箱的两个抓取点按 `center_px[0]` 从小到大排列。但机器人不应仅依赖数组顺序，应优先按 `id` 分组。

## 2. 网络地址

- 视觉盒：WebSocket Server
- 机器人后端：WebSocket Client
- 检测结果：`ws://<视觉盒IP>:9001/vision`
- 视频流：`http://<视觉盒IP>:18182/stream.mjpeg`
- RGB/Depth 固定使用 `640×480`

不同视觉任务部署在不同 IP 的视觉盒上时，WebSocket 端口统一使用 `9001`，仅修改 IP。

## 3. 正常 detection 消息

一个纸箱、两个抓取点：

```json
{
  "type": "detection",
  "request_id": "box-001",
  "frame_id": 1024,
  "timestamp": 1690000000.123,
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.92,
      "position_camera": [-164.0, 0.2, 930.0],
      "center_px": [228.5, 278.5]
    },
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.92,
      "position_camera": [148.6, -6.8, 926.0],
      "center_px": [441.5, 271.5]
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

两个纸箱时，假设目标 ID 分别为 0、1，则通常返回四个抓取点：

```json
{
  "type": "detection",
  "frame_id": 1025,
  "timestamp": 1690000000.223,
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.92,
      "position_camera": [-164.0, 0.2, 930.0],
      "center_px": [228.5, 278.5]
    },
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.92,
      "position_camera": [148.6, -6.8, 926.0],
      "center_px": [441.5, 271.5]
    },
    {
      "id": 1,
      "class_id": 0,
      "confidence": 0.88,
      "position_camera": [-120.2, 91.7, 1040.0],
      "center_px": [252.0, 356.0]
    },
    {
      "id": 1,
      "class_id": 0,
      "confidence": 0.88,
      "position_camera": [101.5, 87.2, 1035.0],
      "center_px": [405.0, 350.0]
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

## 4. 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `type` | string | 是 | 固定为 `detection` |
| `frame_id` | int | 是 | 视觉盒递增帧序号 |
| `timestamp` | float | 是 | Unix 秒时间戳，含毫秒小数 |
| `items` | array | 是 | 抓取点列表；无目标时为空数组 |
| `request_id` | string/int | trigger 时是 | 原样返回触发请求的关联 ID |
| `fault_code` | int | 是 | `0` 正常，非 0 故障 |
| `fault_type` | string | 是 | 与 `fault_code` 对应的故障类型 |

核心抓取协议只依赖 `type/frame_id/timestamp/items`。`request_id/fault_code/fault_type` 保留现有 VisionOps 的请求配对和故障处理能力。

## 5. 与 tube_pick_vision 的统一关系

两个任务使用相同的 `items[]` 字段：

```text
id / class_id / confidence / position_camera / center_px
```

差异仅在抓取点数量：

- `tube_pick_vision`：一个产品通常只有一个抓取点，因此每个目标 ID 通常出现一次；
- `box_grasp_vision`：一个纸箱有两个抓取点，因此每个纸箱 ID 正常出现两次。

机器人侧可使用统一解析逻辑：

1. 按 `id` 对 `items[]` 分组；
2. 一组 1 个点：执行单点抓取逻辑；
3. 一组 2 个点：执行双点/双臂抓取逻辑；
4. 超出预期数量时，按任务规则报警或筛选。

## 6. 坐标约定

像素坐标：

- 图像分辨率：`640×480`
- 原点：RGB 图像左上角
- u/x：向右
- v/y：向下

相机坐标：

- 坐标系：Orbbec 336L 彩色相机坐标系
- 原点：相机光心
- X：向右
- Y：向下
- Z：向前
- 单位：毫米

深度或反投影无效时，`position_camera` 返回：

```json
[0.0, 0.0, 0.0]
```

此时仍可保留 `center_px`，但机器人不得使用零三维坐标执行抓取。

## 7. 无目标与故障

正常但无目标：

```json
{
  "type": "detection",
  "frame_id": 1026,
  "timestamp": 1690000000.323,
  "items": [],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

相机异常：

```json
{
  "type": "detection",
  "frame_id": 1027,
  "timestamp": 1690000000.423,
  "items": [],
  "fault_code": 3101,
  "fault_type": "CAMERA_DISCONNECTED"
}
```

推理、深度或三维反投影异常：

```json
{
  "type": "detection",
  "frame_id": 1028,
  "timestamp": 1690000000.523,
  "items": [],
  "fault_code": 3201,
  "fault_type": "VISION_INFERENCE_ERROR"
}
```

| fault_code | fault_type | 含义 |
|---:|---|---|
| 0 | `NONE` | 正常 |
| 3101 | `CAMERA_DISCONNECTED` | RGB/Depth 相机不可用或帧过旧 |
| 3201 | `VISION_INFERENCE_ERROR` | 推理、深度采样或反投影异常 |

## 8. 控制指令

```json
{"type":"control","command":"start","request_id":"ctrl-001"}
```

```json
{"type":"control","command":"stop","request_id":"ctrl-002"}
```

```json
{"type":"control","command":"trigger","request_id":"box-003"}
```

`trigger` 的 `request_id` 会在对应 detection 消息中原样返回。

## 9. 机器人解析示例

```python
from collections import defaultdict


def group_grasp_points(message: dict) -> dict[int, list[dict]]:
    if int(message.get("fault_code", 0)) != 0:
        raise RuntimeError(message.get("fault_type", "VISION_ERROR"))

    grouped = defaultdict(list)
    for point in message.get("items", []):
        grouped[int(point["id"])].append(point)

    for points in grouped.values():
        points.sort(key=lambda point: float(point["center_px"][0]))
    return dict(grouped)
```

## 10. 可视化数据

机器人 WebSocket 报文只发送抓取点。以下数据仍保留在 `visualization_result` 和本地调试文件中，不发送给机器人：

- 分割外轮廓；
- 四个透视角点；
- 纸箱几何中心；
- 深度采样详情；
- 四边形拟合质量；
- Runtime 原始结果。
