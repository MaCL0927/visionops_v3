# First Layer Placement（OBB）

输入必须是 Runtime 标准 OBB `inference_result`：

```json
{
  "task_type": "obb",
  "detections": [
    {
      "class_id": 0,
      "class_name": "box",
      "bbox_xyxy": [100, 100, 300, 220],
      "obb": {
        "cx": 200,
        "cy": 160,
        "w": 200,
        "h": 120,
        "angle_deg": 3.0,
        "points": [[100, 95], [305, 105], [300, 225], [95, 215]]
      }
    }
  ]
}
```

类别：

```text
0 = box
1 = tray
```

输出 `placement`：

```json
{
  "layer": 1,
  "state": "LAYER_1_FILLING",
  "occupied_count": 2,
  "slot_count": 4,
  "next_slot_id": "P3",
  "tray": {
    "obb_points": [],
    "angle_deg": 2.8
  },
  "slots": []
}
```

`footprint` 给出由托盘短边生成的居中正方形垛型区域；每个 slot 含绝对像素 `polygon`、`bbox_xyxy`、实际图像朝向 `orientation_deg`、模板方向 `orientation_label`、`occupied` 和 `visible_mask`。前端只绘制 `visible_mask=true` 的区域。

默认摆放顺序为 `P3 -> P1 -> P2 -> P4`，从左下角开始按顺时针方向进行。

匹配依据：

- 纸箱 OBB 与 slot 多边形 IoU；
- 纸箱中心是否进入 slot；
- 中心距离；
- 纸箱长轴方向与 slot 方向差。

占位状态默认连续两帧确认并保持粘滞；更换托盘时通过 `/api/app/reset` 清除。
