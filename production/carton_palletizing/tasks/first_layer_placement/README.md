# Multi-Layer Placement（OBB + RGB-D）

目录名为兼容早期第一层版本而保留，实际算法类已经升级为 `MultiLayerPlacementAlgorithm`，并保留 `FirstLayerPlacementAlgorithm` 别名。

## 分层依据

- 第 1 层：纸箱 OBB 与四个 slot 进行几何匹配；
- 第 2 层及以后：使用上一层完成后的深度图作为基准，按 slot 判断新增高度；
- 每层完成：采集稳定深度基准并自动进入下一层；
- 下一层掩膜：优先继承上一层实际检测纸箱的 OBB，多层均可循环使用。

深度占位判定：

```text
height_delta = baseline_depth - current_depth
```

只有同时满足以下条件才确认：

```text
有效深度比例 >= min_valid_ratio
高度差统计值位于 [min_height_delta_mm, max_height_delta_mm]
达到最小高度差的像素覆盖率 >= min_coverage_ratio
连续满足 occupied_confirm_frames 帧
```

## 输出示例

```json
{
  "layer": 3,
  "max_layers": 4,
  "state": "LAYER_3_FILLING",
  "completed_layers": [1, 2],
  "occupied_count": 1,
  "slot_count": 4,
  "next_slot_id": "P1",
  "next_slot_key": "L3:P1",
  "layer_complete": false,
  "stack_complete": false,
  "slots": [
    {
      "slot_id": "P3",
      "slot_key": "L3:P3",
      "occupied": true,
      "visible_mask": false,
      "depth": {
        "height_delta_mm": 146.0,
        "coverage_ratio": 0.88,
        "valid_ratio": 0.96
      }
    }
  ]
}
```
