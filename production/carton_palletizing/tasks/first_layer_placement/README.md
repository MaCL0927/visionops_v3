# Multi-Layer Placement（OBB + RGB-D）

目录名为兼容早期第一层版本而保留，实际算法类已经升级为 `MultiLayerPlacementAlgorithm`，并保留 `FirstLayerPlacementAlgorithm` 别名。

## 分层依据

- 第 1 层：纸箱 OBB 与四个 slot 进行几何匹配；
- 第 2 层及以后：使用上一层完成后的深度图作为基准，按 slot 判断新增高度；
- 每层完成：采集稳定深度基准并自动进入下一层；
- 奇数层使用摆放模板 A，偶数层使用错开的摆放模板 B；第 3 层重新使用 A，依次交替。
- 两套模板都基于同一个托盘居中正方形 footprint，因此不会随着层数累积检测框漂移。

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

## 触发式机器人通信

当前 App 同时作为 WebSocket Server，固定监听：

```text
ws://<视觉盒IP>:9001/vision
```

支持两个任务，并兼容数字任务号：

```text
pallet_place_target / 1  M29.2：返回托盘或当前最上层1～4个纸箱的实测位姿
held_box_pose / 2        机器人手持纸箱 OBB 位姿
```

`task_id` 可发送字符串任务名、字符串 `"1"/"2"` 或 JSON 数字 `1/2`。响应中的
`trigger_task_id` 原样回显请求值。

`pallet_place_target` 不再返回 `layer/slot_id/next_slot`，也不推进视觉层状态；摆放策略由机器人侧决定。

初期与后期不需要修改代码，只切换：

```yaml
task:
  communication:
    held_box_selection:
      mode: nearest_depth   # 初期
      # mode: outside_tray  # 后期
```

初期只摆两层时同时设置 `layering.max_layers: 2`；后期可改为目标层数或 `0`。完整报文见 `PROTOCOL.md`。

## 奇偶层模板

```yaml
task:
  algorithm:
    layering:
      next_layer_geometry: layer_template
    template:
      layer_strategy: odd_even
      default_template: odd
      templates:
        odd:   # 模板 A：第1/3/5...层
          template_id: A
          slots: [...]
        even:  # 模板 B：第2/4/6...层
          template_id: B
          slots: [...]
```

模板 B 将模板 A 的四块横竖方向互换，使上下相邻层交错咬合。返回结果顶层的
`template.key/template_id` 以及每个 `slots[]` 的 `template_key/template_id` 可用于现场确认。

## ROI 只能由 VisionOps Web 控制

Runtime 会使用 Web 页面保存到 `data/runtime/roi_carton_palletizing.json` 的 ROI 对原始 OBB
结果进行过滤。机器人 WebSocket `config.detect_region` 会被兼容接收但不会参与
`pallet_place_target` 或 `held_box_pose` 的二次筛选。

```yaml
task:
  communication:
    remote_config:
      roi_control_source: visionops_web_runtime
      allow_confidence_threshold: true
```

诊断字段 `robot_detect_region_applied=false` 表示机器人下发的像素 ROI 未生效。

## status 推送

trigger 模式默认关闭周期 `status`，只发送 trigger 对应的 `detection` 和心跳 `pong`：

```yaml
task:
  communication:
    websocket:
      status_enabled: false
      status_on_connect: false
```

需要恢复状态心跳时将两项改为 `true`。模拟机器人客户端默认也会隐藏 `status`，使用
`--show-status` 才显示。
