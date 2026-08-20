# plastic_bag_grasp_vision

俯视 RGB-D 场景下检测整个透明塑料包装目标 `plastic_bag`，机器人抓取点固定为 detection bbox 中心。

处理链：

```text
Runtime detection
  -> 过滤 plastic_bag / confidence
  -> 选择最高置信度目标（默认最多 1 个）
  -> bbox center => center_px
  -> 可选 D2C 小 ROI depth sample => position_camera
  -> WebSocket JSON :9001/vision
```

`algorithm.py` 只负责 detection 解析和中心点；`depth_coordinate.py` 优先使用 Orbbec 共享深度，必要时回退 `/api/coordinate/sample_deproject`；`service.py` 保留双线程 latest-only 流水线、显式 trigger 可靠保留、raw local HTTP 和分段计时。

本任务**不再检测或定位塑料袋打结头部**，打结方向不会改变机器人 XY 抓取目标。
