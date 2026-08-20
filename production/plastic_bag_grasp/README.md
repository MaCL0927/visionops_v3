# plastic_bag_grasp production

透明塑料包装袋俯视抓取生产任务。当前生产定义已经收敛为一个非常简单、稳定的目标：

1. detection 模型检测**整个 `plastic_bag` 包装目标**；
2. 机器人抓取点固定取检测框中心；
3. `center_px=[u,v]` 始终返回原始 RGB 图像坐标；
4. RGB-D 有效时同时通过 D2C 小 ROI 采样返回 `position_camera=[X,Y,Z]`（mm）；
5. 深度无效不否定 RGB 检测，`position_camera` 返回 `[0,0,0]`，机器人仍可使用像素标定。

当前模型目录默认：

```text
models/rk3576-252_plastic_bag_grasp_det_20260819_094048
```

## 服务端口

- Runtime: `127.0.0.1:28088`
- App: `127.0.0.1:19214`
- Collector/Web: `0.0.0.0:18097`
- Robot WebSocket JSON: `0.0.0.0:9001/vision`

WebSocket 端口和 `items[]` 核心字段与 `tube_pick_vision`、`detergent_grasp` 保持一致。机器人侧建议按 `request_id` 关联一次 trigger 与对应 detection，避免把连续预览结果误当成当前抓取周期结果。

## FPS 路径

本任务保留已经验证过的生产性能优化：

- 推理线程 + 后处理线程双线程流水线；
- 连续结果使用容量 1 的 latest-only 队列；
- 显式 `trigger/request_id` 结果可靠保留，不允许被连续帧覆盖；
- 本机 App→Runtime 使用 raw HTTP socket：`TCP_NODELAY`、header/body 一次 `sendall`、raw response bytes 入队，JSON 在后处理线程解析；
- WebSocket **没有独立固定 5 Hz 推送节流**，每完成一个有效结果就推送一次；
- `production_inference_fps` / `detection_fps` 只作为推理上限，默认 30 FPS；
- Orbbec 336L 优先读取共享 RGB/Depth；深度只围绕目标中心做小区域采样，不搬运整张 depth PNG；
- 生产环境建议 `VISIONOPS_RUNTIME_HTTP_WORKERS=1`，这是当前 RK3576 已验证的稳定设置。

## 目录

```text
production/plastic_bag_grasp/
├── config.py
├── config/line.yaml
├── launcher.py
├── deploy/
├── scripts/
└── tasks/plastic_bag_grasp_vision/
    ├── algorithm.py
    ├── depth_coordinate.py
    ├── service.py
    ├── websocket_server.py
    ├── mock_robot_client.py
    ├── PROTOCOL.md
    └── README.md
```

机器人协议见 `tasks/plastic_bag_grasp_vision/PROTOCOL.md`。
