# tube_pick_vision

该任务把 RK3576 视觉盒作为“外部推理盒子”接入机器人箱中取物模块。

## 任务边界

- RGB 与 D2C 深度固定为 `640×480`；
- detection 模型：`class_id=0` 正常纸筒产品，`class_id=1` 大隔板，`class_id=2` 倒伏纸筒 `lying`；
- 模型始终对完整图像推理；ROI 由 VisionOps Web 设置，并在 Runtime 统一后处理阶段过滤；
- 产品、隔板和倒伏纸筒都以检测框中心取深度；
- 通过 Orbbec SDK 将 `[u,v,depth_mm]` 反投影成彩色相机坐标系 `[X,Y,Z]`；
- 深度无效时返回 `[0,0,0]`；
- 不做手眼标定，不输出 `base_link`；机器人后端负责后续坐标变换。

## 服务角色

视觉盒是 WebSocket Server：

```text
ws://盒子IP:9001/vision
```

机器人后端是 WebSocket Client。WebSocket 只传 JSON 文本帧：

- 盒子 → 机器人：`detection`、`status`、`ack`；
- 机器人 → 盒子：`control`；
- 心跳使用 WebSocket 原生 Ping/Pong。

原始 RGB 视频使用 MJPEG 软同步：

```text
http://盒子IP:18182/stream.mjpeg
```

视频只用于显示和标定观察；抓取逻辑使用 `detection.items[].position_camera`。检测到
`class_id=2` 时，视觉盒按普通目标返回完整坐标和置信度，机器人系统负责告警及后续动作。

## 控制命令

```json
{"type":"control","command":"start","request_id":1}
{"type":"control","command":"stop","request_id":2}
{"type":"control","command":"trigger","request_id":3}
```

`trigger` 必须带非空 `request_id`。对应检测消息会原样返回该字段。连续推送的检测消息不带 `request_id`。

ROI 和置信度不接受机器人侧动态下发，统一在 VisionOps Web / 模型配置中管理。

## 启动

```bash
cd /opt/visionops_v3

./production/carton_line/scripts/start_runtime.sh pick
./production/carton_line/scripts/start_ws_pick.sh
./production/carton_line/scripts/start_collector.sh pick
```

systemd 安装：

```bash
sudo bash production/carton_line/deploy/install_services.sh --profile tube-pick
```

安装的主要服务：

```text
visionops-v3-runtime-pick.service
visionops-v3-ws-pick.service
visionops-v3-collector-pick.service
visionops-v3-runtime-pick-watchdog.timer
visionops-orbbec336l-bridge-watchdog.timer
```

sudo systemctl start visionops-v3-runtime-pick.service visionops-v3-ws-pick.service visionops-v3-collector-pick.service visionops-v3-runtime-pick-watchdog.timer
visionops-orbbec336l-bridge-watchdog.timer
systemctl status visionops-v3-runtime-pick.service visionops-v3-ws-pick.service visionops-v3-collector-pick.service visionops-v3-runtime-pick-watchdog.timer
visionops-orbbec336l-bridge-watchdog.timer


## 配置

现场配置：

```text
/etc/visionops_v3/carton_line.yaml
```

重点字段：

```yaml
pick:
  websocket:
    listen_host: 0.0.0.0
    listen_port: 9001
    path: /vision
    auto_start: true
    detection_hz: 10.0
  video:
    public_url: http://192.168.213.137:18182/stream.mjpeg
  algorithm:
    image: {width: 640, height: 480, require_fixed_size: true}
    classes:
      product_ids: [0]
      separator_ids: [1]
      lying_ids: [2]
      lying_names: [lying]
      lying_min_confidence: 0.50
```

同时确认 336L Bridge 实际环境文件使用：

```bash
VISIONOPS_ORBBEC336L_HTTP_HOST=0.0.0.0
VISIONOPS_ORBBEC336L_COLOR_WIDTH=640
VISIONOPS_ORBBEC336L_COLOR_HEIGHT=480
VISIONOPS_ORBBEC336L_DEPTH_WIDTH=640
VISIONOPS_ORBBEC336L_DEPTH_HEIGHT=480
```

## 调试

状态：

```bash
curl -s http://127.0.0.1:19130/api/app/status | python3 -m json.tool
```

本机手动触发：

```bash
curl -s -X POST http://127.0.0.1:19130/api/app/evaluate_once \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"manual-1"}' | python3 -m json.tool
```

cd /opt/visionops_v3

python3 -m \
  production.carton_line.tasks.tube_pick_vision.mock_robot_client \
  --url ws://127.0.0.1:9001/vision

模拟机器人客户端：

```bash
python3 -m production.carton_line.tasks.tube_pick_vision.mock_robot_client \
  --url ws://127.0.0.1:9001/vision
```
python3 -m \
  production.carton_line.tasks.tube_pick_vision.mock_robot_client \
  --url ws://192.168.213.137:9001/vision

详细协议见 [PROTOCOL.md](PROTOCOL.md)。新增异常类别的机器人侧对接说明见
[LYING_CLASS_INTEGRATION.md](LYING_CLASS_INTEGRATION.md)。

## USB 相机断线恢复

Bridge 会在 RGB/Depth 超过 3 秒未更新时清除旧缓存、断开 MJPEG、完整重建 Orbbec Pipeline 并指数退避重连。`tube_pick_vision` 在恢复期间只发送空 `items` 和相机告警，不会继续发送旧检测结果。独立 systemd watchdog 仅在 SDK 恢复线程卡死或 HTTP 不可访问时重启 Bridge。

未来 PLC Modbus-TCP 告警已预留稳定故障码与配置段，但当前未实现寄存器通信。详细字段见 [PROTOCOL.md](PROTOCOL.md)。
