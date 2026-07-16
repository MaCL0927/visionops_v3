# Carton Line 生产方案

该目录管理同一条产线、部署在不同 RK3576 盒子上的两套视觉方案。

## 1. 部署 profile

### `partition-tube`

部署到“纸隔板 + 原纸筒检测”盒子：

```text
partition Runtime :28081
Tube Runtime      :28082
Modbus Gateway    :5046 / HTTP :19090
Partition Web     :18091
Tube Web          :18092
```

安装：

```bash
sudo bash production/carton_line/deploy/install_services.sh --profile partition-tube
```

### `tube-pick`

部署到“箱中取物：产品 / 大隔板”盒子：

```text
HP60C Bridge             :18181
336L Bridge              :18182
Pick Runtime             :28083
External-box WebSocket   :9001/vision
Tube-pick status HTTP    :19130
Pick Web                 :18093
Runtime watchdog timer
```

安装：

```bash
sudo bash production/carton_line/deploy/install_services.sh --profile tube-pick
```

## 2. 目录

```text
production/carton_line/
├── config/line.yaml
├── gateway/                         # partition-tube Modbus 业务
├── tasks/
│   ├── carton_partition_check/
│   ├── carton_tube_check/
│   └── tube_pick_vision/
│       ├── algorithm.py
│       ├── depth_coordinate.py
│       ├── websocket_server.py
│       ├── service.py
│       ├── mock_robot_client.py
│       ├── PROTOCOL.md
│       └── README.md
├── scripts/
│   ├── start_runtime.sh
│   ├── start_collector.sh
│   ├── start_gateway.sh
│   ├── start_ws_pick.sh
│   └── watch_pick_runtime.sh
└── deploy/
```

所有现场配置合并在：

```text
/etc/visionops_v3/carton_line.yaml
```

仓库模板：

```text
production/carton_line/config/line.yaml
```

## 3. 手动启动

### partition-tube 盒子

```bash
cd /opt/visionops_v3
./production/carton_line/scripts/start_runtime.sh partition
./production/carton_line/scripts/start_runtime.sh tube
./production/carton_line/scripts/start_gateway.sh
./production/carton_line/scripts/start_collector.sh partition
./production/carton_line/scripts/start_collector.sh tube
```

### tube-pick 盒子

```bash
cd /opt/visionops_v3
./production/carton_line/scripts/start_runtime.sh pick
./production/carton_line/scripts/start_ws_pick.sh
./production/carton_line/scripts/start_collector.sh pick
```

## 4. tube_pick_vision 最终契约

- detection 模型：0=正常纸筒产品、1=大隔板、2=倒伏纸筒 `lying`；
- RGB 和 D2C Depth 固定 `640×480`；
- 产品、隔板和倒伏纸筒均以检测框中心取深度；
- 336L Bridge 使用 Orbbec SDK 反投影，输出彩色相机坐标 `[X,Y,Z]`，单位 mm；
- 深度无效返回 `[0,0,0]`；
- 盒子是 WebSocket Server：`ws://盒子IP:9001/vision`；
- 机器人是 WebSocket Client；
- 原始视频：`http://盒子IP:18182/stream.mjpeg`，软同步；
- `trigger` 必须携带 `request_id`，对应 detection 原样返回；
- ROI 只在 VisionOps Web 设置，Runtime 统一过滤，机器人不下发 ROI；
- 手眼标定与 `base_link` 变换由机器人系统完成。

详细协议：

```text
production/carton_line/tasks/tube_pick_vision/PROTOCOL.md
```


## 双相机选择

Orbbec 336L（18182）与 HP60C（18181）可以同时连接同一视觉盒。Web“设置 → 相机设置”保存型号后，系统先验证目标 Bridge，再更新 `config/active_camera.json`，随后重启正在运行的 Runtime 和 RGB-D 业务服务。浏览器仍访问 Runtime 的 `snapshot.jpg`，因此采集、模型验证和生产画面统一切换，不需要分别修改各页面 URL。详细安装和验收见 `docs/HP60C_ORBBEC_DUAL_CAMERA_INTEGRATION.md`。

## 5. 336L 配置

`tube-pick` 任务需要对机器人开放 MJPEG，并固定 RGB/Depth 为 640×480。修改实际环境文件：

```text
/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env
```

关键值：

```bash
VISIONOPS_ORBBEC336L_HTTP_HOST=0.0.0.0
VISIONOPS_ORBBEC336L_HTTP_PORT=18182
VISIONOPS_ORBBEC336L_COLOR_WIDTH=640
VISIONOPS_ORBBEC336L_COLOR_HEIGHT=480
VISIONOPS_ORBBEC336L_DEPTH_WIDTH=640
VISIONOPS_ORBBEC336L_DEPTH_HEIGHT=480
VISIONOPS_ORBBEC336L_FPS=30
```

修改 C++ Bridge 后需重新安装编译：

```bash
sudo bash edge/camera_bridge/orbbec336l_bridge/install_orbbec336l_bridge_service.sh
sudo systemctl restart visionops-orbbec336l-bridge.service
```

### 相机故障升级与整机重启

相机异常首先由 Bridge 内部重建 Pipeline；外部 watchdog 在相机持续不可用时重启
`visionops-orbbec336l-bridge.service`。同一个相机故障事件中，如果 Bridge 服务连续重启
10 次后 RGB/Depth 仍未恢复，watchdog 会执行一次：

```bash
systemctl reboot --no-block
```

watchdog 以 root 身份运行，因此等价于人工执行 `sudo reboot`。计数保存在：

```text
/var/lib/visionops_v3/watchdog/
```

只要相机成功恢复一次，失败计数和 reboot 标记都会清零。若重启后相机仍物理断开，
同一故障事件不会再次 reboot，避免无人值守设备陷入循环重启。

配置位于：

```text
/etc/visionops_v3/carton_line.env
```

关键参数：

```bash
VISIONOPS_CAMERA_WATCHDOG_RESTART_WHILE_UNHEALTHY=true
VISIONOPS_CAMERA_WATCHDOG_UNHEALTHY_RESTART_AFTER_S=30
VISIONOPS_CAMERA_WATCHDOG_MAX_SERVICE_RESTARTS=10
VISIONOPS_CAMERA_WATCHDOG_REBOOT_ENABLED=true
VISIONOPS_CAMERA_WATCHDOG_REBOOT_DELAY_S=5
VISIONOPS_CAMERA_WATCHDOG_REBOOT_ONCE_PER_INCIDENT=true
VISIONOPS_CAMERA_WATCHDOG_PERSIST_DIR=/var/lib/visionops_v3/watchdog
```

计划维护或拔相机测试前，可临时停用 timer：

```bash
sudo systemctl stop visionops-orbbec336l-bridge-watchdog.timer
```

测试结束后重新启用：

```bash
sudo systemctl enable --now visionops-orbbec336l-bridge-watchdog.timer
```

## 6. tube-pick 服务

安装后启用：

```text
visionops-v3-runtime-pick.service
visionops-v3-ws-pick.service
visionops-v3-collector-pick.service
visionops-v3-runtime-pick-watchdog.timer
```

启动：

```bash
sudo systemctl start \
  visionops-v3-runtime-pick.service \
  visionops-v3-ws-pick.service \
  visionops-v3-collector-pick.service \
  visionops-v3-runtime-pick-watchdog.timer
```

查看：

```bash
systemctl status visionops-v3-ws-pick.service
journalctl -u visionops-v3-ws-pick.service -f
curl -s http://127.0.0.1:19130/api/app/status | python3 -m json.tool
```

模拟机器人：

```bash
python3 -m production.carton_line.tasks.tube_pick_vision.mock_robot_client \
  --url ws://127.0.0.1:9001/vision
```

## 7. 模型目录

```text
models/carton_partition_check/current/model.rknn
models/carton_partition_check/current/model.yaml
models/carton_tube_check/current/model.rknn
models/carton_tube_check/current/model.yaml
models/tube_pick_vision/current/model.rknn
models/tube_pick_vision/current/model.yaml
```

模型验证 Web 设置的 Pick ROI 保存到：

```text
data/runtime/roi_pick.json
```
