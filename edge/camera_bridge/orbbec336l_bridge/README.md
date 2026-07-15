# VisionOps Orbbec Gemini 336L SDK Bridge

本目录提供 Orbbec Gemini 336L SDK HTTP Bridge 的源码与 systemd 安装脚本。

本版新增：

- `GET /stream/profiles`：从 Orbbec SDK 实时枚举 Color / Depth 支持的 `(width, height, fps, format)` 组合。
- Collector Web 设置 API 可读取该 profile 列表，写入 `orbbec336l_bridge.env`（由 `orbbec336l_bridge.env.example` 初始化） 并重启 `visionops-orbbec336l-bridge.service`。

安装/更新：

```bash
cd /opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl restart visionops-orbbec336l-bridge.service
```

检查：

```bash
curl -s http://127.0.0.1:18182/stream/profiles | python3 -m json.tool
curl -s http://127.0.0.1:18182/stream/status | python3 -m json.tool
```

## Tube-pick 相机三维坐标接口

`tube_pick_vision` 固定 RGB/Depth 为 640×480，并启用 D2C。Bridge 新增：

```text
GET  /stream/camera_info
POST /api/coordinate/deproject
```

批量反投影请求：

```json
{"points":[[320.0,240.0,1260.0],[410.0,260.0,1185.0]]}
```

响应：

```json
{
  "ok": true,
  "coordinate_frame": "color_camera",
  "unit": "mm",
  "points": [
    {"valid":true,"position_camera":[0.0,0.0,1260.0]},
    {"valid":true,"position_camera":[175.0,40.0,1185.0]}
  ]
}
```

内部调用 Orbbec SDK `CoordinateTransformHelper::calibration2dTo3d()`。深度为 0 或转换失败时返回 `[0,0,0]`。

机器人需要读取 MJPEG 时，实际 env 文件必须设置：

```bash
VISIONOPS_ORBBEC336L_HTTP_HOST=0.0.0.0
VISIONOPS_ORBBEC336L_COLOR_WIDTH=640
VISIONOPS_ORBBEC336L_COLOR_HEIGHT=480
VISIONOPS_ORBBEC336L_DEPTH_WIDTH=640
VISIONOPS_ORBBEC336L_DEPTH_HEIGHT=480
```

## 7×24 USB 断线恢复

Bridge 不再把最后一帧无限当作实时画面。RGB 或 D2C Depth 任一路超过
`VISIONOPS_ORBBEC336L_STALE_TIMEOUT_MS` 未更新时：

1. `/health` 切换为 `camera_connected=false`，并给出 `camera_state`、故障码和重连计数；
2. 立即使旧 RGB/Depth 缓存失效；`snapshot.jpg`/`depth.png` 返回 HTTP 503；
3. 已连接的 MJPEG 客户端主动断流，客户端应自动重连；
4. Bridge 完整销毁旧 Pipeline/设备句柄，重新枚举相机、恢复 D2C 和标定参数；
5. 重连使用 1/2/4/8…30 秒指数退避；相机重新插入后自动恢复。

关键环境变量：

```bash
VISIONOPS_ORBBEC336L_STALE_TIMEOUT_MS=3000
VISIONOPS_ORBBEC336L_FIRST_FRAME_TIMEOUT_MS=5000
VISIONOPS_ORBBEC336L_RECONNECT_INITIAL_MS=1000
VISIONOPS_ORBBEC336L_RECONNECT_MAX_MS=30000
VISIONOPS_ORBBEC336L_RECONNECT_FAILURE_ALARM_SEC=15
```

Orbbec SDK 在 USB 异常时若阻塞在 `waitForFrames()` 或 `pipeline->stop()`，独立的
`visionops-orbbec336l-bridge-watchdog.timer` 会检测恢复线程长期无进展并重启进程。
它是 oneshot timer，配套 `.service` 平时显示 `inactive (dead)` 属于正常状态。

检查：

```bash
curl -s http://127.0.0.1:18182/health | python3 -m json.tool
systemctl status visionops-orbbec336l-bridge-watchdog.timer
journalctl -t visionops-orbbec-watchdog -n 100 --no-pager
```
