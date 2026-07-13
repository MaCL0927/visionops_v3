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

内部调用 Orbbec SDK `CoordinateTransformHelper::transformation2dto3d()`。深度为 0 或转换失败时返回 `[0,0,0]`。

机器人需要读取 MJPEG 时，实际 env 文件必须设置：

```bash
VISIONOPS_ORBBEC336L_HTTP_HOST=0.0.0.0
VISIONOPS_ORBBEC336L_COLOR_WIDTH=640
VISIONOPS_ORBBEC336L_COLOR_HEIGHT=480
VISIONOPS_ORBBEC336L_DEPTH_WIDTH=640
VISIONOPS_ORBBEC336L_DEPTH_HEIGHT=480
```
