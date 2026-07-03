# M14 Orbbec 336L SDK Bridge 设置 API

本次在 M14 设置界面基础上继续接入 Orbbec Gemini 336L 的真实设置 API。

## Collector Web 新增接口

- `GET /api/settings/sdk_bridge/orbbec336l`
  - 读取 `/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env`
  - 查询 `visionops-orbbec336l-bridge.service` 状态
  - 优先从 Orbbec Bridge 的 `GET /stream/profiles` 动态枚举 SDK 支持的 RGB / Depth profile
  - 如果旧版 Bridge 不支持 `/stream/profiles`，回退显示当前 env 中的 profile

- `POST /api/settings/sdk_bridge/orbbec336l`
  - 校验 RGB / Depth profile 格式
  - 如果 profile 来源为 Bridge API，则校验所选 profile 是否在 SDK 支持列表内
  - 写入 Orbbec Bridge env
  - 重启 `visionops-orbbec336l-bridge.service`
  - 轮询 `/health` 确认服务恢复

## 写入的 env 字段

- `VISIONOPS_ORBBEC336L_COLOR_WIDTH`
- `VISIONOPS_ORBBEC336L_COLOR_HEIGHT`
- `VISIONOPS_ORBBEC336L_DEPTH_WIDTH`
- `VISIONOPS_ORBBEC336L_DEPTH_HEIGHT`
- `VISIONOPS_ORBBEC336L_FPS`
- `VISIONOPS_ORBBEC336L_JPEG_QUALITY`
- `VISIONOPS_ORBBEC336L_MJPEG_FPS`
- `VISIONOPS_ORBBEC336L_FLIP_VERTICAL`
- `VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL`
- `VISIONOPS_ORBBEC336L_DEPTH_UNIT`
- `VISIONOPS_ORBBEC336L_SERIAL`

注意：当前 Orbbec Bridge env 只有一个 `VISIONOPS_ORBBEC336L_FPS`，因此 Web 保存时要求 RGB 与 Depth FPS 一致。

## Orbbec Bridge 新增接口

新增源码目录：

```text
edge/robot_gateway/orbbec336l_bridge/
```

新增/更新接口：

- `GET /stream/profiles`

该接口通过 Orbbec SDK 的 StreamProfileList 枚举 Color / Depth 支持的分辨率、帧率与格式组合，返回给 Collector Web 动态填充下拉框。

更新 Bridge：

```bash
cd /opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl restart visionops-orbbec336l-bridge.service
curl -s http://127.0.0.1:18182/stream/profiles | python3 -m json.tool
```

## 权限说明

Collector Web 要真实应用设置，需要具备：

1. 写入 `/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env` 的权限；
2. 执行 `systemctl restart visionops-orbbec336l-bridge.service` 的权限。

如果 Collector 不是 root 运行，需要配置受限 sudo 权限，例如只允许重启这一个 service。
