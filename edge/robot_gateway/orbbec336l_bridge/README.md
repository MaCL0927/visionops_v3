# VisionOps Orbbec Gemini 336L SDK Bridge

本目录提供 Orbbec Gemini 336L SDK HTTP Bridge 的源码与 systemd 安装脚本。

本版新增：

- `GET /stream/profiles`：从 Orbbec SDK 实时枚举 Color / Depth 支持的 `(width, height, fps, format)` 组合。
- Collector Web 设置 API 可读取该 profile 列表，写入 `orbbec336l_bridge.env` 并重启 `visionops-orbbec336l-bridge.service`。

安装/更新：

```bash
cd /opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl restart visionops-orbbec336l-bridge.service
```

检查：

```bash
curl -s http://127.0.0.1:18182/stream/profiles | python3 -m json.tool
curl -s http://127.0.0.1:18182/stream/status | python3 -m json.tool
```
