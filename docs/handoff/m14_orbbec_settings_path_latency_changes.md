# M14 Orbbec 设置 API 路径与应用耗时修正

## 背景

上一版 Orbbec 设置 API 默认写入旧路径：

- `/opt/visionops/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env`

但 VisionOps v3 的 Orbbec Bridge 独立路径为：

- `/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env`

因此本次将 Collector Web 设置 API 与 Orbbec Bridge install 脚本统一到 v3 路径。

## 修改

### 1. 默认 env 路径改为 v3

`apps/collector_web/backend/sdk_bridge_settings.py`

- 默认 env：`/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env`
- 仍支持环境变量覆盖：`VISIONOPS_ORBBEC336L_BRIDGE_ENV`
- 服务名仍为：`visionops-orbbec336l-bridge.service`
- 服务名可通过 `VISIONOPS_ORBBEC336L_SERVICE` 覆盖

### 2. Orbbec Bridge 安装脚本改为 v3 路径

`edge/robot_gateway/orbbec336l_bridge/install_orbbec336l_bridge_service.sh`

- `DST_DIR=/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge`
- `BIN_DIR=/opt/visionops_v3/bin`
- systemd `WorkingDirectory` 和 `EnvironmentFile` 均指向 v3 路径

### 3. 取消 env 自动备份

保存设置时不再生成：

- `orbbec336l_bridge.env.bak.*`

返回中固定：

```json
"backup_path": null,
"backup_enabled": false
```

### 4. 优化保存耗时

上一版 POST 保存时会重新访问 `/stream/profiles` 并在返回前再次枚举 profile。SDK profile 枚举可能较慢，且设置页打开时已经完成过枚举，因此本版改为：

- GET 设置时：从 `/stream/profiles` 枚举 SDK profile；
- POST 保存时：前端把已枚举的 `known_profiles` 一并提交；
- 后端用 `known_profiles` 校验 RGB/Depth profile；
- 保存后不再重复访问 `/stream/profiles`；
- 保存后只做短 `/health` 检查。

POST 返回新增：

```json
"apply_timings_ms": {
  "read_env_ms": 0,
  "profile_validation_ms": 0,
  "write_env_ms": 0,
  "restart_service_ms": 0,
  "wait_health_ms": 0,
  "systemd_status_ms": 0,
  "total_apply_ms": 0
}
```

用于定位实际耗时步骤。

## 验证建议

```bash
curl -s http://127.0.0.1:18091/api/settings/sdk_bridge/orbbec336l | python3 -m json.tool
```

确认：

- `env_path` 为 `/opt/visionops_v3/.../orbbec336l_bridge.env`
- `profiles.source` 为 `bridge_api` 时代表 SDK 实时枚举成功

保存后确认：

```bash
cat /opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env
ls /opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/*.bak* 2>/dev/null
systemctl cat visionops-orbbec336l-bridge.service | grep -E 'WorkingDirectory|EnvironmentFile|ExecStart'
```

如 systemd 仍指向 `/opt/visionops/...`，需要重新安装 v3 版 Bridge：

```bash
cd /opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl daemon-reload
sudo systemctl restart visionops-orbbec336l-bridge.service
```
