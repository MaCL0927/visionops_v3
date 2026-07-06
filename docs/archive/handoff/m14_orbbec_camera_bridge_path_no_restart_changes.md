# M14 Orbbec 设置 API：camera_bridge 路径、无变更跳过重启、日志处理

## 本次修改

1. Orbbec 336L SDK Bridge 代码目录从 `edge/robot_gateway/orbbec336l_bridge` 移到 `edge/camera_bridge/orbbec336l_bridge`。
2. Collector Web 设置 API 默认 env 路径改为 `/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env`。
3. 保存设置时会先比较 env 中相关字段。如果没有任何变化，直接返回 `changed=false`，跳过写 env、跳过 `systemctl restart`、跳过健康检查。
4. Orbbec Bridge service 的 `WorkingDirectory` 改为 `/run/visionops-orbbec336l-bridge`，并增加 `ExecStartPre=/bin/rm -rf .../Log`，避免 Orbbec SDK 在项目目录下持续生成 `Log/OrbbecSDK.log.txt`。
5. Bridge C++ 程序启动时会切换到 `VISIONOPS_ORBBEC336L_RUNTIME_DIR`，默认 `/run/visionops-orbbec336l-bridge`。
6. Bridge C++ signal handler 会关闭 server socket，避免 `accept()` 阻塞导致 `systemctl restart` 等到 TimeoutStopSec。
7. 相机设置页删除每个设置栏下方的大量提示小字，减少界面纵向占用。

## 验证

```bash
cd /opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl daemon-reload
sudo systemctl restart visionops-orbbec336l-bridge.service
systemctl cat visionops-orbbec336l-bridge.service | grep -E "WorkingDirectory|EnvironmentFile|ExecStart|RuntimeDirectory"
```

期望：

```text
WorkingDirectory=/run/visionops-orbbec336l-bridge
EnvironmentFile=/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env
ExecStart=/opt/visionops_v3/bin/visionops_orbbec336l_bridge
```

重复点击 Web 保存且未修改设置时，应显示跳过重启，`restart_service_ms=0`、`wait_health_ms=0`。
