# VisionOps v3 Orbbec 336L + HP60C 双 SDK Bridge 接入说明

版本：M25.3

## 1. 目标

同一台 LB3576 同时连接 Orbbec Gemini 336L 和 HP60C / HP60CN。两款相机由各自厂商 SDK 独立采集，避免端口、设备句柄和配置文件冲突。

| 相机 | Bridge 服务 | HTTP 端口 | 默认地址 |
|---|---|---:|---|
| Orbbec Gemini 336L | `visionops-orbbec336l-bridge.service` | 18182 | `http://127.0.0.1:18182` |
| HP60C / HP60CN | `visionops-hp60c-sdk-bridge.service` | 18181 | `http://127.0.0.1:18181` |

两款 Bridge 可以同时运行，但 VisionOps Runtime、采集、模型验证和生产任务统一使用 `config/active_camera.json` 中选中的一款相机。

## 2. 图像切换原理

Web 页面不直接访问 18181 或 18182，而是始终请求当前 Runtime：

```text
浏览器
  -> Collector /api/runtime/snapshot.jpg
  -> 当前 Runtime /api/runtime/snapshot.jpg
  -> active_camera.json 选中的 Camera Bridge
```

在“设置 → 相机设置”中选择型号并保存后，后端执行：

1. 写入对应 Bridge 的 env；
2. 重启并等待所选 Bridge 产生新 RGB/Depth 帧；
3. Bridge 健康后更新 `config/active_camera.json`；
4. 重启所有正在运行的 Runtime；
5. 重启依赖相机配置的产线服务，例如 `visionops-v3-ws-pick.service`、Robot Gateway 或托盘任务 App；
6. 采集、模型验证和生产页面继续访问原有 Runtime URL，显示内容自动切换到新相机。

如果所选 Bridge 无法启动或没有新 RGB/Depth 帧，选择文件不会切换，避免 Web 被切换到无效图像源。

## 3. 统一 HTTP 接口

HP60C Bridge 与 336L 尽量保持相同接口：

```text
GET  /health
GET  /stream/profiles
GET  /stream/camera_info
GET  /stream/snapshot.jpg
GET  /stream/depth.png
GET  /stream/depth_vis.jpg
GET  /stream/depth_meta
GET  /stream.mjpeg
POST /api/coordinate/deproject
```

HP60C 示例：

```bash
curl -s http://127.0.0.1:18181/health | python3 -m json.tool
curl -o /tmp/hp60c.jpg http://127.0.0.1:18181/stream/snapshot.jpg
curl -o /tmp/hp60c_depth.png http://127.0.0.1:18181/stream/depth.png
```

336L 地址只需将端口改为 `18182`。

## 4. HP60C 参数

配置文件：

```text
/opt/visionops_v3/edge/camera_bridge/hp60c_bridge/hp60c_sdk_bridge.env
```

主要参数：

```bash
VISIONOPS_HP60C_HTTP_HOST=0.0.0.0
VISIONOPS_HP60C_HTTP_PORT=18181
VISIONOPS_HP60C_CONFIG=/path/to/hp60c_configEncrypt.json
VISIONOPS_HP60C_JPEG_QUALITY=85
VISIONOPS_HP60C_MJPEG_FPS=10
VISIONOPS_HP60C_FLIP_VERTICAL=true
VISIONOPS_HP60C_FLIP_HORIZONTAL=false
VISIONOPS_HP60C_RGB_SOURCE=auto
VISIONOPS_HP60C_RGB_ORDER=bgr
```

HP60C 的曝光、真实 RGB/Depth profile 等由 Angstrong 加密配置文件控制。Web 中显示的分辨率和 FPS 用于说明/校验当前输出，不会改写厂商加密配置内部参数。

## 5. 深度与三维反投影

HP60C Bridge 将 SDK 回调中的深度缓冲保存为 16 位 PNG，并通过 `/stream/depth.png` 发布。深度元数据由 `/stream/depth_meta` 提供。

`/api/coordinate/deproject` 使用 HP60C 彩色相机内参：

```bash
VISIONOPS_HP60C_FX=0
VISIONOPS_HP60C_FY=0
VISIONOPS_HP60C_CX=0
VISIONOPS_HP60C_CY=0
```

内参为 0 时，原始 RGB/Depth 获取仍可用，但反投影接口返回 503，防止使用虚构标定参数。可在 Web“相机设置”中录入经 HP60C 标定工具确认的原始相机 `fx/fy/cx/cy`。Bridge 会对 RGB 和 Depth 应用完全相同的水平/垂直翻转，并在反投影前把显示像素还原到未翻转的传感器像素，因此无需手工修改主点符号。

RGB 与 Depth 是否已经对齐由厂商配置决定，配置项：

```bash
VISIONOPS_HP60C_DEPTH_ALIGNED_TO_COLOR=true
```

只有确认厂商输出确实完成 D2C 对齐后，才能直接使用 RGB 检测中心读取对应深度。

## 6. HP60C 断线重连

HP60C Bridge 具备与 336L 同级别的恢复链路：

```text
RGB 或 Depth 超时
  -> 清除旧图缓存
  -> snapshot/depth 返回 503，MJPEG 断开
  -> 停止并释放旧 SDK listener / camera handle
  -> 重新初始化 SDK、重新枚举 USB、重新打开相机
  -> 等待 RGB + Depth 首帧
  -> 恢复 running
```

相关参数：

```bash
VISIONOPS_HP60C_STALE_TIMEOUT_MS=3000
VISIONOPS_HP60C_FIRST_FRAME_TIMEOUT_MS=8000
VISIONOPS_HP60C_RECONNECT_INITIAL_MS=1000
VISIONOPS_HP60C_RECONNECT_MAX_MS=30000
VISIONOPS_HP60C_RECONNECT_FAILURE_ALARM_SEC=15
```

外部兜底：

```text
visionops-hp60c-sdk-bridge-watchdog.timer
visionops-hp60c-sdk-bridge-watchdog.service
```

watchdog 只在 HP60C Bridge unit 已安装并启用/运行时工作。未安装 HP60C Bridge 不会累计失败次数，也不会触发视觉盒重启。

## 7. 安装

先覆盖 M25.3 文件，再安装 HP60C Bridge：

```bash
cd /opt/visionops_v3
sudo bash edge/camera_bridge/hp60c_bridge/install_hp60c_sdk_bridge_service.sh
sudo systemctl restart visionops-hp60c-sdk-bridge.service
sudo systemctl enable --now visionops-hp60c-sdk-bridge-watchdog.timer
```

安装脚本会停止旧的 `visionops-hp60c-ros1-bridge.service`，防止两个进程同时占用 HP60C。

重新安装当前产线 profile，使 Runtime/watchdog unit 获得双相机依赖和相机选择配置：

```bash
sudo bash production/carton_line/deploy/install_services.sh --profile tube-pick
```

确认两个 Bridge：

```bash
systemctl status visionops-orbbec336l-bridge.service --no-pager
systemctl status visionops-hp60c-sdk-bridge.service --no-pager
curl -s http://127.0.0.1:18182/health | python3 -m json.tool
curl -s http://127.0.0.1:18181/health | python3 -m json.tool
```

## 8. Web 切换验收

1. 两个 Bridge 均处于 `camera_connected=true`；
2. 打开“设置 → 相机设置”；
3. 选择 HP60C，填写配置文件、翻转、RGB 数据源等参数；
4. 点击保存；
5. 检查：

```bash
cat /opt/visionops_v3/config/active_camera.json
curl -s http://127.0.0.1:28083/api/runtime/status | python3 -m json.tool
```

6. 采集预览、模型验证和生产画面应切换为 HP60C；
7. 再选择 336L 保存，画面应切回 336L；
8. `active_camera.json` 中应对应显示 `hp60c` 或 `orbbec336l`。

## 9. 覆盖范围

统一相机选择已接入：

- carton_line partition/tube/pick Runtime；
- tube_pick_vision RGB-D / WebSocket 服务；
- carton_palletizing Runtime 与 RGB-D App；
- Collector 采集、模型验证、生产画面；
- Pick Runtime watchdog。

两款相机仍保留各自独立端口和独立 watchdog，不互相覆盖。
