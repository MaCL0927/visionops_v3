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

## 第一阶段帧率优化

本版将 SDK 采集与 JPEG 生产拆为两个线程。相机线程持续消费 RGB/Depth，JPEG
线程只对新 RGB 帧按 `VISIONOPS_ORBBEC336L_MJPEG_FPS` 编码一次，并把同一份
JPEG 缓存提供给 `snapshot.jpg` 和所有 MJPEG 客户端。多个浏览器或 Runtime
连接不会再重复执行 `cv::imencode()`。

MJPEG 调度采用绝对截止时间，JPEG 编码耗时计入目标周期，不再执行“编码耗时 +
完整 sleep 周期”。`/stream/status` 新增以下诊断字段：

```text
capture_fps_configured
capture_fps_measured
mjpeg_fps_configured
mjpeg_fps_measured
jpeg_encode_ms_latest
jpeg_encode_ms_average
jpeg_thread_alive
last_jpeg_age_ms
```

建议在设备上连续观察：

```bash
watch -n 1 'curl -s http://127.0.0.1:18182/stream/status | python3 -m json.tool'
```

其中 `capture_fps_measured` 表示 SDK 实际采集吞吐，`mjpeg_fps_measured` 表示共享
JPEG 缓存的实际生产吞吐。二者和浏览器解码 FPS 是三个不同指标。

## 第二阶段：深度采样与反投影合并接口

`box_grasp_vision` 只需要纸箱四角、中心和两个抓取点共 7 个位置的深度。
为避免每次检测都执行整幅 16-bit Depth PNG 的复制、压缩、HTTP 传输和
Python 解码，本版新增：

```text
POST /api/coordinate/sample_deproject
```

请求中的每个点使用：

```text
[sample_u, sample_v, project_u, project_v]
```

`sample_u/sample_v` 用于在 D2C 深度图的小 ROI 内采样深度，
`project_u/project_v` 使用采样到的深度进行 SDK 三维反投影。这样角点可以向箱体
中心内缩采样，避免读到背景，同时三维坐标仍对应原始角点或抓取点。

示例：

```bash
curl -s -X POST http://127.0.0.1:18182/api/coordinate/sample_deproject \
  -H 'Content-Type: application/json' \
  -d '{
    "points":[[236,222,232,218],[414,246,420,248]],
    "image_width":640,
    "image_height":480,
    "radius_px":4,
    "percentile":50,
    "min_valid_pixels":3,
    "min_depth_mm":100,
    "max_depth_mm":5000,
    "max_depth_age_ms":1500
  }' | python3 -m json.tool
```

Bridge 直接引用当前内存中的 `latest_depth_mm_`，只读取每个点附近的小 ROI，
不会克隆或编码整幅深度图。响应包含：

```text
depth_age_ms
depth_sequence
sample_ms
points[].depth_valid
points[].depth_mm
points[].sample_px
points[].valid_pixels
points[].position_camera
```

`/stream/status` 同时增加：

```text
sample_deproject_count
sample_deproject_ms_latest
sample_deproject_ms_average
```

`/stream/depth.png` 仍保留给多层堆垛、调试和需要整幅深度图的任务使用。

## 第三阶段：Bridge → Runtime 原始 RGB 共享内存

Orbbec 336L 与 Runtime 位于同一台视觉盒时，Runtime 不再逐帧请求
`/stream/snapshot.jpg`、下载 JPEG、执行 `cv::imdecode()` 和 BGR→RGB 转换。Bridge
会把每个新 Color 帧发布到 POSIX 共享内存，Runtime 直接读取 `RGB888`。

默认配置：

```bash
VISIONOPS_ORBBEC336L_SHARED_RGB_ENABLED=true
VISIONOPS_ORBBEC336L_SHARED_RGB_NAME=/visionops_orbbec336l_rgb
```

共享内存使用双缓冲和递增序列号：Bridge 先完整写入非活动缓冲区，再以 release
store 发布 buffer index、时间戳和 sequence；Runtime 在复制前后校验 sequence，避免
读到撕裂帧。HTTP JPEG/MJPEG 接口仍然保留给浏览器、远程调试和旧 Runtime 使用。

检查发布状态：

```bash
curl -s http://127.0.0.1:18182/stream/status | jq '{
  shared_rgb_enabled,
  shared_rgb_name,
  shared_rgb_ready,
  shared_rgb_publish_count,
  shared_rgb_last_publish_age_ms,
  shared_rgb_publish_ms_latest,
  shared_rgb_publish_ms_average,
  shared_rgb_error
}'

ls -lh /dev/shm/visionops_orbbec336l_rgb
```

正常运行时 `shared_rgb_ready=true`、`shared_rgb_publish_count` 持续增长，且
`shared_rgb_last_publish_age_ms` 应维持在一个相机周期附近。相机断线时 Runtime 会按
配置自动回退到 HTTP JPEG，避免共享内存异常导致生产服务完全不可用。

## D2C 深度共享内存

Bridge 默认同时发布 `/visionops_orbbec336l_depth`。该对象使用两个 `uint16` 毫米深度
缓冲区，协议定义在 `interfaces/cpp/visionops_shared_depth.hpp`。Header 固定 256 字节，
包含 sequence、时间戳、D2C 图像尺寸以及已经考虑水平/垂直翻转的有效彩色相机内参。

```bash
VISIONOPS_ORBBEC336L_SHARED_DEPTH_ENABLED=true
VISIONOPS_ORBBEC336L_SHARED_DEPTH_NAME=/visionops_orbbec336l_depth
```

状态检查：

```bash
curl -s http://127.0.0.1:18182/stream/status | jq '{
  shared_depth_ready,
  shared_depth_publish_count,
  shared_depth_last_publish_age_ms,
  shared_depth_publish_ms_average,
  shared_depth_calibration_ready,
  shared_depth_error
}'
ls -lh /dev/shm/visionops_orbbec336l_depth
```

`box_grasp_vision` 只在活动缓冲区上读取七个小 ROI，并在 sequence 改变时重试，不会
复制、编码或传输整幅深度图。共享内存不可用时可按配置回退
`/api/coordinate/sample_deproject`。

Bridge 在发布共享深度前会把彩色内参缩放到实际 D2C 深度缓冲区尺寸，再应用可选的
水平/垂直翻转，因此 Header 中 `fx/fy/cx/cy` 与共享深度像素坐标系一致。共享路径使用
针孔反投影，不包含 SDK 可能执行的额外畸变校正；生产启用前应与
`/api/coordinate/sample_deproject` 做中心和边缘点误差核对。
