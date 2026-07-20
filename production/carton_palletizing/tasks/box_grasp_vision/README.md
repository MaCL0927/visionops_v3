# box_grasp_vision

该任务位于 `production/carton_palletizing` 下，但与多层托盘摆放任务相互独立。它使用机器人眼睛位置的 Orbbec 336L 倾斜俯视画面和 segmentation 模型，从纸箱 mask 中计算：

- 外轮廓；
- 四个透视角点；
- 中心点；
- 左右两条边中点向箱体中心内缩后的抓取点；
- 上述 7 个点的相机三维坐标。

## 模型目录

```text
/opt/visionops_v3/models/carton_box_grasp/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 中应为：

```yaml
task_type: segmentation
labels:
  - id: 0
    name: box
```

Runtime 必须真正输出 `mask.source=proto` 的多边形。默认配置会拒绝 `bbox_fallback`，因为水平框无法表达倾斜视角下的纸箱透视边缘。

## 手动启动

```bash
cd /opt/visionops_v3
./production/carton_palletizing/scripts/start_box_grasp_runtime.sh
./production/carton_palletizing/scripts/start_box_grasp_app.sh
./production/carton_palletizing/scripts/start_box_grasp_collector.sh
```

默认端口：

- Runtime：28085；
- HTTP App：19211；
- WebSocket：9001 `/vision`；
- Collector Web：18095；
- 336L MJPEG：18182 `/stream.mjpeg`。

## systemd

```bash
sudo bash production/carton_palletizing/deploy/install_box_grasp_services.sh
```

首次部署或升级后，应修改 `/etc/visionops_v3/carton_palletizing.yaml` 中：

```yaml
box_grasp:
  video:
    public_url: http://视觉盒实际IP:18182/stream.mjpeg
```

机器人报文采用统一抓取点结构：`items[]` 中每一项代表一个抓取点。一个纸箱会输出两项，两项使用相同 `id/class_id/confidence`，分别携带各自的 `position_camera` 和 `center_px`。该字段结构与 `tube_pick_vision` 一致，区别仅在同一目标 ID 对应的抓取点数量。

协议详见 [PROTOCOL.md](PROTOCOL.md)。

## FPS 第二阶段优化

本任务的生产主链路已改为两级流水线：

```text
线程 1：Runtime / RKNN 推理帧 N+1
线程 2：帧 N 的 mask 几何、深度采样和三维反投影
```

两级之间使用有界最新结果队列，默认容量为 1。CPU 后处理跟不上时覆盖旧的连续
检测结果，不积压历史帧；机器人显式 `trigger` 请求不会被连续结果覆盖。

配置：

```yaml
box_grasp:
  pipeline:
    enabled: true
    result_queue_size: 1
    max_result_age_ms: 500
```

深度链路默认使用 Orbbec Bridge 的合并接口：

```yaml
box_grasp:
  algorithm:
    depth:
      use_sample_deproject: true
```

该接口直接在 Bridge 的 D2C 深度缓存中采样 7 个小 ROI 并完成 SDK 反投影，正常
生产路径不再请求 `/stream/depth.png`，也不再在 Python 中执行整图 PNG 解码。
旧 PNG 路径仍可通过把 `use_sample_deproject` 设为 `false` 临时回退。

每个 `app_decision` 及 `visualization_result.box_grasp` 中新增 `app_timing`：

```text
runtime_http_ms / runtime_roundtrip_ms
runtime_lock_wait_ms
runtime_headers_wait_ms
runtime_body_read_ms
runtime_json_decode_ms
runtime_response_bytes
runtime_server_queue_ms
runtime_server_route_ms
runtime_internal_ms
runtime_transport_overhead_ms
runtime_non_route_ms
classify_ms
depth_sample_deproject_ms
depth_bridge_internal_ms
depth_http_roundtrip_ms
depth_http_headers_wait_ms
depth_http_body_read_ms
depth_json_decode_ms
result_build_ms
postprocess_stage_ms
pipeline_age_ms
total_ms
```


`runtime_http_ms` 是 App 客户端从发起请求到读完响应正文的完整往返时间。
优化后 JSON 解码已移到 CPU 后处理线程，因此该字段不再包含 `json.loads`；
`runtime_server_queue_ms` 和 `runtime_server_route_ms` 来自 Runtime 响应头，可用于区分
HTTP 工作队列等待与实际路由处理。Runtime 默认启动 4 个 HTTP worker，但通过
`inference_mutex_` 保证单个 RKNN context 仍只执行一路推理。

状态接口同时给出流水线和最近一次分阶段耗时：

```bash
curl -s http://127.0.0.1:19211/api/app/status | jq '{
  configured_detection_fps,
  detection_fps,
  last_latency_ms,
  last_app_timing,
  pipeline,
  counters
}'
```

部署后应确认快速深度接口生效：

```bash
curl -s http://127.0.0.1:18182/stream/status | jq '{
  sample_deproject_count,
  sample_deproject_ms_latest,
  sample_deproject_ms_average
}'
```

画面中有目标时，`sample_deproject_count` 应持续增长，而生产检测过程不应频繁访问
或编码整幅 `depth.png`。



## 统一生产 FPS

Box grasp 不再从 `box_grasp.websocket.detection_hz` 读取固定值，也没有独立的机器人推送频率。
唯一目标值通过以下接口设置：

```bash
curl -s -X POST http://127.0.0.1:19211/api/app/inference_settings \
  -H 'Content-Type: application/json' \
  -d '{"detection_fps":15}' | python3 -m json.tool
```

响应同时提供 `production_inference_fps`、`actual_inference_fps` 和
`push_mode=every_completed_result`。每个成功完成的连续推理结果都会广播给 WebSocket 客户端；
若硬件只能达到 13 FPS，实际推送也自然是约 13 FPS，不会再被 YAML 中的 5 Hz 二次限速。
生产画面读取同一 `latest_decision`，显示 App 的精确设定值和实测值。

## 抓取点内缩与深度稳定性

左右抓取点不再直接使用 mask 四边形左右边的原始中点。算法先计算左右边中点，
再沿两点连线向纸箱中心内缩，避免 segmentation 轮廓轻微外扩或抖动时抓取点落到
箱体外侧：

```yaml
box_grasp:
  algorithm:
    geometry:
      # 0=保持边中点，0.18=从边中点向中心移动 18%
      grasp_inward_ratio: 0.18
    depth:
      # 深度采样位置在实际抓取点基础上再额外向中心移动 5%
      grasp_extra_inward_ratio: 0.05
```

推荐先使用 `grasp_inward_ratio=0.18`，现场可在 `0.12～0.25` 范围调整。数值越大，
两个抓取点越靠近纸箱中心。`grasp_extra_inward_ratio` 只改变深度采样位置，Bridge
仍将该深度反投影到实际抓取点坐标，因此机器人收到的 `center_px` 与
`position_camera` 对应的是同一个内缩后的抓取点。

Collector 可视化中的左右抓取点以及 WebSocket `items[].center_px` 会同步使用新位置。
调试结果中的 `grasp_geometry.edge_midpoints_px` 保留原始左右边中点，便于比较内缩量。

## FPS 第三阶段：原始 RGB 与 RKNN 缓冲复用

Orbbec 336L 场景默认启用：

```yaml
camera_bridge:
  shared_rgb_enabled: true
  shared_rgb_name: /visionops_orbbec336l_rgb
  shared_rgb_fallback_http: true
```

Launcher 会为 Orbbec Runtime 选择 `frame_source=shared_memory`。相机切换到 HP60C 时仍
选择 `hp60c_bridge`，不改变 HP60C 现有链路。

这一阶段删除 Runtime 每帧 JPEG 下载和 OpenCV 解码，并减少 RKNN 输入、输出的 host
内存申请与复制。验证时同时观察：

```bash
curl -s http://127.0.0.1:18182/stream/status | jq '{
  shared_rgb_ready,
  shared_rgb_publish_count,
  shared_rgb_publish_ms_average
}'

curl -s http://127.0.0.1:28085/api/runtime/status | jq '.frame_source'

curl -s -X POST http://127.0.0.1:28085/api/runtime/infer_once | jq '{
  timing,
  timing_detail,
  debug: {
    host_input_copy_avoided: .debug.host_input_copy_avoided,
    output_buffers_preallocated: .debug.output_buffers_preallocated,
    output_view_bytes: .debug.output_view_bytes
  }
}'
```

`decode_ms` 应降为 0；生产模式的最终吞吐仍由 `rknn_run_ms`、分割后处理和 App
流水线中的最慢阶段决定。
