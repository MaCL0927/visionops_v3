# box_grasp_vision

该任务位于 `production/carton_palletizing` 下，但与多层托盘摆放任务相互独立。它使用机器人眼睛位置的 Orbbec 336L 倾斜俯视画面和 segmentation 模型，从纸箱 mask 中计算：

- 外轮廓；
- 四个透视角点；
- 中心点；
- 左右两条边的中点（抓取点）；
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
runtime_http_ms
runtime_internal_ms
runtime_transport_overhead_ms
classify_ms
depth_sample_deproject_ms
result_build_ms
postprocess_stage_ms
pipeline_age_ms
total_ms
```

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
