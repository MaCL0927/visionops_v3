# 洗衣液抓取独立生产任务（M32.8.1）

该任务与 `carton_line`、`carton_palletizing` 完全独立，目录、模型、Runtime、App、Collector 和 systemd 服务均单独配置。

## 目录与端口

| 组件 | 默认地址/路径 |
|---|---|
| 模型包 | `models/detergent_grasp/current/{model.rknn,model.yaml}` |
| Runtime | `http://127.0.0.1:28087` |
| App HTTP | `http://127.0.0.1:19212` |
| Collector Web | `http://<视觉盒IP>:18096` |
| Robot WebSocket | `ws://<视觉盒IP>:9001/vision` |
| 原始 MJPEG | HP60C 默认 `http://<视觉盒IP>:18181/stream.mjpeg` |

`9001` 沿用此前视觉盒协议。通常该任务部署在独立视觉盒上；若要与其他 WebSocket 任务在同一盒子同时运行，需要在 `/etc/visionops_v3/detergent_grasp.yaml` 修改端口。

## 模型包

```text
/opt/visionops_v3/models/detergent_grasp/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 的 task 必须为 `obb`。当前默认类别映射是：

```text
0=big, 1=head, 2=box, 3=small
```

如果实际模型的 ID 不同，修改 `algorithm.classes`；类别名匹配优先于 ID。

## 首次部署

```bash
cd /opt/visionops_v3
sudo bash production/detergent_grasp/deploy/install_services.sh
```

然后在 Collector Web 的“设置 → 相机设置”选择 `hp60c` 并应用。相机切换会同时重启本任务 Runtime 和 App。

检查：

```bash
systemctl status visionops-v3-detergent-grasp-runtime.service
systemctl status visionops-v3-detergent-grasp-app.service
systemctl status visionops-v3-detergent-grasp-collector.service

curl -s http://127.0.0.1:28087/api/runtime/status | python3 -m json.tool
curl -s http://127.0.0.1:19212/api/app/status | python3 -m json.tool
curl -s -X POST http://127.0.0.1:19212/api/app/trigger \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"detergent_grasp","request_id":"test-001"}' | python3 -m json.tool
```

模拟机器人：

```bash
/opt/visionops_v3/venv/bin/python3 -m \
  production.detergent_grasp.tasks.detergent_grasp_vision.mock_robot_client \
  --url ws://127.0.0.1:9001/vision
```

## 视频与生产画面

- 机器人报文中的 `video_url` 指向 Camera Bridge 原始 MJPEG；
- Collector 生产模式使用 App 的 `visualization_result`，直接显示 Runtime 的 OBB 检测结果；
- App 每完成一次实际推理就推送一次，不再配置独立的 5 Hz 推送频率。

机器人字段见 `tasks/detergent_grasp_vision/PROTOCOL.md`。生产画面还会从瓶体中心绘制红色箭头，箭头指向 `angle_deg` 所表示的把手方向；黄色点是瓶体中心，青色点是抓取点。


## 瓶体 360° 把手方向

机器人协议字段名 `angle_deg` 保持不变。瓶体角度不再只是 OBB 的 `[-90,90]` 无向长轴，而是根据匹配的 `head` 抓取点判断正反方向：沿长轴距离抓取点更远的一端作为把手方向，最终输出 `[0,360)`。图像向右为 0°、向下为 90°、向左为 180°、向上为 270°，角度按图像坐标顺时针递增。纸箱没有正反判据，仍返回无向长轴角度。

## M32.8.1：Raw Runtime IPC 与延迟 JSON 解码

现场测试中 urllib 单次 Runtime 请求约 62 ms，而 localhost raw socket 约 37 ms。M32.8.1 保持 HTTP 协议不变，但本机请求使用 `TCP_NODELAY`，并将 header 与请求体一次发送。推理线程只把原始响应 bytes 放入队列，`json.loads` 移到后处理线程。

```yaml
runtime_ipc:
  raw_http_enabled: true
  raw_http_fallback_urllib: true
  max_response_bytes: 33554432
```

通过 `/api/app/status` 检查：

```text
last_app_timing.runtime_transport = raw_socket
pipeline.runtime_ipc.raw_request_count 持续增加
pipeline.runtime_ipc.urllib_request_count = 0（正常本机运行时）
```

详细说明见 `docs/M32.8.1_RAW_RUNTIME_IPC.md`。

## M32.7 / M32.8：App 计时与双线程流水线

默认生产架构已经改为：

```text
Runtime inference producer
    ↓ capacity=1 latest-only queue
CPU postprocess worker
    ↓ robot JSON / visualization / WebSocket
```

连续检测允许覆盖积压旧帧；带 `request_id` 的显式机器人 trigger 受到保护，不会被连续结果替换。
`/api/app/status` 提供 `last_app_timing`、`app_timing_stats`、`latency_ms.p50/p95` 和 `pipeline` 状态。

```bash
python3 tools/benchmark_detergent_app.py \
  --app-url http://127.0.0.1:19212 \
  --count 60 --interval 1 \
  --output /tmp/detergent_app_pipeline.json
```

配置：

```yaml
pipeline:
  enabled: true
  result_queue_size: 1
  max_result_age_ms: 500
```
