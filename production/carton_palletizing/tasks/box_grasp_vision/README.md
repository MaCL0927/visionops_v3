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
