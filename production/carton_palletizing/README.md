# Carton Palletizing（独立纸箱托盘多层堆垛任务）

该目录是一条独立现场方案，不属于 `production/carton_line/`。当前版本实现第一层 OBB 占位检测，以及第 2、3、4 层和任意更多层的 RGB-D 堆垛状态机。

## 工作流程

1. 使用 OBB 模型检测并锁定托盘，类别固定为 `0=box`、`1=tray`；
2. 以托盘短边构建居中的正方形垛型区域，生成 P1～P4 四块横竖交错摆放掩膜；
3. 第一层根据纸箱 OBB 与 slot 的多边形 IoU、中心距离和朝向确认占位；
4. 第一层放满后采集多帧稳定的 D2C 对齐深度图，作为第二层基准；
5. 第二层及以上逐 slot 计算 `上一层基准深度 - 当前深度`；高度差和覆盖率达到阈值后，确认该位置新增一箱并隐藏掩膜；
6. 当前层放满后重新采集深度基准，自动进入下一层；
7. 下一层优先使用上一层实际检测到的纸箱 OBB 作为摆放掩膜，缺少可靠 OBB 时退回标准 slot；
8. `max_layers` 默认为 4，可改成任意正整数；设为 `0` 时支持持续增加层数，直到人工 reset。

## 模型和相机要求

模型必须是 OBB 模型：

```text
0 = box（纸箱）
1 = tray（托盘）
```

模型目录：

```text
/opt/visionops_v3/models/carton_palletizing/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 应声明：

```yaml
task: obb
class_names: [box, tray]
```

第 2 层及以上依赖 Orbbec 336L Bridge 的 D2C 对齐深度：

```text
http://127.0.0.1:18182/stream/depth.png
```

深度 PNG 必须是 `uint16` 毫米值，并与 RGB 画面空间对齐。

## 多层参数

主要配置位于 `production/carton_palletizing/config/line.yaml`：

```yaml
task:
  algorithm:
    layering:
      max_layers: 4                  # 任意正整数；0=不限层数
      auto_advance: true
      baseline_capture_frames: 3
      baseline_settle_frames: 5
      baseline_stability_mm: 15.0
      use_previous_detected_boxes: true

    depth:
      min_depth_mm: 100
      max_depth_mm: 5000
      slot_roi_shrink_ratio: 0.12
      min_valid_ratio: 0.45
      baseline_min_valid_ratio: 0.55
      min_height_delta_mm: 80.0
      max_height_delta_mm: 600.0
      min_coverage_ratio: 0.55
      height_percentile: 50.0
      occupied_confirm_frames: 3
      occupied_stability_mm: 20.0
```

`min_height_delta_mm=80` 和 `max_height_delta_mm=600` 只是初始值。现场应测量一层纸箱顶面相对上一层的实际深度减小量，再设定合理范围。

## 状态

主要状态包括：

```text
WAIT_TRAY
LAYER_N_FILLING
LAYER_N_WAIT_DEPTH
LAYER_N_CAPTURING_BASELINE
LAYER_N_COMPLETE
STACK_COMPLETE
```

输出同时提供：

- `layer`：当前层；
- `max_layers`：配置层数，0 表示不限；
- `completed_layers`：已经完成的层；
- `layer_complete`：当前层是否完成；
- `stack_complete`：整垛是否达到最大层数；
- `next_slot_id` / `next_slot_key`：下一位置；
- 每个 slot 的 `depth.height_delta_mm`、`coverage_ratio` 和 `valid_ratio`。

## 手动启动

```bash
cd /opt/visionops_v3
./production/carton_palletizing/scripts/start_runtime.sh
./production/carton_palletizing/scripts/start_app.sh
./production/carton_palletizing/scripts/start_collector.sh
```

默认端口：

| 服务 | 地址 |
|---|---|
| RKNN Runtime | `127.0.0.1:28084` |
| 多层堆垛业务应用 | `127.0.0.1:19210` |
| Collector Web | `0.0.0.0:18094` |

生产界面只显示当前层未占用的掩膜；绿色是当前建议位置，黄色是后续空位。层完成时显示深度基准采集进度，达到 `max_layers` 后显示“堆垛完成”。

## 调试接口

```bash
curl http://127.0.0.1:28084/api/runtime/status | python3 -m json.tool
curl http://127.0.0.1:19210/health | python3 -m json.tool

curl -X POST http://127.0.0.1:19210/api/app/evaluate_once \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool

curl http://127.0.0.1:19210/api/app/latest_decision | python3 -m json.tool

# 更换托盘或开始新一垛时，清空全部层数、托盘锁定、深度基准和占位状态
curl -X POST http://127.0.0.1:19210/api/app/reset \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
```

## 开机自启

```bash
cd /opt/visionops_v3
sudo bash production/carton_palletizing/deploy/install_services.sh
```

已有 `/etc/visionops_v3/carton_palletizing.yaml` 时，安装脚本不会覆盖它。需要手动合并新增加的 `camera_bridge.depth_path`、`layering` 和 `depth` 配置。

---

## 机器人眼睛视角纸箱抓取点任务

M25.3 新增第二个独立子任务：

```text
production/carton_palletizing/tasks/box_grasp_vision/
```

该任务使用 336L 倾斜俯视 RGB-D 画面和 segmentation 模型，不使用 OBB 近似箱体边缘。它从 mask 外轮廓计算透视四边形，固定输出：

- 左上、右上、右下、左下四角；
- 箱体中心；
- 左右两条边的中点（机器人抓取点）；
- 以上七个点的像素坐标、深度和相机三维坐标；
- 原始外轮廓点。

模型目录：

```text
/opt/visionops_v3/models/carton_box_grasp/current/
├── model.rknn
└── model.yaml
```

手动启动：

```bash
./production/carton_palletizing/scripts/start_box_grasp_runtime.sh
./production/carton_palletizing/scripts/start_box_grasp_app.sh
./production/carton_palletizing/scripts/start_box_grasp_collector.sh
```

默认接口：

| 服务 | 地址 |
|---|---|
| Segmentation Runtime | `127.0.0.1:28085` |
| HTTP App | `127.0.0.1:19211` |
| Robot WebSocket | `0.0.0.0:9001/vision` |
| Collector Web | `0.0.0.0:18095` |
| 336L MJPEG | `:18182/stream.mjpeg` |

安装独立 systemd 服务：

```bash
sudo bash production/carton_palletizing/deploy/install_box_grasp_services.sh
```

机器人报文中 `items[]` 的每一项表示一个抓取点。每个纸箱输出两个同 ID 的抓取点，字段统一为 `id/class_id/confidence/position_camera/center_px`，与 `tube_pick_vision` 的单抓取点结构一致。

机器人协议与完整字段见：

```text
production/carton_palletizing/tasks/box_grasp_vision/PROTOCOL.md
```

### Box grasp 第一阶段 FPS 优化

`box_grasp_vision` 的后台 worker 是生产模式唯一推理生产者。机器人 WebSocket 和
Collector 生产页面都读取 `latest_decision`，避免两条链路重复提交 NPU 推理。

后台频率接口：

```bash
curl -s http://127.0.0.1:19211/api/app/inference_settings | python3 -m json.tool

curl -s -X POST http://127.0.0.1:19211/api/app/inference_settings \
  -H 'Content-Type: application/json' \
  -d '{"detection_fps":10}' | python3 -m json.tool
```

设置持久化到 `/opt/visionops_v3/config/box_grasp_inference_settings.json`。默认关闭
`debug.save_every_trigger`，避免每次推理创建保存线程并写入 RGB、Depth、JSON 和
Overlay。需要现场取证时再临时开启。

该任务启动 Runtime 时会使用：

```text
--max-detections 1
--mask-max-points 64
```

它只影响输出候选和 mask polygon 后处理，不改变模型输入或 NPU 网络结构。
