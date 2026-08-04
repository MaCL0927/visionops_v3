# 泡沫圆环抓取：M35.4 有向定长 3D 轴杆投影

M35.4 在 M35.2 抓取几何基础上，将成功配对的 `foam_ring` / `ring_mouth` 三维轴线改为**有向、定长的空间短轴杆**后再投影到 RGB 图像。它不修改配对、夹取点选择、完整夹爪模型或碰撞判断。

## 1. 为什么使用定长 3D 轴杆

轴杆先在相机坐标系中保持固定物理长度，再投影到二维图像。因此：

```text
接近正对相机：二维投影较短
倾斜角增大：在相同深度和画面位置下，二维投影通常逐渐变长
```

轴杆以圆环三维中心为中点：

```text
near = center + axis_toward_camera * rod_length / 2
far  = center - axis_toward_camera * rod_length / 2
```

## 2. 图像编码

```text
绿色轮廓/半透明区域：成功配对的 foam_ring
橙色轮廓/半透明区域：对应 ring_mouth
红色实线箭头 + 实心端点：near，沿 ring_axis_toward_camera 朝向相机
青色虚线 + 空心端点：far，背离相机
白色点：圆环三维中心的投影
L2D：固定 3D 轴杆投影后的像素长度
dz：far_z - near_z，正常应为正值
E：ellipse_stabilized
D：depth_plane / near_frontal_depth_plane
SIGN?：轴向量的“朝向相机”符号与端点深度顺序不一致
```

当轴杆几乎沿相机光轴时，near/far 投影会接近同一点。此时显示青色空心圆和红色实心圆叠加，不再人为放大亚像素方向。

## 3. 输出

每个采集帧目录可生成：

```text
paired_axis_overlay.jpg
paired_axis_projection.json
```

JSON 记录：

```text
center_camera_mm
axis_toward_camera
rod_length_mm
near_camera_mm / far_camera_mm
near_uv / far_uv
near_depth_mm / far_depth_mm
depth_delta_far_minus_near_mm
projected_rod_length_px
depth_order_status
```

## 4. 配置开关

配置文件：

```text
production/foam_ring_grasp/config/line.yaml
```

```yaml
axis_direction:
  enabled: true
  rod_length_mm: 80.0
  minimum_projected_rod_px: 1.0
  mask_alpha: 0.18
  line_thickness: 3
  far_dash_length_px: 9.0
  far_gap_length_px: 6.0
  endpoint_radius_px: 7
  draw_vector_values: true
  draw_depth_values: true

output:
  save_paired_axis_overlay: true
```

关闭 M35.4 诊断：

```yaml
axis_direction:
  enabled: false
```

关闭后不会计算短轴杆端点、不会投影、不会绘图，也不会写出 `paired_axis_overlay.jpg` 和 `paired_axis_projection.json`。如果复用旧输出目录，程序会删除旧的轴线诊断文件，防止误看历史结果。

> `ring_axis_toward_camera` 的核心姿态法向仍会由 M35.2 主链路计算，因为倾角、夹取姿态和碰撞检查依赖它。若连核心法向也关闭，现有抓取逻辑将无法成立。

## 5. 离线运行

```bash
cd /home/pc/桌面/visionops_v3

rm -rf /home/pc/桌面/img_m35_4_axis

python -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.offline_validate \
  --data-root \
  /home/pc/桌面/visionops_v3/server_data/batches/rk3576-001_ring_20260728_132732/raw \
  --all \
  --model \
  /home/pc/桌面/visionops_v3/server_data/jobs/rk3576-001_ring_seg_job_20260729_100613/work/runs/segment_train/weights/best.pt \
  --device 0 \
  --output /home/pc/桌面/img_m35_4_axis
```

## 6. M35.2 主链路保持不变

保留功能包括：

- `foam_ring` / `ring_mouth` 配对；
- 深度平面与椭圆稳定化姿态；
- 圆环中心、倾角与 12 时钟夹取点；
- 标定 3D 箱体；
- 邻近圆环三维点云；
- 完整夹爪最终静态碰撞检查；
- `pregrasp -> open -> approach -> insert -> close` 完整抓取前运动扫掠。

离线结果仍保持 `robot_ready=false`，因为机器人关节可达性、手眼变换和在线触发链路尚未纳入当前阶段。

## M36.2：3576 实时 RGB RKNN 推理

M36.2 将 Orbbec 336L Bridge 的 RGB 共享内存接入 C++ RKNN Runtime，暂不读取深度：

```bash
cd /opt/visionops_v3
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh \
  /opt/visionops_v3/models/rk3576-001_ring_seg_20260729_100731

python3 production/foam_ring_grasp/scripts/verify_rgb_runtime.py \
  --samples 20 \
  --report /tmp/m36_2_report.json
```

正式验收要求 `frame_source.transport=posix_shared_memory`、`fallback_active=false`，且共享内存 sequence 和推理结果 `capture_timestamp_ms` 持续变化。详细说明见 `docs/M36.2_ORBBEC_SHARED_RGB_RKNN_RUNTIME.md`。

## M36.3：精确同步 RGB-D 短时缓存

M36.3 在 M36.2 通过后启动后台 `RgbdFrameCache`，持续复制同一 Orbbec SDK FrameSet 的 RGB 与 D2C 深度，并按 `timestamp_epoch_ms` 保存最近 12 帧。Runtime 返回分割结果后，仅允许使用：

```python
frame = cache.get_exact(result["capture_timestamp_ms"])
```

找不到完全相同的时间戳时，本次三维计算必须失败；禁止把最新或最近的深度静默拼到旧 RGB 掩膜上。

真机验收：

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_rgbd_cache.py \
  --samples 20 \
  --report /tmp/m36_3_report.json
```

详细说明见 `docs/M36.3_EXACT_RGBD_FRAME_CACHE.md`。


## M36.4：单次在线三维几何验证

M36.4 将 Runtime 的 RKNN `legacy_proto` polygon 转成现有
`SegmentationInstance`，按 `capture_timestamp_ms` 取得完全相同的 RGB-D，
再调用既有 `analyze_scene()`。当前箱体模型在 Runtime input ROI
（528×455）中标定，因此在线处理会同步裁剪 RGB、Depth 和 mask，并将
内参变换为 `cx-=roi_x1`、`cy-=roi_y1`。

```bash
cd /opt/visionops_v3
bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh
```

输出位于 `data/foam_ring_online_geometry/<timestamp>/`。该阶段仅用于在线
诊断，结果始终保持 `robot_ready=false`。详细说明见
`docs/M36.4_ONLINE_GEOMETRY_ONCE.md`。

## M36.4.1：分段计时与候选分级计算

M36.4.1 将在线默认模式改为 `staged`：全部钟点先做轻量评分，只对全场排名靠前的少数候选运行完整 3-D 箱壁、邻环和整夹爪碰撞；同时缓存每个圆环的基础点云和每个目标的二维距离变换。原始全部候选完整检查保留为 `exhaustive` 模式。

```bash
cd /opt/visionops_v3
bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh

python3 production/foam_ring_grasp/scripts/summarize_geometry_timing.py
```

A/B 对照：

```bash
VISIONOPS_FOAM_RING_GEOMETRY_MODE=staged \
  bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh

VISIONOPS_FOAM_RING_GEOMETRY_MODE=exhaustive \
  bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh
```

详细说明见 `docs/M36.4.1_STAGED_GEOMETRY_TIMING_OPTIMIZATION.md`。
## M36.4.2：首个有效目标提前退出与自适应钟点搜索

在线默认模式改为 `first_valid`。成功配对先用稀疏前表面深度、配对质量和分割置信度进行低成本排序；只逐个执行目标的 RANSAC、姿态和碰撞计算。找到第一个完整有效抓取后立即停止，未处理目标标记为 `deferred`。每个目标先搜索 8 个主钟点，全部失败时才补充剩余 4 个钟点。

```bash
cd /opt/visionops_v3
bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh

python3 production/foam_ring_grasp/scripts/summarize_geometry_timing.py
```

模式对照：

```bash
VISIONOPS_FOAM_RING_GEOMETRY_MODE=first_valid \
  bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh

VISIONOPS_FOAM_RING_GEOMETRY_MODE=staged \
  bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh

VISIONOPS_FOAM_RING_GEOMETRY_MODE=exhaustive \
  bash production/foam_ring_grasp/scripts/run_online_geometry_once.sh
```

详细说明见 `docs/M36.4.2_FIRST_VALID_ADAPTIVE_CLOCK_OPTIMIZATION.md`。


## M36.5：常驻触发式在线服务

M36.5 将 M36.4.2 的单次在线几何流程封装为常驻服务。RGB-D 缓存、Runtime IPC、几何配置和箱体模型只初始化一次；显式 `request_id` 通过 inference/geometry 双线程有界可靠流水线处理，重复 request_id 幂等返回原任务/结果。生产触发默认不保存文件，`save_debug=true` 才保存完整证据链。

```bash
cd /opt/visionops_v3
bash production/foam_ring_grasp/scripts/start_online_service.sh

curl -s -X POST http://127.0.0.1:19213/api/foam_ring/infer_once \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"robot-000001","wait":true,"save_debug":false}' \
  | python3 -m json.tool

python3 production/foam_ring_grasp/scripts/verify_online_service.py \
  --samples 5 \
  --report /tmp/m36_5_report.json
```

详细说明见 `docs/M36.5_PERSISTENT_TRIGGER_SERVICE.md`。

## M37 侧躺圆环参数化 3D 模板

对于没有可靠 `ring_mouth` 的侧躺圆环，M37 使用已知外径、内径和轴向长度的短空心圆柱模型拟合 `foam_ring` 内部 RGB-D 点云。轴线从远端指向距离深度相机更近的端面。M37.1 在近端后方的真实可见圆柱弧面上计算抓取点，不再使用开口端投影最高点。M37.2 默认按 `foam_ring` 置信度排序，逐个尝试并在首个有效抓取点后立即退出，同时使用快速粗到细轴线搜索和保守高精度回退。该流程当前仅用于离线验证，详见 `docs/M37_SIDE_RING_PARAMETERIZED_TEMPLATE.md`、`docs/M37.1_NEAR_VISIBLE_CROWN_GRASP_POINT.md` 和 `docs/M37.2_CONFIDENCE_FIRST_FAST_TEMPLATE_FITTING.md`。

## M37.3 统一触发：开口可见优先、侧躺自动回退

M37.3 沿用 M36.5 的 `/api/foam_ring/infer_once` 接口。调用方不需要指定
目标姿态。每次触发只执行一次 Runtime 分割和一次精确 RGB-D 匹配：

1. 先运行 M36 `foam_ring + ring_mouth` 抓取；
2. M36 没有有效候选时，才对其未匹配的 `foam_ring` 按置信度降序运行
   M37.2 短圆柱模板拟合；
3. M37 首个有效目标成功后立即退出。

响应中的 `selected_grasp_branch` 为：

- `m36_mouth_visible_rim_pinch`
- `m37_side_ring_near_visible_crown`
- `none`

查看分支耗时：

```bash
python3 production/foam_ring_grasp/scripts/summarize_hybrid_timing.py
```

完整说明见 `docs/M37.3_HYBRID_REALTIME_GRASP.md`。
