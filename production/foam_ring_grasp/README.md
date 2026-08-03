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
