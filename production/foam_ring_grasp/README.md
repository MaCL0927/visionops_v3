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

## M37.4 深度分层抓取排序与有界高精度回退

M37.4 将统一触发策略改为“深度层级优先”：全部 `foam_ring` 先计算稳健
前表面深度并按 30 mm 分层，严格从最近层开始；同层内先尝试 M36，随后
才尝试 M37。M37 不再对单个不确定目标立即执行完整 accurate 全局搜索，
而是先按深度顺序执行 fast；当前层全部 fast 不确定时，只对最佳 fast 结果
进行一次 warm-start 局部精修。

```bash
python3 production/foam_ring_grasp/scripts/summarize_hybrid_timing.py
python3 production/foam_ring_grasp/scripts/replay_m37_4_result_set.py \
  /path/to/foam_ring_online_geometry --output /tmp/m37_4_replay
```

完整说明见 `docs/M37.4_DEPTH_LAYERED_BOUNDED_REFINEMENT.md`。

## M37.5 圆柱侧面点云分离、法向约束轴线拟合与姿态拒绝

M37.5 修复“径向残差很低但轴线明显错误”的系统性问题。在线流程会排除
相邻实例接触区和深度跳变边缘，从组织化 RGB-D 计算局部表面法向，并将
“侧面法向垂直于轴线、法向与圆柱径向一致”加入姿态评分。程序保留多个
轴线假设，并通过候选分数间隔、法向种子一致性和 bootstrap 重采样稳定性
拒绝欠约束姿态。M36 同时增加深度平面与椭圆法向的硬冲突拒绝。

查看最近一次姿态安全与耗时：

```bash
python3 production/foam_ring_grasp/scripts/summarize_hybrid_timing.py
```

批量回放调试结果：

```bash
python3 production/foam_ring_grasp/scripts/replay_m37_5_result_set.py \
  /path/to/foam_ring_online_geometry --output /tmp/m37_5_replay
```

完整说明见 `docs/M37.5_NORMAL_CONSTRAINED_POSE_SAFETY.md`。

## M37.5.1：分阶段姿态验证

M37.5.1 将侧躺圆环在线流程改为“轻量候选预筛选 → preliminary screen → 延迟 final validation → 单次局部精修兜底”。`fit_score` 只用于质量排序，只有 `final_pose_safe=true` 的候选能够输出抓取点。详细配置与输出字段见 `docs/M37.5.1_STAGED_POSE_VALIDATION.md`。


## M37.6 空心短圆柱多曲面联合拟合

M37.6 在 M37.5.1 上将外壁、内壁和两端环形面联合建模；`ring_mouth` 作为可选端面约束，未检测到 mouth 的混合姿态可使用浅/深 RGB-D 点云质心差提供轴线初值。详见 `docs/M37.6_HOLLOW_CYLINDER_MULTISURFACE.md`。

## M37.6.1 快速止损

M37.6.1 将 depth-gradient 降为纯诊断证据，关闭在线 `local_accurate`，并将 M36 椭圆/深度平面冲突显式转交 M37.6，避免错误梯度否决优质候选和单次十余秒的局部精修。详见 `docs/M37.6.1_FAST_STOPLOSS.md`。

## M38.1 分支 A：清晰开口三维端面环带

M38.1 全局优先处理 `foam_ring + ring_mouth` 配对且开口证据清楚的实例。算法从开口外围构造泡沫端面环带，剔除内孔、其他圆环、分割边缘和深度跳变区域，再使用三维点云 RANSAC/SVD 直接拟合端面平面；二维椭圆只负责开口定位与完整性检查，不再用于推导三维倾角。通过 M38.1 后继续沿用原有钟点搜索、内外夹持和全夹爪碰撞验证。

原 M36 和 M37.6 代码仍保留；M38.5 生产配置暂时关闭两者，执行 M38.1、M38.3、纯侧面外接触分支和显式分支 C。完整算法、配置、输出字段和回放方式见 `docs/M38.1_CLEAR_MOUTH_FRONT_ANNULUS.md`。
## M38.3 分支 B：深度开口证据约束的局部圆柱

M38.3 修复 M38.2 对 `ring_mouth` 分割过度依赖的问题。对于未配对的 `foam_ring`，算法会在实例自身深度中查找偏心深孔及其周围近侧端部支撑；对于已经分割的 mouth，只有明显偏心的部分开口才进入分支 B。轴线不再自由三维搜索，而是约束其图像投影必须从 ring 主体指向观测开口，仅采样有限的相机视向分量，并以已知外半径执行小规模局部精修。

拟合通过后，M38.3 在恢复的三维开口平面上使用名义内外半径生成内外夹持边界，避免半侧躺投影下二维 mask 射线边界退化。只看到外侧壁、没有开口/端部证据的实例仍会拒绝；邻居、箱体、完整夹爪和运动碰撞检查继续拥有最终否决权。旧 M38.2 实现仅作为历史回放保留。生产开关仍为 `m38_branch_b.enabled`，独立回放脚本为 `production/foam_ring_grasp/scripts/replay_m38_3_branch_b.py`。完整说明见 `docs/M38.3_DEPTH_PARTIAL_OPENING_CONSTRAINED_CYLINDER.md`。

## M38.4 分支 C：显式拒绝与快速终止

M38.4 将 M38.1/M38.3 都无法提供安全内外夹持候选的场景显式归入分支 C。纯侧面、疑似深孔但端部证据不足、或所有钟点均碰撞时，系统返回 `m38_4_branch_c_fast_reject` 和 `turn_or_agitate_box`，不再进入 M36/M37.6 的昂贵欠约束搜索。生产配置默认 `legacy_m36_enabled: false`、`side_ring_fallback_enabled: false`，旧代码仍保留，可通过 YAML 恢复。完整说明见 `docs/M38.4_BRANCH_C_FAST_REJECT.md`。

## M38.5 纯侧面外接触几何

M38.5 在整帧没有 `ring_mouth` 时，使用观测到的外圆柱侧面直接恢复外接触点、外表面法向、向内闭合方向和无向圆柱轴线。它不补全隐藏开口、不生成内夹爪点，也不运行 12 钟点完整夹爪检查，输出始终为 `robot_ready: false`。M38.3 的弱深孔证据会先快速门控，再优先交给 M38.5，避免约 3 秒以上的无效完整夹爪计算。详见 `docs/M38.5_PURE_SIDE_OUTER_CONTACT.md`。

## M38.6 方向、碰撞与接触点修复

M38.6 将 M38.1 从“首个有效方向”改为至少完整比较前 3 个优先钟点；已确认的箱壁或完整夹爪箱体相交成为硬拒绝；分支 C 分别显示“姿态不可靠”“开口观测不足”或“夹爪碰撞”，不再混用碰撞提示。M38.5 纯侧面接触点改为选择相机更近的轴向端点，并从该端向内移动 15%，而不是固定在轴向中部。回放脚本为 `scripts/replay_m38_6_result_set.py`，完整说明见 `docs/M38.6_DIRECTION_COLLISION_CONTACT_FIX.md`。
