# Production Operations — M39.4.0.1

## 1. 整目录替换后预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

预检会同时检查：

- clock-3-only；
- hand-eye / `robot_default_base` / `left_link7`；
- M39.4.0.1 已启用；
- M39.4.0.1 仍为 diagnostic-only，机器人路由关闭。

## 2. 启动

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

`start_online_service.sh` 会在启动 19213 前再次执行生产预检。

## 3. 现场验证

建议继续使用同一个脚本：

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

### 3.1 Visible-mouth

保持已有行为：

```text
Enter → vision → FLAT/TILTED route check → LEFT_LINK7 check
      → second Enter → robot grasp cycle
```

### 3.2 No-mouth pure-side

M39.4.0.1 检出后终端会打印：

```text
M39.4.0.1 SIDE-LYING AXIS RECOVERY [VALIDATION ONLY]
axis_reliable
axis_image_angle_deg
axis_camera_undirected
axis_score_margin
radial residual med/p90
center_height_error_mm
endpoint A/B uv
entry_endpoint
entry_selection_rule
entry_wall_clearance_mm
robot motion: DISABLED BY M39.4.0.1 CONTRACT
```

即使脚本带 `--execute`，该分支也会直接结束本轮，不执行机械臂运动。

## 4. Debug 文件

`save_debug=true` 时目录：

```text
/opt/visionops_v3/data/foam_ring_online_geometry/<capture_timestamp_ms>/
```

M39.4.0.1 新增：

```text
m39_4_0_side_axis_recovery.json
```

同时 `online_geometry_overlay.jpg` 会画：

- 恢复的 opening axis；
- A/B 两个理论端面中心；
- 最终 entry endpoint；
- axis angle / score margin；
- `robot=OFF`。

常用查看：

```bash
LATEST=$(ls -dt /opt/visionops_v3/data/foam_ring_online_geometry/* | head -1)
ls -lh "$LATEST"
jq '.scene.m39_4_0_side_axis_recovery' "$LATEST/online_geometry_result.json"
```

## 5. 当前 M39.4.0.1 质量门限

主要门限位于 `line.yaml -> m39_4_0_side_axis_recovery`：

- pseudo-mouth ellipse axis ratio（默认 `<=0.50` 回流侧躺分支）；
- quick two-axis margin（只决定是否启动双轴完整拟合，不再直接拒绝）；
- dual-seed full-geometry score margin；
- fixed-radius radial median / p90；
- radial inlier ratio；
- center-height warning（只影响 `center_reliable`）；
- vertical-axis threshold。

出现 `axis_uncertain` 时优先检查 full-geometry margin 与 radial quality；`center_reliable=false` 本身不代表轴线错误。

## 6. 暂不处理

以下情况继续拒绝：

- 一头翘起；
- 明显 Z 向倾斜；
- 深度缺失/大面积污染；
- 双轴完整拟合后 full-geometry 仍无法拉开差距；
- 圆柱半径残差不符合已知 85 mm 外径。

这些属于 M39.4 后续阶段，不在 M39.4.0.1 内增加复杂 fallback。
