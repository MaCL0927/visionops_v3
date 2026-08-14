# Foam Ring Grasp Production Architecture — M39.4.1

## 1. 当前双主分支

```text
Orbbec 336L exact aligned RGB-D
              │
              ▼
       RKNN segmentation
     foam_ring / ring_mouth
              │
      ┌───────┴──────────────┐
      │                      │
front-accessible mouth   no mouth / pseudo mouth
      │                      │
      ▼                      ▼
M39.3 FLAT/TILTED       M39.4.0.1 side axis
clock-3 only                 │
      │                      ▼
robot production       reliable pure-side axis
                             │
                             ▼
                   M39.4.1 camera-facing arc
                             │
                    fixed-radius centre line
                             │
                   selected-end support drop
                             │
                    opening plane / center
                             │
                    Side Grasp Frame preview
                             │
                       robot_ready=false
```

## 2. M39.4.0.1 的职责

M39.4.0.1 只确定：

- 纯侧躺无向轴线；
- 进入端策略（非竖直取画面右端；近似竖直取更远箱壁端）；
- semantic `ring_mouth` 是否只是 `SIDE_VIEW_PSEUDO_MOUTH`。

轴线约束在 calibrated box XY，但**圆环不要求接触箱底**。`center_height_error_mm` 只作为 `FLOOR_RESTING_LIKE / ELEVATED_STACKED_OR_CENTER_UNCERTAIN` 诊断，不参与 M39.4.1 center 重建。

## 3. M39.4.1 camera-facing outer arc

M39.4.1 不从 RGB 猜隐藏孔，也不使用箱底高度恢复圆心。

对目标 ring：

1. 去 mask 边缘、邻环实例和 depth discontinuity；
2. 按 M39.4.0.1 axis 去掉轴向两端，仅保留中央约 60%；
3. 按轴向分 bin，每个 bin 保留局部更靠近相机的 depth envelope；
4. 在垂直 axis 的截面中，用 `R=42.5 mm` 固定半径做鲁棒圆弧中心拟合；
5. 粗拟合后剔除明显非目标圆柱点，再二次 fixed-radius refit；
6. 只使用 camera-facing hemisphere 产生 closing radial。

因此即使目标侧躺圆环下面是箱底、竖放圆环或其他支撑物，只要目标自身可见上半圆柱弧足够，仍可恢复 centre line。

## 4. Opening plane

从所有符合目标外圆柱 `R=42.5 mm` 的 shell points 得到轴向坐标。按 M39.4.0.1 已选 entry end 定向后，统计 2 mm axial bins 的 support profile。

从圆柱内部向外移动时，outer-arc support 会在真实端面附近下降。M39.4.1 用该 support drop 恢复 opening plane，而不是直接相信旧 `center ±35 mm`。70 mm 轴向长度只保留为先验/诊断。

## 5. Side Grasp Frame

```text
+X = closing: cross-section center → measured camera-facing outer wall
+Z = approach/insertion: selected opening → cylinder interior
+Y = +Z × +X
```

保证 `+X × +Y = +Z`，并与现有 `m38_6_visual_grasp` / `T_grasp_hand_tcp` 坐标契约一致：Visual +Z 最终映射到 TCP +X 前进方向，Visual +X 映射到 TCP +Z 夹紧方向。

当前 frame 同时输出：

- `opening_frame_camera`：origin = reconstructed opening center；
- `side_grasp_frame_camera`：origin = opening center + 18 mm * +Z。

第二个 origin 只是下一阶段的 preview insertion pose，不代表已经通过夹爪/碰撞验证。

## 6. Safety contract

```yaml
m39_4_1_side_opening_reconstruction:
  mode: online_validation_only
  robot_routing_enabled: false
```

M39.4.1 不写 `scene.robot_candidate`，因此不会进入 LEFT_LINK7 可执行路径。下一阶段再加入 inner-finger hole containment、outer-finger clearance、PREGRASP→ENTRY→GRASP sweep collision 和机器人路由。
