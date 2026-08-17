# Foam Ring Grasp Production Architecture — Clean M39.4.2.2 Baseline

## 1. 双主分支

```text
Orbbec aligned RGB-D
        ↓
RKNN segmentation: foam_ring / ring_mouth
        ↓
 ┌───────────────┬───────────────────────────┐
 │               │                           │
real visible mouth                 no-mouth / pseudo-mouth
 │                                           │
M39.3 FLAT/TILTED                        M39.4.0.1
clock-3 only                         side axis + selected end
 │                                           ↓
 │                                        M39.4.1
 │                              camera-facing arc reconstruction
 │                                           ↓
 │                                        M39.4.2.2
 │                            rim-pinch + clearance + collision
 └───────────────────────────────┬───────────┘
                                 ↓
                     camera → robot_default_base
                                 ↓
                         hand_tcp → LEFT_LINK7
```

## 2. Side opening reconstruction

M39.4.1 不使用箱底高度恢复圆环截面中心。已知可靠 cylinder axis 和 `Ro` 后，从目标 mask 的 camera-facing outer arc 做固定半径鲁棒拟合，再由 selected end 的 axial support drop 恢复 opening plane。

`opening_center` 位于圆柱中心轴；rim-pinch 的 ENTRY origin 位于 camera-facing 环壁壁厚中点：

```text
rim_radius = (Ri + Ro) / 2
ENTRY = opening_center + rim_radius * closing_axis
```

## 3. Side Visual Grasp Frame

沿用 `m38_6_visual_grasp` 坐标合同：

```text
Visual +X = closing axis = hole → camera-facing outer wall
Visual +Z = insertion axis = selected opening → ring interior
Visual +Y = right-handed lateral axis
```

已有固定变换保持：Visual +Z → TCP +X；Visual +X → TCP +Z。

## 4. 当前 PREGRASP / ENTRY / GRASP

以 `config/line.yaml` 为唯一参数真值。当前：

```text
PREGRASP = ENTRY - 20 mm * +Z
ENTRY    = opening-plane rim midpoint, diagnostic reference only
GRASP    = ENTRY + 39 mm * +Z
```

因此机器人最后一段 `PREGRASP → GRASP` 为 59 mm 共轴直线，机器人不在 ENTRY 停车。

注意：M39.4.1 的 `preview_insertion_depth_mm` 不是机器人 GRASP 深度。

## 5. M39.4.2.2 安全门禁

对 keyframe 与 PREGRASP→GRASP 路径采样检查：

- inner finger 必须进入理论内孔，并满足 `inner_hole_clearance_margin_mm`；
- outer finger 必须保持在理论外圆柱之外，并满足 `outer_finger_clearance_margin_mm`；
- palm / contact block / mounting disk / pneumatic fitting 不得与目标、邻环、箱壁产生超阈值冲突；
- CLOSED GRASP 环境必须通过；
- selected entry 发生硬碰撞时直接拒绝，不自动换另一端。

侧躺箱壁采用 component-wise physical-clearance policy；mounting disk 使用保守圆柱模型，允许最多 6 mm 模型/标定容差，但不是关闭箱壁碰撞。

历史通用 `robot_wrist` 圆柱默认不参与视觉工具硬碰撞，机器人腕部/手臂由机器人控制/规划层负责。

## 6. Robot transform

production candidate 通过后生成：

```text
LEFT_LINK7 PREGRASP
LEFT_LINK7 ENTRY   # diagnostic/reference
LEFT_LINK7 GRASP
```

变换链：

```text
T_base_grasp
= T_base_camera
  @ T_camera_grasp

T_base_hand_tcp
= T_base_grasp
  @ T_grasp_hand_tcp

T_base_left_link7
= T_base_hand_tcp
  @ T_hand_tcp_left_link7
```

Eye-to-Hand 必须满足 `PASS + PARK + robot_default_base + SHA256` 契约。

## 7. 实际 side robot cycle

```text
SIDE_INITIAL[OPEN]
→ SIDE_AVOIDANCE[OPEN]
→ SIDE_PREGRASP[OPEN]
→ SIDE_GRASP[OPEN]
→ CLOSE
→ SIDE_AVOIDANCE[CLOSED]
→ SIDE_INITIAL[CLOSED]
→ OPEN
```

抓紧后不沿 ENTRY/PREGRASP 原路退出，而是直接返回固定 SIDE_AVOIDANCE。
