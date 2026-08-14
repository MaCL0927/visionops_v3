# Foam Ring Grasp Production Architecture — M39.4.0.1

## 1. 当前双分支

```text
Orbbec 336L exact aligned RGB-D
              │
              ▼
       RKNN segmentation
     foam_ring / ring_mouth
              │
      ┌───────┴────────┐
      │                │
 ring_mouth matched   no matched ring_mouth
      │                │
      ▼                ▼
M38.1 front annulus   M39.4.0.1 side-axis recovery
      │                │
M39.3.1 tilt evidence RGB OBB: 2 axis seeds
      │                │
M39.3.4 analytic      depth-gradient anisotropy
      │                + 70/85 shape prior
 FLAT/TILTED           │
      │                ▼
clock-3 only       winning image axis
      │                │
collision gates     box-XY hard constraint
      │                │
robot transform     R=42.5 mm cylinder verify
      │                │
LEFT_LINK7          ±2° local refinement
                       │
                       ▼
                 undirected 3-D axis
                       │
              center ± 35 mm endpoints
                       │
              entry-end selection only
                       │
                 robot_ready=false
```

## 2. 为什么 M39.4.0.1 不找隐藏孔

侧躺且离相机光轴较远时，即便物体没有 Z 向倾斜，也可能因透视看到局部内壁。当前 `foam_ring` segmentation 仍主要提供外轮廓实例，不能把局部深色/深度区域可靠解释成真实 `ring_mouth`。

因此 M39.4.0.1 的设计原则是：

1. RGB 只提供**两个正交轴候选**；
2. 端部和局部内壁不负责决定轴线正负；
3. 深度沿正确圆柱轴方向变化应小于横向变化；
4. 已知物理尺寸 `axial_length / outer_diameter = 70 / 85` 提供弱外形先验；
5. 最终轴线被硬限制在 calibrated box XY 平面；
6. 用已知外半径 42.5 mm 的局部圆柱残差做验证，而不是全空间搜索。

## 3. Semantic mouth topology arbitration

`ring_mouth` 是语义观测，不再直接等价于可访问正面开口。M39.4.0.1 对已匹配 mouth 的 ellipse 做 `minor/major` 检查：

- `ratio > 0.50`：保持原 M39.3 visible-mouth 路由；
- `ratio <= 0.50`：标记 `SIDE_VIEW_PSEUDO_MOUTH`，撤销该 ring 的 visible-mouth production candidate，并送入 side-axis recovery。

这用于处理完全侧躺但因离相机主光轴较远而透视可见部分内壁的情况。

## 4. Pure-side axis recovery

M39.4.0.1 只接受“圆环放在箱底、轴线平行箱底”的简单侧躺情况。

箱体深度为 90 mm、圆环外半径为 42.5 mm，因此理想圆柱中心在 box-Z：

```text
90 - 42.5 = 47.5 mm
```

`center_height_error_mm` 现在只评估圆心可靠度，不再否决轴线。超过 `center_height_warning_mm` 时输出 `center_reliable=false` 与 warning，但只要固定半径圆柱几何可靠，`axis_reliable` 仍可为 true。真正的一头翘起/Z 向姿态将在后续阶段用独立几何证据处理，避免单侧可见圆弧反推圆心的不稳定性污染轴线判断。

## 5. Entry endpoint policy

恢复的轴线首先是无向的。M39.4.0.1 只做进入端选择，不做夹爪插入姿态。

- 非近似竖直轴：选择**画面右侧** endpoint。
- 近似竖直轴：比较两个 endpoint 沿轴向外侧到 calibrated box wall 的可用距离，选择 clearance 更大的一端。
- 如果右端靠墙：M39.4.0.1 仍保留“右端”这一生产策略并报告 clearance；M39.4.1 做完整夹爪碰撞后应直接拒绝，不回退左端。

## 6. Robot safety contract

M39.4.0.1 配置必须保持：

```yaml
m39_4_0_side_axis_recovery:
  mode: online_diagnostic_only
  robot_routing_enabled: false
```

它不会写入 `scene.robot_candidate`。因此 no-mouth side target 即使轴线恢复成功，也不会进入 `robot_pose_transform` 的可执行路径。

Visible-mouth FLAT/TILTED 的既有安全契约保持不变。
