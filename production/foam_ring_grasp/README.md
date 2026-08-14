# foam_ring_grasp production package

当前生产基线：**M39.4.0.1 Side-Lying Axis Recovery**。

## 当前承诺范围

- `ring_mouth` 可见并与 `foam_ring` 匹配：继续支持 **FLAT / TILTED** 生产抓取。
- FLAT / TILTED 抓取方向均固定 **3 点钟**，失败即拒绝，不切换其他钟点。
- `ring_mouth` 不可见、目标为**完全侧躺且轴线平行箱底**：M39.4.0.1 恢复圆柱无向轴线、两个理论开口中心以及推荐进入端。
- M39.4.0.1 **只做几何诊断，不生成 robot candidate，不允许机器人运动**。
- 一头翘起、倚墙、明显 Z 向倾斜的侧躺目标：当前拒绝，留给 M39.4 后续阶段。

## M39.4.0.1 核心规则

```text
foam_ring detected + no matched ring_mouth
        ↓
RGB OBB → 两个正交轴候选
        ↓
Depth directional-gradient score
+ 70/85 physical silhouette prior
        ↓
选出 winning axis
        ↓
calibrated box XY hard constraint
        ↓
fixed-radius (R=42.5 mm) local cylinder verification ±2°
        ↓
重建两个 opening centers (±35 mm)
        ↓
非竖直：固定选择画面右端
近似竖直：选择 outward box-wall clearance 更大的一端
        ↓
robot_ready = false
```

透视导致的局部内壁可见不会被解释成 `ring_mouth`；算法不会依靠“哪一侧看起来像孔”来判断轴向。

## 目录

```text
foam_ring_grasp/
├── config/
│   ├── line.yaml
│   ├── box_model.json
│   ├── handeye_left_20260810_190310_robot_default_base.json
│   ├── online_service.env.example
│   └── runtime_rgb.env.example
├── scripts/
│   ├── start_rgb_runtime.sh
│   ├── start_online_service.sh
│   ├── verify_production.py
│   └── detect_move_validate.py
├── tasks/foam_ring_grasp_vision/
│   ├── side_axis_recovery.py          # M39.4.0.1
│   └── ...                            # visible-mouth production chain
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OPERATIONS.md
│   └── CLEANUP_MANIFEST.md
└── README.md
```

## 替换后预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

应确认：

- `status = ok`
- FLAT / TILTED clock 均为 `[3]`
- M39.4.0.1 `mode = online_diagnostic_only`
- M39.4.0.1 `robot_routing_enabled = false`
- hand-eye = PASS
- base = `robot_default_base`
- flange = `left_link7`

## 启动

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

## 一键验证

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

- visible-mouth FLAT/TILTED：仍按原安全流程允许人工二次确认后运动。
- no-mouth pure-side：脚本打印 M39.4.0.1 axis/endpoint 结果并明确 `robot motion: DISABLED`，不会执行运动。

详见 `docs/OPERATIONS.md`。

## M39.4.0.1 optimization contract

- A detected `ring_mouth` with ellipse `minor/major <= 0.50` is a `SIDE_VIEW_PSEUDO_MOUTH`; it is not allowed to enter the frozen M39.3 production branch.
- Pseudo-mouth rings and genuinely unmatched rings both enter side-axis recovery.
- If the cheap orthogonal-axis score margin is below `1.20`, both RGB axis seeds receive fixed-radius 3-D refinement and full geometry decides the axis.
- `center_height_error_mm` is diagnostic only; an unreliable center no longer rejects an otherwise reliable axis.
- M39.4.0.1 remains validation-only: no side-lying robot candidate is generated.
