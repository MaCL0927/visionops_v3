# foam_ring_grasp production package

当前生产基线：**M39.4.1 Camera-Facing Arc Opening Reconstruction**。

## 当前承诺范围

- `ring_mouth` 可见并与 `foam_ring` 匹配：继续支持 **FLAT / TILTED** 生产抓取。
- FLAT / TILTED 抓取方向均固定 **3 点钟**，失败即拒绝，不切换其他钟点。
- `ring_mouth` 不可见，或被 M39.4.0.1 判定为 `SIDE_VIEW_PSEUDO_MOUTH`：先恢复纯侧躺轴线，再进入 M39.4.1。
- M39.4.1 使用**目标圆环自身的 camera-facing outer arc + 已知外半径**恢复截面中心，不依赖圆环必须接触箱底，因此允许“侧躺但叠在其他圆环上”的情况。
- M39.4.1 再沿选定进入端检测轴向 outer-arc support drop，重建 opening plane / opening center，并输出 Side Grasp Frame。
- **M39.4.1 仍为 validation-only**：不生成侧躺机器人候选，不允许侧躺机器人运动。

## M39.4.1 几何链

```text
M39.4.0.1 reliable side axis
        ↓
target foam_ring exact depth
        ↓
central axial region
        ↓
local front-depth envelope
        ↓
camera-facing outer arc
        ↓
fixed R=42.5 mm robust circle fit
        ↓
cross-section centre line
        ↓
selected entry side
        ↓
axial outer-shell support profile
        ↓
opening support drop
        ↓
opening plane / opening center
        ↓
Side Grasp Frame
  +X = closing: hole → measured camera-facing outer wall
  +Z = approach/insertion: opening → ring interior
  +Y = lateral, maintaining the existing M38.6 visual-grasp convention
        ↓
preview grasp origin = opening + 18 mm * +Z
        ↓
robot_ready = false
```

`center_height_error_mm` / `floor_resting_consistent` 只作为支撑状态诊断。M39.4.1 不使用箱底高度恢复中心。

## 目录

```text
foam_ring_grasp/
├── config/
├── scripts/
│   ├── start_rgb_runtime.sh
│   ├── start_online_service.sh
│   ├── verify_production.py
│   └── detect_move_validate.py
├── tasks/foam_ring_grasp_vision/
│   ├── side_axis_recovery.py             # M39.4.0.1
│   ├── side_opening_reconstruction.py    # M39.4.1
│   └── ...                               # visible-mouth production chain
├── docs/
└── README.md
```

## 替换后预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

必须确认：

- `status = ok`
- FLAT / TILTED clock 均为 `[3]`
- M39.4.0.1 side-axis enabled
- M39.4.1 `mode = online_validation_only`
- M39.4.1 `robot_routing_enabled = false`
- hand-eye = PASS
- base = `robot_default_base`
- flange = `left_link7`

## 启动与验证

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

- visible-mouth FLAT/TILTED：仍按原安全流程允许人工二次确认后运动。
- side-lying：终端打印 M39.4.1 arc / opening / frame 结果并明确 `robot motion: DISABLED`。
