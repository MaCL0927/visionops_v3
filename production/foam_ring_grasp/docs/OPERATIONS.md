# Production Operations — Clean M39.4.2.2

## 1. 生产预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

必须确认：

- `status=ok`；
- visible-mouth FLAT/TILTED clock 固定 `[3]`，fallback disabled；
- hand-eye = PASS / PARK / robot_default_base / SHA256 verified；
- M39.4.2.2 `mode=side_grasp_production`；
- `production_grasp_enabled=true`；
- `gripper_closing_enabled=true`。

## 2. 启动

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

## 3. Dry-run

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py
```

当前 side 参数应体现：

```text
PREGRASP → ENTRY(ref) = 20 mm
ENTRY(ref) → GRASP    = 39 mm
PREGRASP → GRASP      = 59 mm
```

ENTRY 只用于 opening-plane 几何诊断，机器人不停车。

## 4. 正式抓取

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

Side cycle：

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

侧躺正式抓取保留第二次 Enter 确认。

## 5. 8 点 box calibration

```bash
python3 production/foam_ring_grasp/scripts/fit_box_model_8point.py
```

点击顺序：

```text
OPEN_TL → OPEN_TR → OPEN_BR → OPEN_BL
BOTTOM_TL → BOTTOM_TR → BOTTOM_BR → BOTTOM_BL
```

默认输出 candidate + overlay + points，不覆盖生产模型。确认后再用 `--install`；工具会备份旧 `box_model.json`。

这些 candidate / backup 属于现场标定中间产物，不应长期打包进干净 production 基线。
