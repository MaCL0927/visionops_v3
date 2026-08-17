# foam_ring_grasp production package

当前干净生产基线：**M39.4.2.2 — visible-mouth + pure-side dual-branch production**。

## 当前有效能力

- `ring_mouth` 可见且可匹配：M39.3 FLAT / TILTED 生产抓取，clock 固定为 3 点钟，失败直接拒绝，不搜索其他 clock。
- `ring_mouth` 不可见，或被 topology gate 判定为 `SIDE_VIEW_PSEUDO_MOUTH`：进入 M39.4 纯侧躺分支。
- M39.4.0.1：恢复纯侧躺轴线与 selected entry end。
- M39.4.1：利用目标自身 camera-facing outer arc 重建 cross-section center、opening plane、opening center 与 Side Grasp Frame；不要求目标躺在箱底。
- M39.4.2.2：inner-finger 孔内包络、outer-finger clearance、完整工具/箱壁/邻环碰撞、PREGRASP→GRASP swept path、CLOSED grasp 环境、LEFT_LINK7 与正式闭夹抓取。

## 当前侧躺几何参数

`config/line.yaml` 当前实际值：

```text
grip insertion from ENTRY = 39 mm
PREGRASP outside ENTRY     = 20 mm
PREGRASP → GRASP           = 59 mm
ENTRY                      = geometric reference only; robot does not stop
```

M39.4.1 的 `preview_insertion_depth_mm` 只是 opening reconstruction 的可视化/诊断参数，不控制最终机器人抓取深度；真正机器人侧躺插入深度由 `m39_4_2_side_entry_validation.grasp_insertion_depth_mm` 控制。

## 侧躺机器人路径

```text
SIDE_INITIAL[OPEN]
  → SIDE_AVOIDANCE[OPEN]
  → SIDE_PREGRASP[OPEN]
  → SIDE_GRASP[OPEN]       # PREGRASP→GRASP 共轴直线，途中自然经过 ENTRY
  → CLOSE
  → SIDE_AVOIDANCE[CLOSED]
  → SIDE_INITIAL[CLOSED]
  → OPEN
```

Side initial 与固定防碰撞点保存在 `scripts/detect_move_validate.py`，为当前现场已验证位姿。

## 当前标定

- Eye-to-Hand：`config/handeye_calibration.json`，production workspace local PARK 标定，`quality_status=PASS`。
- 3D box：`config/box_model.json`，相机支架移动后重新拟合的当前模型。
- `config/line.yaml` 保存 hand-eye SHA256，`verify_production.py` 会强制校验 PASS / PARK / frame / SHA256。

## 目录

```text
foam_ring_grasp/
├── config/
│   ├── line.yaml
│   ├── handeye_calibration.json
│   ├── box_model.json
│   ├── online_service.env.example
│   └── runtime_rgb.env.example
├── scripts/
│   ├── start_rgb_runtime.sh
│   ├── start_online_service.sh
│   ├── verify_production.py
│   ├── detect_move_validate.py
│   └── fit_box_model_8point.py
├── tasks/foam_ring_grasp_vision/
├── docs/
└── README.md
```

## 预检、启动、抓取

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py

bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh

# 先 dry-run
python3 production/foam_ring_grasp/scripts/detect_move_validate.py

# 实际抓取
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

侧躺正式抓取仍要求二次 Enter 人工确认。

## 8 点箱体重标定

```bash
python3 production/foam_ring_grasp/scripts/fit_box_model_8point.py
```

依次点击 opening TL/TR/BR/BL + bottom TL/TR/BR/BL 共 8 点。默认只生成 candidate；确认 overlay 后再使用 `--install` 覆盖生产 `box_model.json`。
