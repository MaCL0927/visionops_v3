# Production Operations — M39.4.1

## 1. 预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

预检会检查 clock-3-only、hand-eye、M39.4.0.1、M39.4.1 validation-only 契约。

## 2. 启动

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

## 3. 一键验证

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

Visible-mouth FLAT/TILTED 保持原机器人流程。

侧躺分支打印：

```text
M39.4.1 CAMERA-FACING ARC + SIDE OPENING FRAME [VALIDATION ONLY]
axis_image_angle_deg
arc residual med/p90
arc inlier/raw ratio
arc span
opening support status / drop ratio
opening center camera
opening shift vs M39.4.0.1 nominal endpoint
preview grasp center
frame +X closing
frame +Y lateral
frame +Z insertion
frame quaternion
robot motion: DISABLED
```

## 4. Debug 文件

保存目录：

```text
/opt/visionops_v3/data/foam_ring_online_geometry/<timestamp>/
```

M39.4.1 新增：

```text
m39_4_1_side_opening_reconstruction.json
```

`online_geometry_overlay.jpg` 会叠加：

- M39.4.0.1 side axis；
- reconstructed opening center；
- preview grasp center；
- `+X close`；
- `+Z insert`；
- arc span / opening support drop；
- `robot=OFF`。

常用：

```bash
LATEST=$(ls -dt /opt/visionops_v3/data/foam_ring_online_geometry/* | head -1)
jq '.scene.m39_4_1_side_opening_reconstruction' "$LATEST/online_geometry_result.json"
```

## 5. 当前质量门槛

`line.yaml -> m39_4_1_side_opening_reconstruction` 主要检查：

- camera-facing arc clean support；
- fixed-radius residual median / p90；
- camera-facing arc angular span；
- raw contamination 不得严重到失去目标圆柱证据；
- selected-end axial support drop。

M39.4.0.1 的 `center_height_error_mm` 不参与 M39.4.1 PASS/FAIL。

## 6. 当前明确不做

- 侧躺机器人运动；
- inner-finger 实际孔内包络验证；
- outer-finger / 邻环 / 箱壁完整碰撞；
- PREGRASP→ENTRY→GRASP swept-volume；
- 一头翘起 / axis 有明显 Z 倾角的侧躺圆环。
