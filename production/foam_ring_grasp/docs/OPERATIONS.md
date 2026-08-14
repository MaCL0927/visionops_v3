# Production Operations

## 1. 整目录替换

建议先停止当前泡沫圆环服务，再替换目录：

```bash
cd /opt/visionops_v3
pkill -f 'production.foam_ring_grasp.tasks.foam_ring_grasp_vision.service' || true
rm -rf production/foam_ring_grasp
# 将新的 foam_ring_grasp/ 放入 production/
```

不要把旧目录中的 `line.yaml`、历史脚本或 docs 再复制回来。

## 2. 首次预检

```bash
cd /opt/visionops_v3
python3 production/foam_ring_grasp/scripts/verify_production.py
```

如果服务已经启动，可额外检查 19213：

```bash
python3 production/foam_ring_grasp/scripts/verify_production.py --service
```

## 3. 启动顺序

### 3.1 RGB Runtime

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
```

默认端口：`28081`。

### 3.2 Foam-ring online service

```bash
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

脚本启动前会自动执行 `verify_production.py`。任何生产配置/手眼契约异常都会阻止服务启动。

默认服务端口：`19213`。

## 4. 检测与机器人移动

### 4.1 Dry-run

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py
```

只触发视觉、检查 surface route、保存结果；不连接机器人。

### 4.2 正式执行

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

流程：

```text
Enter trigger
→ visual route validation
→ LEFT_LINK7 pose validation
→ second Enter
→ OPEN
→ PREGRASP
→ GRASP
→ CLOSE
→ PREGRASP
→ INITIAL
→ OPEN
```

### 4.3 Single-enter

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute --single-enter
```

只应在多轮 FLAT/TILTED 验证后使用。

## 5. 调试数据

视觉服务收到 `save_debug=true` 时默认保存到：

```text
/opt/visionops_v3/data/foam_ring_online_geometry/<capture_timestamp_ms>/
```

常用文件：

```text
exact_rgb.png
exact_depth.png
depth_colormap.jpg
online_geometry_overlay.jpg
runtime_inference_result.json
online_geometry_result.json
m39_3_1_tilt_evidence.json
m39_3_4_analytic_conic_surface.json
m39_3_4_1_tilted_production_routing.json
```

`detect_move_validate.py` 会直接在终端打印这些关键文件路径，并在当前工作目录保存完整响应：

```text
foam_ring_robot_validation_logs/<request_id>.json
```

## 6. 当前拒绝策略

下列情况不运动机器人：

- 无可靠 `ring_mouth` 匹配；
- surface classification = `UNCERTAIN`；
- TILTED 但未走 `M39.3.4.1_TILTED`；
- TILTED approach 与 `-analytic normal` 不一致；
- 3 点钟抓取不通过几何/碰撞检查；
- robot pose transform 未 ready；
- base/flange frame 不匹配；
- pregrasp/grasp 距离或四元数异常；
- 机器人起始位置偏离 READY/INITIAL 超限。

无 mouth 的侧躺圆环属于下一阶段 M39.4，不允许由旧 M38/M37 fallback 自动处理。
