# foam_ring_grasp production package

当前生产基线：**M39.3.4.1 Visible-Mouth Production Freeze**。

## 当前承诺范围

- `ring_mouth` 可见且与 `foam_ring` 成功匹配：支持 **FLAT / TILTED** 两种生产抓取。
- FLAT：沿已冻结的箱体/前表面链路生成抓取。
- TILTED：使用 M39.3.4 analytic conic surface，并由 M39.3.4.1 重新生成真实倾斜抓取 frame。
- `UNCERTAIN`：拒绝。
- `ring_mouth` 不可见/侧躺：**当前统一保守拒绝，等待 M39.4**。
- 抓取钟点：**仅 3 点钟**。3 点钟失败即拒绝，不切换其他钟点。

## 目录

```text
foam_ring_grasp/
├── config/
│   ├── line.yaml                         # 唯一生产配置
│   ├── box_model.json                    # 已标定箱体 3D 模型
│   ├── handeye_left_20260810_190310_robot_default_base.json
│   ├── online_service.env.example
│   └── runtime_rgb.env.example
├── scripts/
│   ├── start_rgb_runtime.sh              # 启动 28081 RKNN Runtime
│   ├── start_online_service.sh           # 预检后启动 19213 服务
│   ├── verify_production.py              # 配置/手眼/生产约束预检
│   └── detect_move_validate.py           # 检测 + LEFT_LINK7 运动验证
├── tasks/foam_ring_grasp_vision/         # 在线运行所需算法代码
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OPERATIONS.md
│   └── CLEANUP_MANIFEST.md
└── README.md
```

## 替换后首次检查

在 `/opt/visionops_v3` 下执行：

```bash
python3 production/foam_ring_grasp/scripts/verify_production.py
```

应确认：

- `status = ok`
- FLAT clock = `[3]`
- TILTED clock = `[3]`
- 所有 fallback clock = disabled
- hand-eye calibration = PASS
- base frame = `robot_default_base`
- flange frame = `left_link7`

## 启动

```bash
bash production/foam_ring_grasp/scripts/start_rgb_runtime.sh
bash production/foam_ring_grasp/scripts/start_online_service.sh
```

## 机器人验证

先只检测、不连接机器人：

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py
```

正式执行并保留第二次 Enter 确认：

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute
```

稳定后才使用：

```bash
python3 production/foam_ring_grasp/scripts/detect_move_validate.py --execute --single-enter
```

详细运行与故障定位见 `docs/OPERATIONS.md`。
