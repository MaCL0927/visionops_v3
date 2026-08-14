# M39.3.4.1 Production Cleanup Manifest

## 删除的历史资产

- 根目录阶段性说明 `M35.3_*`。
- `docs/` 下 M38.x、M39.2.x、M39.3.0~3.3 阶段说明和 replay JSON。
- `scripts/` 下所有 `replay_*`、`test_m39_*`、timing summary、共享内存诊断等阶段性测试脚本。
- `config/line_clock3_calibration.yaml`。
- `config/line.yaml.bak_*`。
- 旧 `handeye_left_20260808_*` 以及 chest-base backup。
- `config/box_model_overlay.jpg`。
- task 中不被在线 service 引用的 `offline_validate.py`、`online_validate.py`、`box_calibration.py` 和 `tools/`。
- M39.3.2 ring-prior 在线诊断调用和独立 M39.3.3 在线诊断调用。

## 保留/整合

- 当前唯一 `line.yaml`。
- 当前有效 hand-eye 文件，改为稳定名 `handeye_left_20260810_190310_robot_default_base.json`。
- `box_model.json`。
- 28081 Runtime / 19213 service 启动脚本。
- 一个统一生产预检脚本 `verify_production.py`。
- 一个统一视觉+机器人验证脚本 `detect_move_validate.py`。
- 历史 docs 合并为 README + Architecture + Operations。
- `side_ring_offline_validate.py` 的运行时 overlay 部分提取为 `side_ring_overlay.py`。

## 配置收束

- `line.yaml`: 1461 行 → 约 550 行。
- FLAT clock: 3 only。
- TILTED clock: 3 only。
- fallback clock: disabled。
- M38 branch B: disabled。
- M38 branch D: disabled。
- M37 side fallback: disabled。
- no-mouth: branch C conservative reject until M39.4。
- M39.3.2 / M39.3.3 historical online diagnostics: removed。
