# Clean Production Baseline Manifest — 2026-08-17

本次以用户上传的当前可抓取版本为唯一基线，只整理 production 目录，不修改已经现场验证的抓取算法、hand-eye、box model 或 `line.yaml` 参数。

## 保留

- 当前唯一 `config/line.yaml`（schema 7.2）。
- 当前 production workspace local hand-eye：`config/handeye_calibration.json`。
- 当前有效 `config/box_model.json`。
- Runtime / online service 两个启动脚本。
- `verify_production.py`。
- 唯一现场机器人验证/抓取脚本 `detect_move_validate.py`。
- 8 点箱体重标定工具 `fit_box_model_8point.py`。
- 当前 M39.3 visible-mouth FLAT/TILTED 在线链。
- 当前 M39.4.0.1 / M39.4.1 / M39.4.2.2 pure-side 在线链。
- 当前仍被 active import graph 复用的 M38/M37 几何模块，即使文件名带历史 milestone，也不删除。

## 删除

### 标定中间/历史文件

- `config/box_model.json.bak_20260817_032157`
- `config/box_model_8point_candidate.json`
- `config/box_model_8point_candidate_overlay.jpg`
- `config/box_model_8point_candidate_points.json`
- `config/handeye_left_20260810_190310_robot_default_base.json`

其中 `box_model_8point_candidate.json` 与当前 `box_model.json` SHA256 完全相同，已经完成安装，不再重复保留。

### 已退出生产链的历史验证模块

- `tasks/foam_ring_grasp_vision/offline_validate.py`（M35.4 CLI）
- `tasks/foam_ring_grasp_vision/online_validate.py`（历史 one-shot validator）
- `tasks/foam_ring_grasp_vision/ring_prior_surface.py`（M39.3.2 独立诊断实现，当前无调用）
- `tasks/foam_ring_grasp_vision/side_ring_offline_validate.py`（M37 offline validator）

这些文件均不在当前 service / online processor / robot script 的 active import graph 中。

## 文档同步

旧文档仍残留 90/35/18 mm、entry-only/no-close 等历史描述。本次统一到当前真实配置：

```text
side PREGRASP outside ENTRY = 20 mm
side GRASP inside ENTRY     = 39 mm
PREGRASP → GRASP            = 59 mm
ENTRY                       = diagnostic only
robot cycle                 = full GRASP + CLOSE
```

## 整理结果

```text
before: 49 files, ~1.7 MB
after : 40 files, ~1.5 MB
```

后续开发应继续以这 40 文件干净基线增量修改，不把 candidate、backup、replay、阶段性 test script 或历史 milestone 文档重新放回 production 包。
