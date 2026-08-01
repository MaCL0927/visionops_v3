# VisionOps v3 仓库整理报告

审计日期：2026-08-01

## 1. 审计范围

本次整理范围限定在 `/home/pc/桌面/visionops_v3`，未修改仓库外文件，未执行 `git reset --hard`、`git clean -fd` 或 `git clean -fdx`。

已审计内容：

- Git 状态、当前分支、最近提交。
- 根目录结构、顶层目录体积、已跟踪文件和未跟踪/忽略文件。
- `AGENTS.md`、`.gitignore`、根目录 `README.md`、`docs/`。
- `apps/collector_web`、`apps/server_api`、`edge/runtime_cpp`、`edge/camera_bridge`、`edge/gateway_adapter`、`edge/modbus_adapter`。
- `production/` 下 carton line、carton palletizing、detergent grasp 和 foam-ring grasp。
- `training/pipeline`、`tools/`、`tests/`、`configs/`、`interfaces/`、`scripts/`。
- `yolo26n.pt`、AMP、Ultralytics、训练入口和模型路径引用。

## 2. 当前架构摘要

VisionOps v3 当前由服务端闭环和边缘端生产链路组成：

```text
Server API:
  batch -> annotation/review -> dataset -> training job
    -> ONNX/RKNN export -> model package -> publish/sync

Edge:
  Camera Bridge / SDK Bridge
    -> C++ RKNN Runtime
    -> Collector Web
    -> Production App / Gateway / Modbus / Robot Client
    -> PLC / 机器人调度系统 / 上位机
```

主要模块分类：

- A 生产运行必需：`edge/runtime_cpp`、`edge/camera_bridge`、`apps/collector_web`、`production/*`、`edge/modbus_adapter`。
- B 开发/构建/部署必需：`CMakeLists.txt`、`scripts/`、`edge/deploy/`、各产线 `deploy/` 和 `systemd/`。
- C 有效测试：`tests/unit`、`tests/integration`、`edge/runtime_cpp/tests`。
- D 文档和示例：`README.md`、`docs/`、`interfaces/`、`configs/*.example.yaml`、`*.env.example`。
- E 生成缓存：`__pycache__/`、`*.pyc`、`.pytest_cache/`。
- F 本地运行数据：`server_data/`、`models/` 中真实权重。
- H 保留但需要人工确认的历史资料：`docs/M*.md`、部分 production 下阶段 notes。

## 3. 删除的文件或目录

本次只删除可确认的生成缓存：

- `.pytest_cache/`
- 所有 `__pycache__/` 目录。
- 所有 `*.pyc` 文件。
- 若存在则删除 `CMakeFiles/`、`CMakeCache.txt`、`compile_commands.json`。

删除理由和证据：

- 这些文件由 Python、pytest 或 CMake 自动生成。
- `.gitignore` 已覆盖这些模式。
- `git ls-files` 未显示这些缓存为有意跟踪内容。
- 清理后源码、配置和脚本不应引用这些缓存文件。

未删除：

- `server_data/` 下运行数据目录。该目录可能包含本地服务端状态；仅 `.gitkeep` 和 `registry/devices.json` 可见，未做破坏性清空。
- `models/pretrained/*.pt` 和 `.onnx`。这些是真实本地预训练权重，被 `.gitignore` 忽略但可能用于训练/标注，不作为本次删除对象。
- `production/foam_ring_grasp/config/box_model_overlay.jpg`。虽然是图片，但它属于泡沫圆环任务配置/诊断资产，未删除。

## 4. 移动的文件

移动模型：

```text
yolo26n.pt
  -> models/pretrained/yolo26n.pt
```

原始 SHA-256：

```text
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

目标 SHA-256：

```text
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

证据：

- 源文件未被 Git 跟踪，因此使用普通 `mv`。
- 目标路径此前不存在。
- 移动前后 SHA-256 一致。
- `models/pretrained/yolo26n.pt` 仍被 `.gitignore` 中的 `*.pt` 规则忽略，不会进入 Git。
- 新增 `models/pretrained/.gitkeep`，用于保留规范目录。

## 5. AMP 与训练路径修改

修改文件：

- `training/pipeline/stages/train.py`
- `apps/server_api/backend/services/annotation_service.py`
- `tests/unit/test_server_api_services.py`
- `tests/unit/test_sam_annotation_service.py`

修改摘要：

- 训练 pipeline 会把 `pretrained_model` 解析为基于 `ctx.project_root` 的绝对路径，不依赖调用者 cwd。
- YOLO 训练命令从 `ctx.project_root` cwd 改为 `ctx.work_dir` cwd，防止第三方逻辑在仓库根目录生成临时文件。
- `amp=True` 时，pipeline 要求 `models/pretrained/yolo26n.pt` 存在，并在 job work 目录创建链接或副本。
- Server API 标注器快速学习入口使用同样策略，在 batch quick 目录运行 YOLO，不再以 repo root 作为 cwd。
- 如果缺少 `models/pretrained/yolo26n.pt`，会给出明确错误，不会静默让训练逻辑在仓库根目录下载。

## 6. README 修改摘要

根目录 `README.md` 已重写为当前仓库的统一入口，覆盖：

- VisionOps v3 项目定位和主链路。
- Server API、Collector Web、Camera Bridge、C++ Runtime、Production App、Gateway/Modbus 的边界。
- 顶层目录说明。
- 支持任务类型：detection、OBB、segmentation、classification。
- 数据采集到训练发布再到边缘部署的流程。
- `models/pretrained/` 和 `yolo26n.pt` 规范位置。
- Runtime、Collector Web、Server API 常用启动和检查命令。
- HP60C、Orbbec 336L、foam-ring grasp 任务索引。
- 测试、部署同步和仓库卫生规则。

## 7. `.gitignore` 修改摘要

新增允许规则：

```text
!/models/pretrained/
!/models/pretrained/.gitkeep
```

目的：

- 继续忽略真实模型权重和大文件。
- 保留 `models/pretrained/` 规范目录。
- 防止 `models/pretrained/yolo26n.pt` 之外的同类权重被提交。

## 8. 保留但疑似历史残留的文件

以下内容保留，原因是它们仍可能作为任务文档、部署记录或硬件诊断依据：

- `docs/M*.md`：大量阶段文档仍被 README、AGENTS 或任务文档引用；未证明无价值。
- `production/*/M*_NOTES.md`：与具体 production 任务实现和现场调试有关，未删除。
- `production/foam_ring_grasp/M35.3_PAIRED_AXIS_VISUALIZATION_NOTES.md`：与当前 M35.4 泡沫圆环任务上下文相关。
- `server_data/registry/devices.json`：本地设备注册状态文件，被 `.gitignore` 忽略；可能是本机运行状态，未删除。
- `models/pretrained/*.pt`、`*.onnx`：本地预训练权重，未进入 Git，但训练和标注可能依赖。

## 9. 测试与检查记录

已执行：

- `python -m py_compile training/pipeline/stages/train.py apps/server_api/backend/services/annotation_service.py tests/unit/test_server_api_services.py tests/unit/test_sam_annotation_service.py`：通过。
- `python -m pytest tests/unit/test_server_api_services.py tests/unit/test_sam_annotation_service.py`：20 passed。
- `python -m pytest tests/unit/test_config_validation.py tests/unit/test_render_runtime_env.py`：6 passed。
- `python -m pytest tests/unit/test_foam_ring_rim_pinch_geometry.py tests/unit/test_foam_ring_axis_visualization.py`：21 passed。
- `python -m pytest tests/unit/test_carton_box_grasp_vision.py`：10 passed。
- `python -m pytest tests/unit`：170 passed。
- `git diff --check`：通过。
- 搜索旧模型路径和裸文件名引用：未发现运行代码依赖仓库根目录 `./yolo26n.pt`。
- 生成缓存复查：`__pycache__/`、`*.pyc`、`.pytest_cache/`、CMake 缓存均已清理。

补充修复：

- 完整 `tests/unit` 初次运行时，`tests/unit/test_carton_box_grasp_vision.py` 有 3 个失败。
- 根因是 `production/carton_palletizing/config/line.yaml` 覆盖了 box grasp 推荐配置，把 `grasp_inward_ratio` 设为 `0`、`grasp_extra_inward_ratio` 设为 `0.1`。
- 已恢复为 README、默认配置和测试一致的 `grasp_inward_ratio: 0.18`、`grasp_extra_inward_ratio: 0.05`。
- 修复后 `tests/unit/test_carton_box_grasp_vision.py` 与完整 `tests/unit` 均通过。

硬件相关测试未在本机完成：

- RKNN / RGA 真机推理。
- HP60C / Orbbec 336L 真实取流。
- 机器人、PLC、Modbus 真实联调。
- 生产线长期运行稳定性。

## 10. 尚未解决的风险

- `AGENTS.md` 在本次开始前已有未提交修改，本次未回退也未整理该文件。
- `server_data/` 占用较大，但可能包含本地服务端状态，本次未清空。
- 真实训练、ONNX 导出、RKNN 转换仍依赖本机/板端环境和 conda 环境，自动化测试只能覆盖路径解析与服务逻辑。
- Ultralytics 内部 AMP 检查模型名如果随版本变化，仍可能在 job work 目录产生新的小权重文件；当前改动保证不会落到仓库根目录。
