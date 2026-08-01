# VisionOps v3 AI Agent 协作规则

## 1. 架构约束

1. 边缘主链路为 `Camera Bridge -> C++ RKNN Runtime -> Production Gateway/Modbus`，Collector Web 是管理与观察入口。
2. RKNN 模型加载、预处理、NPU 推理和生产后处理默认使用 C++。
3. `apps/`、`edge/` 和 `training/` 只放可复用平台能力。
4. 现场算法、PLC 语义、标定参数和产线部署必须放入 `production/<line_id>/`。
5. 只有至少被两条产线复用、边界稳定的能力，才允许从 `production/` 上移到平台层。

## 2. 目录规则

1. 新任务放在 `production/<line_id>/tasks/<task_id>/`。
2. 同一产线的配置集中在 `production/<line_id>/config/line.yaml`。
3. 同一产线的 systemd、安装和启动入口放在该产线的 `deploy/` 与 `scripts/`。
4. 不得在根目录、`configs/app/`、`edge/gateway_adapter/` 和根 `scripts/` 中继续增加任务专用文件。
5. 跨进程、跨语言契约放在 `interfaces/`，并包含版本字段。

## 3. 配置与安全

1. 不提交真实 `.env`、密码、Token、私钥或设备私密配置。
2. 仓库只保留 `*.env.example`；实际配置安装到 `/etc/visionops_v3/`。
3. 不提交 `.pt`、`.onnx`、`.rknn`、数据集、图片、视频、日志、诊断包和压缩包。
4. 业务阈值优先进入产线 YAML，不使用大量环境变量分散配置。
5. 代码不得依赖开发者用户名、桌面路径或固定 Conda 安装目录。

## 4. 实现规则

1. Runtime 不读取 PLC 寄存器，不包含现场类别名和业务阈值。
2. Collector 不直接连接相机、不加载模型、不执行任务算法。
3. Gateway 主动调用 Runtime 的 `infer_once`，不得把旧 `latest_result` 当成一次新触发结果。
4. Modbus 基础库不提供隐式默认寄存器表；具体产线必须显式定义。
5. 新依赖必须说明用途、许可证、ARM 支持和部署成本。

## 5. 验证要求

1. 修改 Python 后执行语法检查和相关 pytest。
2. 修改 C++ Runtime 后至少完成 CMake 构建和相关 fixture。
3. 修改 shell/systemd 后执行 `bash -n` 并检查所有路径。
4. 硬件相关结果必须标明真机是否验证，不能用 Mock 结果代替。
5. 完成修改时同步更新当前 README 和架构文档，不保留过程性交接文档。

# VisionOps v3 — Codex Repository Instructions

## 1. Project overview

VisionOps v3 is an industrial computer-vision platform covering the complete workflow from data collection to production deployment.

The repository contains several major layers:

* Server API and Web management console.
* Data collection, annotation, review, training and model publishing pipeline.
* C++ edge Runtime for RK3588, RK3576 and related Rockchip devices.
* Camera bridge services, including HP60C and Orbbec Gemini 336L.
* Production application logic for individual factory tasks.
* Robot, Modbus TCP/RTU and other industrial communication adapters.
* Deployment scripts, systemd service files, configuration files and hardware-side diagnostic tools.

Supported model tasks include detection, OBB detection, segmentation and classification.

The current repository code, configuration and Git history are the source of truth. This document provides context but must not override verified code behavior.

## 2. Important architecture conventions

For newer production vision applications, preserve the verified two-thread pipeline:

* The inference thread continuously requests the Runtime.
* A capacity-1 latest-only queue connects inference to post-processing.
* Explicit trigger and request_id results must be reliably retained and must not be overwritten by continuous-frame results.
* Keep segmented end-to-end timing and p50/p95 statistics.

For local App-to-Runtime communication, preserve the verified Raw Local HTTP design:

* Use TCP_NODELAY.
* Combine HTTP headers and request body into one sendall operation.
* Queue raw response bytes.
* Decode JSON later in the post-processing thread.
* A configurable urllib fallback may remain available when raw local communication fails.

Do not replace these mechanisms with a simpler synchronous urllib implementation unless a task explicitly requires it.

## 3. Current foam-ring task status

The active foam-ring grasp task is under:

`production/foam_ring_grasp/`

Its recent development stages include:

* Foam-ring and ring-mouth segmentation and pairing.
* RGB-D geometric validation.
* Ring-axis estimation.
* Grasp-point selection and collision evaluation.
* Pneumatic gripper 3D model and fitting-layout corrections.
* M35.3 paired-axis visualization.
* M35.4 directed fixed-length 3D axis-rod projection.

M35.4 uses:

* A fixed-length 3D rod centered at the ring center.
* A near endpoint pointing toward the camera.
* A far endpoint pointing away from the camera.
* Red solid near-end encoding.
* Cyan dashed far-end encoding.
* Numeric labels such as tilt, projected length, depth difference and axis vector.

The configuration section `axis_direction` controls the added M35.4 diagnostic projection and visualization work.

Turning this diagnostic option off must not break the core ring-axis calculation required by tilt estimation, grasp pose and collision checking.

## 4. Repository hygiene rules

Before deleting or moving anything:

1. Inspect Git status.
2. Check whether the file is tracked.
3. Search for code, configuration, documentation and script references.
4. Determine whether it is used by production, deployment, testing, data migration or hardware diagnostics.
5. Keep ambiguous files and report them instead of deleting them.

Never use broad destructive commands such as:

* `git clean -fdx`
* `git reset --hard`
* Unreviewed recursive deletion patterns
* Deletion based only on filename age or naming style

Do not discard existing uncommitted user changes.

Generated artifacts, caches, temporary output images, benchmark outputs and obsolete package copies should not remain in the repository unless they are intentionally maintained fixtures.

Do not delete the following merely because they appear old:

* Production task configuration.
* Camera bridge code.
* Runtime code.
* Robot communication code.
* systemd service or timer files.
* Calibration tools and calibration examples.
* Deployment and recovery scripts.
* Hardware diagnostic scripts.
* Tests covering active behavior.
* Model metadata and model-conversion configuration.

## 5. Model-file convention

Pretrained model files must not be stored in the repository root.

The canonical directory is:

`models/pretrained/`

The model:

`yolo26n.pt`

must reside at:

`models/pretrained/yolo26n.pt`

The root-level copy is believed to be associated with Ultralytics training-pipeline AMP validation, but this must be verified by tracing the implementation.

When changing this path:

* Search the entire repository for `yolo26n.pt`.
* Search for AMP validation, `check_amp`, `amp`, Ultralytics model initialization and training startup logic.
* Identify why the weight appeared in the repository root.
* Move the file with `git mv` when tracked, otherwise use a normal filesystem move.
* Update repository-owned code and configuration to resolve the model relative to the repository root.
* Do not depend on the caller's current working directory.
* Do not patch Python site-packages.
* Prevent Ultralytics or the training pipeline from downloading or recreating the file in the repository root.
* Document the canonical pretrained-model path in the root README.

## 6. Testing and validation

After changes:

* Run relevant unit tests.
* Run Python syntax or import checks for modified modules.
* Run configuration parsing tests where applicable.
* Run `git diff --check`.
* Search again for stale model paths.
* Confirm that no production entry point refers to a deleted file.
* Review the complete Git diff.

Hardware-dependent tests that require an RK board, camera, robot or active Runtime service may be skipped, but the skipped checks and reasons must be reported.

Do not claim the entire repository passes unless the complete relevant test suite actually completed successfully.

## 7. Documentation rules

Keep the root README concise, current and operational.

It should explain:

* What VisionOps v3 is.
* Repository architecture.
* Main directory structure.
* Supported task types.
* Server, Runtime, camera bridge and production application relationships.
* Development and deployment workflow.
* Training and model publishing workflow.
* Model-file directory conventions.
* Basic startup and validation commands.
* Hardware-dependent limitations.
* Links to detailed documents under `docs/`.

Do not fill the root README with a chronological dump of every historical patch version. Move detailed history to a dedicated changelog or task documentation when it remains useful.
