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
