# 配置分层设计

## 1. 设计原则

配置必须区分人工意图、任务定义、应用行为和运行时生成结果。源配置可进入 Git，设备私密值和生成环境文件不得进入 Git。

```text
edge 配置 + task 配置 + app 配置
              │
              v
       校验、合并与渲染
              │
              v
 runtime generated env / service config
```

配置加载必须确定、可审计、失败即明确报错。禁止静默猜测关键参数。

## 2. Edge 配置

目录：`configs/edge/`

Edge 配置描述设备和硬件能力，建议包含：

- `device_id`、设备型号和部署环境。
- 平台类型，例如 `rk3588` 或 `lb3576`。
- 相机驱动、来源、像素格式、分辨率和目标帧率。
- NPU 核心策略、内存限制和线程参数。
- 本机服务端点、数据目录和日志策略。
- Gateway、Modbus、NTP 和网络接口引用。

设备密码、私钥、Gateway Token 等不写入该文件，只记录对应的环境变量名或密钥引用。

推荐命名：

```text
configs/edge/base.yaml
configs/edge/rk3588.yaml
configs/edge/lb3576.yaml
configs/edge/devices/<device_id>.yaml
```

设备文件只覆盖差异，不复制完整基础配置。

## 3. Task 配置

目录：`configs/task/`

Task 配置描述模型任务和推理语义，与具体设备身份解耦，建议包含：

- 任务类型和任务版本。
- 输入尺寸、颜色空间和预处理规范。
- 类别表及类别版本。
- 后处理类型、阈值、NMS、TopK 或 ROI 策略。
- 模型清单引用和兼容的目标平台。
- 标准化输出字段与业务标签映射。

Task 配置不包含 SSH 地址、相机密码、Web 端口或 PLC 寄存器地址。

## 4. App 配置

目录：`configs/app/`

App 配置描述应用行为和业务集成，建议包含：

- Collector Web 的显示、采集和诊断选项。
- 生产模式的触发、节流、聚合与结果保留策略。
- Gateway 消息字段映射和上传策略。
- Modbus 寄存器映射、字节序、心跳和故障安全值。
- 服务端 API 地址和非敏感功能开关。

App 配置可以引用 Task 的标准化结果字段，但不得引用 RKNN 原始张量索引。

## 5. Runtime Generated Env

目录：`configs/runtime/`

运行时配置由校验后的 edge、task、app 配置和部署上下文生成，典型产物包括：

```text
configs/runtime/generated/visionops-camera.env
configs/runtime/generated/visionops-runtime.env
configs/runtime/generated/visionops-collector.env
configs/runtime/generated/visionops-gateway.env
configs/runtime/generated/visionops-modbus.env
configs/runtime/generated/effective-config.yaml
```

真实生成文件不得提交 Git。生成结果应包含或关联：

- 配置 schema 版本。
- 源配置文件和内容摘要。
- 模型包版本与校验值。
- 生成工具版本和生成时间。
- 目标设备 ID 与平台。

`effective-config.yaml` 用于诊断最终生效值，但输出时必须遮蔽敏感字段。

## 6. 优先级与覆盖规则

建议采用从低到高的固定优先级：

```text
内置安全默认值
  < edge/base
  < 平台配置
  < 设备配置
  < task 配置
  < app 配置
  < 部署时受控覆盖
  < 密钥或环境变量注入
```

并非所有字段都允许被高层覆盖。任务输入规范不能由 Web 临时参数任意改变，设备能力不能由 Task 配置伪造。配置 schema 应声明字段所有者和可覆盖范围。

命令行覆盖只用于开发、诊断或明确的部署操作，生产服务启动命令应引用生成文件，避免大量不可审计参数。

## 7. 校验与应用

配置生成流程应执行：

1. 语法和 schema 校验。
2. 跨文件引用与版本校验。
3. 平台能力与模型兼容性校验。
4. 端口、路径和服务依赖冲突检查。
5. 敏感字段检查。
6. 生成临时目录。
7. 完整校验后原子替换目标配置。

配置应用失败时保留上一份有效配置，并返回可定位的错误。禁止部分服务使用新配置、部分服务仍使用旧配置而没有版本标识。

## 8. 配置变更分类

- **热更新**：显示选项、非关键阈值、上传开关等，经组件明确支持后生效。
- **服务重启**：相机来源、监听端口、线程与资源参数。
- **模型切换**：模型包、输入规范、类别或后处理版本变化，应走事务化切换与回滚。
- **设备重启**：驱动、网络底层或设备节点相关变化，必须显式提示。

每个配置项应标注生效方式，Collector Web 不得假设所有配置都能热更新。

## 9. M1 配置工具

M1 提供统一校验与 env 渲染骨架，工具只依赖 Python 标准库和 PyYAML。JSON Schema 位于：

```text
interfaces/schemas/config.schema.json
```

示例配置使用 `*.example.yaml` 命名。edge 配置允许按从低到高的优先级重复传入，首个文件必须是完整的 `kind: edge`，后续平台或设备差异文件使用 `kind: edge_overlay`。task 使用单个配置，app 可以重复传入多个独立应用配置。

校验示例：

```bash
python tools/config/validate_config.py \
  --edge configs/edge/base.example.yaml \
  --edge configs/edge/rk3588.example.yaml \
  --task configs/task/detection.example.yaml \
  --app configs/app/collector.example.yaml \
  --app configs/app/gateway_modbus.example.yaml
```

渲染到临时位置的示例：

```bash
python tools/config/render_runtime_env.py \
  --edge configs/edge/base.example.yaml \
  --edge configs/edge/rk3588.example.yaml \
  --task configs/task/detection.example.yaml \
  --app configs/app/collector.example.yaml \
  --app configs/app/gateway_modbus.example.yaml \
  --output /tmp/visionops-runtime.env
```

不传 `--output` 时只输出到标准输出。生成内容包含配置版本、源文件绝对路径、全部源文件的 SHA-256 摘要、UTC 生成时间和渲染器版本。真实运行时文件仍由后续部署工具写入受控目录，不进入 Git。

校验器当前执行以下跨文件规则：

- edge、task、app 的必要字段和配置版本。
- `rk3588`、`rk3576` 平台范围以及任务与设备的平台兼容性。
- 本地 `listen_port`、`metrics_port` 的范围和冲突。
- `_path`、`_dir`、`_root` 路径字段的绝对路径与主目录依赖检查。
- `password`、`token`、`secret`、`private_key` 等敏感字段只能使用 `env:NAME` 或 `${NAME}` 引用。
