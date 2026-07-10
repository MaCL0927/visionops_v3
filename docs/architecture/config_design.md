# 配置设计

## 1. 配置分类

VisionOps v3 使用两类配置：

### 通用平台配置

位于 `configs/`：

- `configs/edge/`：硬件平台与设备能力示例；
- `configs/task/`：模型任务与输入输出规范示例；
- `configs/app/`：通用 App 示例；
- `configs/server/`：服务端配置示例；
- `configs/runtime/`：生成配置说明。

### 现场产线配置

位于：

```text
production/<line_id>/config/line.yaml
```

一份主 YAML 应覆盖该产线的 Runtime 实例、Collector 实例、Gateway、Modbus、算法阈值、相机来源、坐标标定和调试输出。不得再为同一任务同时维护散落的 env、YAML 和 shell 常量。

## 2. 生效优先级

```text
代码安全默认值
  < 仓库产线 YAML
  < /etc/visionops_v3/<line_id>.yaml
  < systemd EnvironmentFile 中允许覆盖的路径
```

环境变量只用于根目录、虚拟环境、模型目录和配置文件路径等部署差异，不承载大量业务阈值。

## 3. 密钥和运行数据

以下内容不得进入 Git：

- 密码、Token、私钥和真实设备凭据；
- 运行时 `.env`；
- 模型、数据集、采集图片和日志；
- 设备生成的 effective config 和 debug 结果。

仓库仅保留 `*.env.example`。

## 4. 通用配置工具

通用示例仍可使用：

```bash
python tools/config/validate_config.py \
  --edge configs/edge/base.example.yaml \
  --edge configs/edge/rk3576.example.yaml \
  --task configs/task/detection.example.yaml \
  --app configs/app/collector.example.yaml
```

生产线主 YAML 由对应 `production/<line_id>/gateway/config.py` 和 launcher 校验。
