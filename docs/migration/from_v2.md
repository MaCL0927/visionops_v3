# 从 VisionOps v2 迁移到 v3

## 1. 原则

1. 按能力迁移，不按旧目录复制。
2. C++ Runtime、Collector、Camera Bridge 和 Modbus 基础库保持平台通用。
3. 现场算法、PLC 协议和标定参数放入 `production/<line_id>/`。
4. 不保留无调用方脚本、一次性调试入口、旧服务副本和历史 `.env`。
5. 模型、数据、日志和设备私密配置不进入仓库。

## 2. 当前纸箱产线迁移结果

v2 中经现场验证的能力已按职责拆分：

- 小方格模板判断 -> `production/carton_line/tasks/carton_partition_check/`；
- 纸筒 RGB-Depth 高度判断 -> `production/carton_line/tasks/carton_tube_check/`；
- PLC 触发、统一寄存器和双任务调度 -> `production/carton_line/gateway/`；
- 阈值、端口和坐标参数 -> `production/carton_line/config/line.yaml`；
- systemd 和启动脚本 -> `production/carton_line/deploy/`、`scripts/`。

没有迁移 v2 的旧 Web、旧 Python 推理主链路、重复 Modbus Server 或散落的网络请求代码。

## 3. 后续迁移要求

新增业务时必须先明确：

- 标准 Runtime 输出；
- 任务算法输入输出；
- PLC 触发和寄存器；
- 模型目录和类别映射；
- 相机与深度依赖；
- 可重复的测试样例和真机验收方式。

然后在对应产线目录内完成，不得重新向根目录增加任务专用配置和启动脚本。
