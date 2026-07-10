# Carton Line 生产方案

## 1. 方案范围

该目录统一管理同一条产线上的三个 PLC 任务：

- `101`：纸隔板 5×8 小方格结构判断；
- `102`：纸筒产品检测，命令 `1/2/3` 分别检查左侧、右侧和全部区域；
- `103`：隔板小方格中心到机器人坐标的转换与寄存器写回。

## 2. 目录

```text
production/carton_line/
├── config/line.yaml
├── gateway/
├── tasks/
│   ├── carton_partition_check/
│   │   ├── algorithm.py
│   │   └── assets/partition_template.json
│   └── carton_tube_check/
│       └── algorithm.py
├── scripts/
└── deploy/
```

算法阈值不再拆成多个 `.env`。类别、置信度、网格、深度、高度、模板阈值、仿射坐标、端口和调试目录全部集中在：

```text
production/carton_line/config/line.yaml
```

## 3. 服务拓扑

```text
Orbbec / HP60C Bridge :18182
       ├── Partition Runtime :28081 -> Partition Web :18091
       └── Tube Runtime      :28082 -> Tube Web      :18092
                         \      /
                    Gateway HTTP :19090
                    Partition App :19120
                    Tube App      :19110
                    Modbus-TCP    :5046
```

Gateway 每次收到 PLC 触发后主动调用对应 Runtime 的 `infer_once`，不会读取旧的 `latest_result`。

## 4. 手动启动

```bash
cd /opt/visionops_v3

./production/carton_line/scripts/start_runtime.sh partition
./production/carton_line/scripts/start_runtime.sh tube
./production/carton_line/scripts/start_gateway.sh
./production/carton_line/scripts/start_collector.sh partition
./production/carton_line/scripts/start_collector.sh tube
```

查看统一配置解析结果：

```bash
python3 -m production.carton_line.launcher show-config
```

## 5. systemd

```bash
sudo bash production/carton_line/deploy/install_services.sh
```

安装脚本会创建：

```text
/etc/visionops_v3/carton_line.yaml
/etc/visionops_v3/carton_line.env
```

现场调参应优先修改 `/etc/visionops_v3/carton_line.yaml`，仓库内文件作为默认模板和版本基线。

## 6. 模型目录

```text
models/carton_partition_check/current/model.rknn
models/carton_partition_check/current/model.yaml
models/carton_tube_check/current/model.rknn
models/carton_tube_check/current/model.yaml
```

可在 `/etc/visionops_v3/carton_line.env` 中覆盖模型目录。

## 7. Modbus 寄存器

| 地址 | 含义 |
|---:|---|
| 0 | 视觉心跳 |
| 1 | 隔板结果：0 空闲、1 正常、2 异常 |
| 2 | 纸筒结果：0 空闲、1 正常、2 异常 |
| 3 | 坐标结果：0 空闲、1 正常、2 异常 |
| 20–99 | 40 个槽位的 X/Y 坐标 |
| 100 | PLC 心跳 |
| 101 | 隔板触发 |
| 102 | 纸筒触发：1 左、2 右、3 全部 |
| 103 | 坐标识别触发 |

## 8. 增加新任务

在同一产线增加任务时：

1. 新建 `tasks/<task_id>/algorithm.py` 和必要的 `assets/`；
2. 在 `line.yaml` 增加任务 Runtime、Collector 和算法配置；
3. 在 Gateway 中增加触发和寄存器映射；
4. 在 `launcher.py` 和 systemd 中增加实例；
5. 在顶层 `tests/` 增加纯算法和协议测试；
6. 不在根目录增加新的任务专用 YAML、env、脚本或 service 文件。
