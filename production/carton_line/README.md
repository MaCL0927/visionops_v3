# Carton Line 生产方案

## 1. 方案范围与板卡划分

该目录统一管理同一条产线上的四类视觉任务：

- `101`：纸隔板 5×8 小方格结构判断；
- `102`：纸筒产品检测，命令 `1/2/3` 分别检查左侧、右侧和全部区域；
- `103`：隔板小方格中心到机器人坐标的转换与寄存器写回；
- `tube_pick_vision`：TCP 触发的纸筒产品/大隔板检测，返回图像 `x/y` 与深度 `z`，不返回机器人坐标。

这些任务属于同一条生产线，但部署在两块不同的 RK3576/LB3576 开发板上，安装时必须选择对应 profile：

| profile | 部署位置 | 包含任务 | 通信方式 |
|---|---|---|---|
| `partition-tube` | 板 A | 纸隔板、小方格坐标、纸筒产品/高度检测 | Modbus-TCP |
| `tube-pick` | 板 B | 纸筒产品中心 RGB-D 坐标、大隔板类别检测 | 自定义 TCP JSON |

两组任务不会绑定安装。安装一个 profile 时，脚本只安装该板卡所需服务，并清理另一 profile 的 systemd unit。

## 2. 目录

```text
production/carton_line/
├── config/line.yaml
├── gateway/
├── tasks/
│   ├── carton_partition_check/
│   │   ├── algorithm.py
│   │   └── assets/partition_template.json
│   ├── carton_tube_check/
│   │   └── algorithm.py
│   └── tube_pick_vision/
│       ├── algorithm.py
│       ├── tcp_client.py
│       ├── service.py
│       ├── mock_scheduler.py
│       └── PROTOCOL.md
├── scripts/
└── deploy/
```

算法阈值不再拆成多个 `.env`。类别、置信度、网格、深度、高度、模板阈值、仿射坐标、端口和调试目录全部集中在：

```text
production/carton_line/config/line.yaml
```

两块板仍使用同一份配置模板，但只会启动当前 profile 对应的配置段和服务。现场配置安装到：

```text
/etc/visionops_v3/carton_line.yaml
/etc/visionops_v3/carton_line.env
```

## 3. 两块板的服务拓扑

### 3.1 板 A：`partition-tube`

```text
Orbbec / HP60C Bridge :18182
       ├── Partition Runtime :28081 -> Partition Web :18091
       └── Tube Runtime      :28082 -> Tube Web      :18092
                         \      /
                    Robot Gateway HTTP :19090
                    Partition App      :19120
                    Tube App           :19110
                    Modbus-TCP          :5046
```

Gateway 每次收到 PLC 触发后主动调用对应 Runtime 的 `infer_once`，不会读取旧的 `latest_result`。

该板安装以下服务：

```text
visionops-v3-runtime-partition.service
visionops-v3-runtime-tube.service
visionops-v3-robot-gateway.service
visionops-v3-collector-partition.service
visionops-v3-collector-tube.service
```

### 3.2 板 B：`tube-pick`

```text
Orbbec 336L Bridge :18182
        └── Pick Runtime :28083 -> Pick Web :18093
                    |
            Tube Pick TCP Client
                    |
       Robot Scheduler TCP Server :10000
                    |
             Status HTTP :19130
```

该板安装以下服务：

```text
visionops-v3-runtime-pick.service
visionops-v3-tcp-pick.service
visionops-v3-collector-pick.service
visionops-v3-runtime-pick-watchdog.service
visionops-v3-runtime-pick-watchdog.timer
```

## 4. Profile 安装

安装脚本要求显式指定 profile，不再支持不带参数的一次性全量安装。

### 4.1 板 A 安装 `partition-tube`

```bash
cd /opt/visionops_v3

sudo bash production/carton_line/deploy/install_services.sh \
  --profile partition-tube
```

安装完成后先确认模型和配置，再启动：

```bash
sudo systemctl start \
  visionops-v3-runtime-partition.service \
  visionops-v3-runtime-tube.service \
  visionops-v3-robot-gateway.service \
  visionops-v3-collector-partition.service \
  visionops-v3-collector-tube.service
```

查看状态：

```bash
systemctl status \
  visionops-v3-runtime-partition.service \
  visionops-v3-runtime-tube.service \
  visionops-v3-robot-gateway.service \
  visionops-v3-collector-partition.service \
  visionops-v3-collector-tube.service
```

Web 页面：

```text
http://<板A-IP>:18091   # 纸隔板
http://<板A-IP>:18092   # 纸筒产品/高度
```

### 4.2 板 B 安装 `tube-pick`

```bash
cd /opt/visionops_v3

sudo bash production/carton_line/deploy/install_services.sh \
  --profile tube-pick
```

安装完成后先修改机器人调度系统地址和模型路径，再启动：

```bash
sudo systemctl start \
  visionops-v3-runtime-pick.service \
  visionops-v3-tcp-pick.service \
  visionops-v3-collector-pick.service \
  visionops-v3-runtime-pick-watchdog.timer
```

查看状态：

```bash
systemctl status \
  visionops-v3-runtime-pick.service \
  visionops-v3-tcp-pick.service \
  visionops-v3-collector-pick.service \
  visionops-v3-runtime-pick-watchdog.timer
```

Web 页面和 TCP 状态接口：

```text
http://<板B-IP>:18093
http://127.0.0.1:19130/health
http://127.0.0.1:19130/api/tcp/status
```

### 4.3 安装脚本的行为

安装脚本会：

1. 合并仓库模板和已有 `/etc/visionops_v3/carton_line.yaml`；
2. 保留现场已有参数，并在升级前生成必要的配置备份；
3. 创建或更新 `/etc/visionops_v3/carton_line.env`；
4. 只复制并启用当前 profile 的 systemd unit；
5. 如果板卡以前装过另一个 profile，停止、禁用并删除另一组 unit；
6. 执行 `systemctl daemon-reload`；
7. 不自动启动当前 profile，避免模型或现场地址尚未配置时反复启动失败。

查看帮助：

```bash
bash production/carton_line/deploy/install_services.sh --help
```

### 4.4 切换 profile

一般不建议同一块生产板切换任务。确需切换时，直接重新执行另一 profile：

```bash
sudo bash production/carton_line/deploy/install_services.sh \
  --profile tube-pick
```

脚本会自动停止、禁用并移除 `partition-tube` 的服务，再安装 `tube-pick` 服务。反向切换同理。

## 5. 手动启动（不安装 systemd）

### 板 A：纸隔板 + 纸筒产品

分别在终端中运行：

```bash
cd /opt/visionops_v3

./production/carton_line/scripts/start_runtime.sh partition
./production/carton_line/scripts/start_runtime.sh tube
./production/carton_line/scripts/start_gateway.sh
./production/carton_line/scripts/start_collector.sh partition
./production/carton_line/scripts/start_collector.sh tube
```

### 板 B：纸筒产品 / 大隔板

分别在终端中运行：

```bash
cd /opt/visionops_v3

./production/carton_line/scripts/start_runtime.sh pick
./production/carton_line/scripts/start_tcp_pick.sh
./production/carton_line/scripts/start_collector.sh pick
```

查看统一配置解析结果：

```bash
python3 -m production.carton_line.launcher show-config
```

## 6. 配置文件

安装脚本会创建或升级：

```text
/etc/visionops_v3/carton_line.yaml
/etc/visionops_v3/carton_line.yaml.example
/etc/visionops_v3/carton_line.env
```

升级旧配置时，安装脚本会先生成 `.bak.<timestamp>`，再补入新任务缺失的配置键，同时保留已有现场参数。

现场调参应优先修改：

```text
/etc/visionops_v3/carton_line.yaml
```

仓库内的 `production/carton_line/config/line.yaml` 作为默认模板和版本基线。

板 B 需要重点修改：

```yaml
pick:
  tcp:
    server_host: <机器人调度系统IP>
    server_port: 10000
```

## 7. 模型目录

### 板 A：`partition-tube`

```text
models/carton_partition_check/current/model.rknn
models/carton_partition_check/current/model.yaml
models/carton_tube_check/current/model.rknn
models/carton_tube_check/current/model.yaml
```

### 板 B：`tube-pick`

```text
models/tube_pick_vision/current/model.rknn
models/tube_pick_vision/current/model.yaml
```

可在 `/etc/visionops_v3/carton_line.env` 中覆盖模型目录。

## 8. Tube Pick TCP 任务

统一配置中的 `runtimes.pick`、`collectors.pick` 和 `pick` 管理新任务。视觉盒作为 TCP Client 主动连接调度系统，消息使用 `*<JSON>#` 帧格式。

- 产品 `class_id=0`：返回彩色图像中心 `x/y`（pixel）和 D2C 对齐深度 `z`（mm）；
- 大隔板 `class_id=1`：只返回类别、置信度和数量；
- `types`/`poses` 保持为空，防止像素坐标被误作机器人坐标。

详细字段见：

```text
production/carton_line/tasks/tube_pick_vision/PROTOCOL.md
```

本地不连接机器人时可执行：

```bash
python3 -m production.carton_line.tasks.tube_pick_vision.mock_scheduler \
  --port 10000
```

然后在另一终端启动：

```bash
./production/carton_line/scripts/start_tcp_pick.sh
```


## 9. Runtime 自动重连与 watchdog

`tube-pick` profile 对实时画面增加两级恢复机制。

### 9.1 Runtime 内部自动重连

Runtime 的取流线程会检查实时帧是否过期，并在连续读取失败后自动关闭、重新打开帧源。默认参数位于：

```yaml
runtime_recovery:
  stale_frame_timeout_ms: 3000
  failure_threshold: 3
  initial_backoff_ms: 200
  max_backoff_ms: 2000
```

正常取流时这些参数不会增加额外等待。只有 Bridge/USB 短暂异常时，Runtime 才按 `200ms -> 400ms -> ... -> 2000ms` 退避重连。

状态接口会新增：

```text
frame_source.latest_frame_age_ms
frame_source.stale
frame_source.thread_alive
frame_source.consecutive_read_errors
frame_source.reconnect_count
frame_source.last_reconnect_timestamp_ms
```

检查：

```bash
curl -s http://127.0.0.1:28083/api/runtime/status | python3 -m json.tool
```

实时帧超过阈值后，`snapshot.jpg` 会返回 HTTP 503，而不是继续返回数分钟前的旧缓存图。

### 9.2 systemd watchdog

`tube-pick` 安装时同时安装并启用：

```text
visionops-v3-runtime-pick-watchdog.service
visionops-v3-runtime-pick-watchdog.timer
```

Timer 每 10 秒检查一次，只在 Pick Runtime 正处于 `preview` 模式且帧过期时采取动作：

```text
Runtime 帧过期
  -> stop_preview + start_preview
  -> 仍未恢复则重启 Pick Runtime
  -> Bridge 自身也过期时先重启 336L Bridge，再重启 Pick Runtime
```

查看 timer：

```bash
systemctl status visionops-v3-runtime-pick-watchdog.timer
systemctl list-timers | grep visionops-v3-runtime-pick-watchdog
```

查看恢复日志：

```bash
journalctl -t visionops-pick-watchdog -f
```

手动执行一次检查：

```bash
sudo systemctl start visionops-v3-runtime-pick-watchdog.service
```

watchdog 参数可在 `/etc/visionops_v3/carton_line.env` 覆盖：

```bash
VISIONOPS_PICK_WATCHDOG_STALE_MS=5000
VISIONOPS_PICK_WATCHDOG_COOLDOWN_S=30
VISIONOPS_PICK_WATCHDOG_RECOVERY_WAIT_S=3
```

### 9.3 对重启耗时的影响

正常执行 `systemctl restart visionops-v3-runtime-pick.service` 时，watchdog 不参与启动链路，也没有增加 `ExecStartPre` 等待，因此正常重启耗时基本不变。Runtime 的退避等待采用可中断睡眠，停止服务时最多主要受当前一次 HTTP 读取超时影响，默认约 1 秒以内。

只有相机或 Bridge 已经异常时，自动恢复才会等待重连；watchdog 在重启 Bridge 的异常路径中最多等待约 10 秒确认新帧到达。该等待发生在 watchdog 的独立 oneshot 进程中，不会拖慢平时的手动重启命令。

## 10. Modbus 寄存器

以下寄存器只用于 `partition-tube` profile：

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

## 11. 常用排查命令

查看当前启用的相关服务：

```bash
systemctl list-unit-files 'visionops-v3-*' --state=enabled
```

查看当前运行进程和端口：

```bash
systemctl --type=service --state=running | grep visionops-v3
sudo ss -ltnp | grep -E ':(18091|18092|18093|19090|19110|19120|19130|28081|28082|28083|5046|10000)'
```

查看日志：

```bash
journalctl -u visionops-v3-robot-gateway.service -f
journalctl -u visionops-v3-tcp-pick.service -f
```

## 12. 增加新任务

在同一产线增加任务时：

1. 新建 `tasks/<task_id>/algorithm.py` 和必要的 `assets/`；
2. 在 `line.yaml` 增加任务 Runtime、Collector 和算法配置；
3. 在 Gateway 中增加触发和寄存器映射，或增加独立协议服务；
4. 在 `launcher.py` 和 systemd 中增加实例；
5. 明确新任务属于哪个部署 profile，避免默认安装到所有板卡；
6. 在顶层 `tests/` 增加纯算法和协议测试；
7. 不在根目录增加新的任务专用 YAML、env、脚本或 service 文件。
