# Carton Palletizing（独立纸箱托盘摆放任务）

该目录是一条独立现场方案，不属于 `production/carton_line/`。第一阶段只实现 RGB 第一层逻辑：

1. 使用 OBB 模型检测并锁定托盘；
2. 依据托盘 OBB 四点建立局部坐标系，并以托盘短边构建上下居中的正方形垛型区域；
3. 在正方形区域内生成四块横竖交错的默认纸箱摆放区域；
4. 使用纸箱 OBB 与 P1～P4 做多边形 IoU、中心距离和朝向匹配；
5. 某个位置连续确认有纸箱后，隐藏该位置掩膜；
6. 四个位置全部确认后输出 `LAYER_1_COMPLETE`；
7. 托盘被纸箱遮挡时继续使用已锁定的托盘位置。

第二层及以上的深度基准、层间状态机和机器人坐标/通信暂未加入。

## 模型要求

模型必须是 OBB 模型，类别固定为：

```text
0 = box（纸箱）
1 = tray（托盘）
```

模型包放到：

```text
/opt/visionops_v3/models/carton_palletizing/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 至少需要正确声明 OBB 任务和类别，例如：

```yaml
model_id: carton_palletizing_obb
model_name: carton_palletizing_obb
task: obb
target_platform: rk3576
input_size: [640, 640]
class_names: [box, tray]
```

业务配置已经固定为：

```yaml
runtime:
  accepted_task_types: [obb, obb_detection]

task:
  algorithm:
    classes:
      box_class_ids: [0]
      tray_class_ids: [1]
    geometry:
      require_obb: true
      footprint_mode: centered_square_by_short_edge
      footprint_fill_ratio: 1.0
```

若 Runtime 返回的 `task_type` 不是 `obb`/`obb_detection`，业务应用会直接给出明确错误，而不会按普通 detection 继续运行。

## 手动启动

先启动 336L Camera Bridge，然后分别运行：

```bash
cd /opt/visionops_v3
./production/carton_palletizing/scripts/start_runtime.sh
./production/carton_palletizing/scripts/start_app.sh
./production/carton_palletizing/scripts/start_collector.sh
```

默认端口：

| 服务 | 地址 |
|---|---|
| RKNN Runtime | `127.0.0.1:28084` |
| 第一层业务应用 | `127.0.0.1:19210` |
| Collector Web | `0.0.0.0:18094` |

浏览器打开：

```text
http://<视觉盒子IP>:18094
```

生产模式会根据实时舞台尺寸显式计算最大 `contain` 尺寸：保持原始长宽比、完整显示整张图，并主动放大到当前区域允许的最大尺寸。底部模型/FPS 信息位于画面之外，不再压缩或遮挡图像。

黄色为未占用位置，绿色为当前推荐位置；位置确认占用后不再绘制掩膜。

## 调试接口

```bash
# Runtime 应确认 task_type=obb
curl http://127.0.0.1:28084/api/runtime/status | python3 -m json.tool

# 应用健康
curl http://127.0.0.1:19210/health | python3 -m json.tool

# 主动执行一次 Runtime 推理 + 第一层判断
curl -X POST http://127.0.0.1:19210/api/app/evaluate_once \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool

# 查看最近决策
curl http://127.0.0.1:19210/api/app/latest_decision | python3 -m json.tool

# 更换托盘或重新测试时清空托盘锁定和四个占位状态
curl -X POST http://127.0.0.1:19210/api/app/reset \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
```

## 第一层正方形垛型与顺序

托盘检测框可能是上下长、左右窄的矩形。算法会计算托盘 OBB 的平均宽高，以短边作为正方形边长，并在长边方向两端等距留白：

- 托盘竖向较长时：左右占满、上下居中；
- 托盘横向较长时：上下占满、左右居中；
- 托盘旋转时：正方形区域和四块掩膜一起旋转。

四块 `polygon_norm` 相对于这个正方形区域定义：

- P1：左上横向；
- P2：右上竖向；
- P3：左下竖向；
- P4：右下横向。

推荐顺序为 `P3 -> P1 -> P2 -> P4`，即从左下角开始，按画面中的顺时针方向依次摆放。现场调试时先保持托盘为空，观察正方形垛型和四块掩膜是否与实际落位一致，再微调 `footprint_fill_ratio` 或各 slot 的 `polygon_norm`。

## Python 版本

新增业务代码兼容 LB3576 系统自带的 Python 3.8，不再使用 `tuple[...]`、`dict[...]`、`X | None` 等 Python 3.9/3.10 才支持的运行时类型写法。

## 开机自启

```bash
cd /opt/visionops_v3
sudo bash production/carton_palletizing/deploy/install_services.sh
```

安装后主要配置位于：

```text
/etc/visionops_v3/carton_palletizing.yaml
/etc/visionops_v3/carton_palletizing.env
```
