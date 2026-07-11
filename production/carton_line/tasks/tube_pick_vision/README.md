# tube_pick_vision

该任务使用一个 detection 模型检测两类对象，并要求使用能够提供 D2C 对齐深度的 Orbbec 336L Camera Bridge。仅有 RGB 的 HP60C 无法生成产品 `z`。

- `class_id=0`：纸筒产品。对外返回检测框中心在彩色图像中的 `x/y`，以及 D2C 对齐深度图中的 `z`（毫米）。
- `class_id=1`：上下层之间的大隔板。对外只返回类别、置信度和数量，不发送位置。

## TCP 角色与帧格式

视觉盒是 TCP Client，主动连接机器人调度系统的 TCP Server。每帧为：

```text
*<UTF-8 JSON>#
```

收到调度触发后，服务主动调用 pick Runtime 的 `/api/runtime/infer_once`。只有检测到产品时才读取深度图。

## 坐标定义

```text
x/y：Runtime 所见彩色图像像素坐标
z：D2C 对齐深度图中中心邻域的有效深度中位数，单位 mm
```

本任务不执行机器人坐标或 `base_link` 坐标转换。响应保留 `types: []` 和 `poses: []`，避免旧调度程序把像素坐标误当成机器人坐标。

## 模型包

模型包目录：

```text
models/tube_pick_vision/current/
├── model.rknn
└── model.yaml
```

`model.yaml` 中类别顺序必须与训练模型一致：

```yaml
schema_version: "1.0"
model_id: tube_pick_vision_v1
model_name: tube_pick_vision
model_version: v1
task_type: detection
target_platform: rk3576
input_size: [640, 640]
class_names: [tube_product, large_separator]
classes:
  - {id: 0, name: tube_product}
  - {id: 1, name: large_separator}
postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_det: 200
```

## 配置

全部配置位于：

```text
production/carton_line/config/line.yaml
```

重点字段：

```yaml
runtimes.pick
collectors.pick
pick.tcp
pick.algorithm
pick.debug
```

## 手动启动

```bash
./production/carton_line/scripts/start_runtime.sh pick
./production/carton_line/scripts/start_tcp_pick.sh
./production/carton_line/scripts/start_collector.sh pick
```

不连接机器人时，可以手动触发整条推理链路：

```bash
curl -s -X POST http://127.0.0.1:19130/api/app/evaluate_once \
  -H 'Content-Type: application/json' \
  -d '{"camera":"manual","task_id":"tube_pick_vision"}' \
  | python3 -m json.tool
```

## 调度端改动边界

服务响应保留原协议要求的触发关联字段，并保留空的 `types/poses`。但现有机器人 `VisionInterfacer` 不会自动消费自定义 `products/separators`，调度端仍需增加这两个字段的解析。不要把像素 `x/y` 或毫米深度 `z` 填入原 `types` 位姿字段。

## HTTP 调试结果与 TCP 线上帧的区别

`/api/app/evaluate_once` 是本机 HTTP 调试接口，响应头是
`application/json`，因此返回的是**纯 JSON 对象**，不会显示 TCP 帧界定符。
下面的输出是正常的：

```json
{"function":"tube_pick_result","products":[],"separators":[]}
```

真实机器人通信走 `tcp_client.py`，发送前会统一编码为：

```text
*{"function":"tube_pick_result","products":[],"separators":[]}#
```

代码发送路径为：

```python
sock.sendall(StarHashJsonCodec.encode(response))
```

如需在不连接机器人时直接查看与真实 TCP 完全相同的 `*...#` 文本帧，使用：

```bash
curl -s -X POST http://127.0.0.1:19130/api/tcp/evaluate_once_frame \
  -H 'Content-Type: application/json' \
  -d '{"function":"manual_test","camera":"cam_1","task_id":"tube_pick_vision"}'
```

该接口返回 `text/plain`，不能再接 `python3 -m json.tool`；输出首字符应为
`*`，末字符应为 `#`。

调试文件默认写入临时目录：

```text
/tmp/visionops_v3/carton_line/tube_pick_vision/latest
```
