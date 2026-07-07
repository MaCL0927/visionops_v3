# VisionOps v3 标准模型包规范

服务端生成的边缘端模型包必须适配 v3 Collector Web 的扫描逻辑。

## 边缘端同步内容

```text
models/<model_name>/
├── model.rknn
└── model.yaml
```

边缘端只依赖这两个文件。训练日志、指标、导出报告等保留在服务端。

## 服务端完整模型包

```text
server_data/model_packages/<model_id>/
├── model.rknn
├── model.yaml
├── package.json
├── metrics.json
├── train_config.yaml.json
├── export_report.json
└── logs/
```

## model.yaml 推荐字段

```yaml
schema_version: '1.0'
model_id: carton_tube_det_20260706
model_name: carton_tube_det
model_version: 20260706_001
task_type: detection
target_platform: rk3576
input_size: [640, 640]
model:
  name: carton_tube_det
  version: 20260706_001
  task: detection
  format: rknn
  target_platform: rk3576
  input_size: [640, 640]
classes:
  - id: 0
    name: tube
class_names:
  - tube
postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_det: 100
runtime:
  preprocess: letterbox
  color: rgb
```
