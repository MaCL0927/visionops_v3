# M15 模型包读取简化

本次将边缘端模型包标准固定为一个模型一个目录：

```text
models/<model_dir>/
├── model.rknn
└── model.yaml
```

## 设计决定

- Runtime 启动只需要 `--model-dir <模型目录>`。
- `model.yaml` 是唯一元信息来源。
- 不再读取 `manifest.json`。
- 不再读取 `labels.txt`。
- 不再兼容平铺 `*.rknn + *.yaml`。
- Web 模型列表只扫描 `models_root` 下一级目录，并且只以 `model.yaml` 展示模型信息。

## model.yaml 常用字段

```yaml
schema_version: 1
model_id: product1-rk3576-0.1.0
model_name: product1
model_version: 0.1.0
task: detection
# 或 task_type: detection
target_platform: rk3576
input_size: [640, 640]
class_names:
  - product
  - point
conf_threshold: 0.5
nms_threshold: 0.45
```

其中 `task` / `task_type`、`conf_threshold` / `score_threshold` 均可识别。

## Runtime 启动示例

```bash
MODEL_DIR=/opt/visionops_v3/models/product1

./build-rknn/edge/runtime_cpp/visionops_runtime_mock \
  --backend rknn \
  --preprocess-backend rga \
  --frame-source hp60c_bridge \
  --hp60c-url http://127.0.0.1:18182 \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 28081 \
  --device-id lb3576-dev
```

## 影响文件

- `edge/runtime_cpp/src/model_package.cpp`
- `edge/runtime_cpp/src/model_config.cpp`
- `edge/runtime_cpp/src/runtime_app.cpp`
- `apps/collector_web/backend/model_catalog.py`
- 相关测试和文档
