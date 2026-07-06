# M15.1 OBB input_size mismatch fix

## 背景

M15 将模型包简化为 `models/<model>/model.rknn + model.yaml` 后，Runtime 只从 `model.yaml` 读取模型元信息。现场发现某些旧 OBB 模型的 `model.yaml` 中 `input_size` 仍为 `640x640`，但实际 `model.rknn` 的输入 tensor 是 `1280x1280`，导致 Runtime 按 640 预处理后调用 `rknn_inputs_set` 返回 `-5`。

## 修复

- `RknnRunner` 增加 `input_infos()` / `output_infos()` 查询接口。
- RKNN real runner 在 `load_model()` 后保留 RKNN 查询得到的输入 tensor 维度。
- Runtime 准备模型后，从 RKNN 输入 tensor 动态推导真实输入尺寸。
- 如果 `model.yaml` 的 `input_size` 与 RKNN 真实输入尺寸不一致，推理时优先使用 RKNN 真实输入尺寸，避免 `rknn_inputs_set -5`。
- 不改变模型包标准；`model.yaml` 仍然是模型元信息来源，但 RKNN 输入 tensor 尺寸是实际推理的硬约束。

## 注意

建议后续仍然修正对应模型包的 `model.yaml`：

```yaml
input_size: [1280, 1280]
```

这样 Web 模型列表和 Runtime 实际输入尺寸可以保持一致。
