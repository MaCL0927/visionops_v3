# M13 OBB dynamic input compatibility

本次修改目标：在 RGA-only 代码基础上修复 OBB RKNN 后处理对 1280×1280 输入模型不兼容的问题。

## 问题现象

1280×1280 OBB 模型输出：

```text
output[0] dims=[1,67,160,160]
output[1] dims=[1,67,80,80]
output[2] dims=[1,67,40,40]
output[3] dims=[1,1,33600]
```

模型配置 `labels_count=2`，原 OBB 后处理要求 head 通道数必须严格等于 `64 + labels_count`，即 66，因此 `[1,67,H,W]` 被判定为 `UNSUPPORTED_OUTPUT_SHAPE`。

## 修改内容

- Rockchip YOLOv8-OBB split-DFL head 检测逻辑由严格 `64 + nc` 改为 `>= 64 + configured_class_count`。
- 兼容部分 RKNN 导出模型保留一个额外辅助通道的情况，例如 `64 + nc + 1`。
- angle tensor 不再固定假设 640 输入的 8400，而是动态根据三个 head 的空间尺寸求和：
  - 640 输入：80×80 + 40×40 + 20×20 = 8400
  - 1280 输入：160×160 + 80×80 + 40×40 = 33600
  - 其他输入尺寸只要 head/angle 空间数量匹配，也可以识别。
- 解码时只使用模型配置中的类别数对应的 class channels，多出的辅助通道默认忽略，避免把辅助通道误当成类别。

## 修改文件

- `edge/runtime_cpp/src/postprocess_obb.cpp`
- `edge/runtime_cpp/tests/postprocess_fixture.cpp`

## 验证

新增 fixture：

```bash
./visionops_postprocess_fixture obb_rockchip_extra_channel
```

用于验证 `[1,67,*,*] + [1,1,N]` 形式的 OBB split 输出不再被判为 unsupported。

## 说明

该修改不影响 detection 后处理，不修改 RGA 预处理，不引入 RKNN input/output buffer 深度复用，也不引入 HP60C raw 原始帧入口。
