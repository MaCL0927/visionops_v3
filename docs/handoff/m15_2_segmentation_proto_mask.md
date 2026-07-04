# M15.2 Segmentation Proto Mask Rasterization

本次在 `visionops_v3_m15_1_obb_input_size_fix` 基础上继续增强 C++ Runtime segmentation 后处理。

## 目标

此前 Rockchip YOLOv8-seg split-DFL 13 输出已经能够被识别，但 mask 只输出 bbox polygon，用于临时可视化。本次改为使用 segmentation 模型真实输出的 `mask coefficients` 与 `proto` 生成实例 mask，再将 mask 边界转为 polygon 输出给 Web overlay。

## 修改点

- `edge/runtime_cpp/src/postprocess_seg.cpp`
  - `SegItem` 增加输入坐标、mask coefficients 和 mask polygon。
  - split-DFL 解码时从 `[1, mask_dim, H, W]` mask coefficient head 读取每个候选的 mask coefficients。
  - fused segmentation 解码时从 detection tensor 尾部读取 mask coefficients。
  - 新增 `coeff × proto -> sigmoid -> binary mask -> boundary polygon` 流程。
  - mask 输出中增加 `mask.source` 字段：
    - `proto`：真实 proto mask polygon。
    - `bbox_fallback`：proto mask 为空时回退 bbox polygon。
  - polygon 做轻量 RDP 简化并限制最多约 160 点，避免 JSON 过大。

- `edge/runtime_cpp/src/model_config.cpp`
  - 顺手修复 `input_size:` 后接缩进 list 的解析，例如：
    ```yaml
    input_size:
      - 1280
      - 1280
    ```
  - 仍建议部署侧统一写成 `input_size: [1280, 1280]`，但 Runtime 现在也能识别缩进列表。

## 输出格式

输出仍使用 Web 已支持的 polygon mask 结构：

```json
"mask": {
  "encoding": "polygon",
  "source": "proto",
  "size": [720, 1280],
  "polygon": [[[x1, y1], [x2, y2], ...]]
}
```

## 限制

- 当前只输出最大外轮廓，不输出洞和多连通区域的多个 ring。
- 当前 mask 阈值固定为 0.5，后续可接入算法设置中的 mask 阈值。
- NMS 仍基于 bbox IoU。
