# M13 segmentation split-DFL 兼容说明

本次在 `edge/runtime_cpp/src/postprocess_seg.cpp` 中新增 Rockchip YOLOv8-seg split-DFL 多输出后处理。

## 支持的 RKNN 输出结构

当前新增支持以下结构：

```text
[1,64,H1,W1]        bbox DFL
[1,nc,H1,W1]        class scores
[1,1,H1,W1]         objectness / score
[1,mask_dim,H1,W1]  mask coefficients

[1,64,H2,W2]
[1,nc,H2,W2]
[1,1,H2,W2]
[1,mask_dim,H2,W2]

[1,64,H3,W3]
[1,nc,H3,W3]
[1,1,H3,W3]
[1,mask_dim,H3,W3]

[1,mask_dim,proto_h,proto_w] proto
```

典型 640 输入为：

```text
80x80, 40x40, 20x20 + proto 160x160
```

如果后续模型输入为 1280 或其他尺寸，只要仍保持上述 split-DFL 结构，后处理会根据每个 head 的 H/W 和模型输入尺寸动态推导 stride。

## 当前限制

- NMS 使用外接矩形 bbox IoU。
- 当前输出的 `mask` 是 bbox polygon 简化表示，用于 Web 可视化。
- 尚未实现 `mask_coeff @ proto` 的真实实例 mask 栅格化。
