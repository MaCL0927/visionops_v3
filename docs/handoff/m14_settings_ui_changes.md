# M14 设置界面优化

本次修改在 v3 前端设置弹窗中参考 v2 的现场设置界面风格，将原来简单的 Runtime/Gateway/刷新间隔表单，重构为三个设置页：

1. 相机设置
2. 视觉盒子设置
3. 算法设置

当前阶段只完成界面与前端 localStorage 临时配置，不写入 `.env`、`model.yaml` 或 systemd 服务配置。后续可逐步接入真实设置 API。

## 主要功能

- 大字号、大按钮、大卡片，适配工厂现场触屏使用。
- 相机设置页包含 HP60C Bridge URL、snapshot path、预览刷新间隔、图像参数和标定预留入口。
- 视觉盒子设置页包含 Runtime/Gateway/Business App URL、Device ID、状态刷新间隔、模型目录、数据目录、日志目录和端口预留入口。
- 算法设置页包含自动推理间隔、预处理后端偏好、任务类型偏好和可视化显示开关。

## 已接入前端即时生效的设置

算法可视化开关已在模型验证页 overlay 中生效：

- 显示标签和置信度
- 显示中心点
- Detection 显示水平框
- OBB 显示旋转框
- OBB 显示外接水平框
- Segmentation 显示检测框
- Segmentation 显示 mask polygon
- Mask 透明度

其中 OBB 外接水平框默认关闭，避免与旋转框重复显示；需要时可在算法设置中打开。

## 注意事项

当前 segmentation 后处理输出的 mask 仍是 bbox polygon 简化表示，不是真实 `mask_coeff @ proto` 栅格化结果。Web 端已按 mask polygon 统一绘制，后续 Runtime 输出真实 polygon/RLE 后，可复用同一开关。

## M14.1 SDK Bridge 相机设置页调整

- 设置中心高度从 `94vh` 调整为接近全屏的 `calc(100vh - 16px)`。
- 相机设置页从 HP60C 专用表述调整为 SDK Bridge 通用表述。
- 移除可编辑的 Bridge URL 与 Snapshot Path 字段，固定路径由 systemd/env 和 bridge 程序管理。
- 使用“画面帧率 FPS”统一控制预览刷新和快照刷新。
- 使用 RGB profile 与 Depth profile 下拉框表示分辨率 + FPS 组合。
- 新增 RGB / Depth profile 匹配提示。
- 增加 JPEG 质量、RGB 数据优先级、翻转、RGB 顺序、深度单位、Orbbec 序列号等字段占位。
