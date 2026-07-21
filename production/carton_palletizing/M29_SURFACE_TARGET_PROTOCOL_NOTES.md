# M29 carton_palletizing：垛顶目标观测通信

M29 基于 M28.7 odd/even + trigger v2.1 修改。

## 核心变化

`pallet_place_target` 不再返回视觉规划的层号、slot 或下一摆放位置，而是返回实际检测目标：

- 无纸箱：1 个托盘；
- 有纸箱：最上层 1～4 个纸箱；
- 纸箱跨层时使用中心深度聚类，仅保留最近深度簇；
- 每个目标保留 OBB 中心、相机坐标和长轴角度；
- `held_box_pose` 逻辑完全不变。

## 保留内容

多层 slot 状态机、奇偶层模板和生产界面可视化仍保留，但不再参与机器人 `pallet_place_target` 返回值。

## M29.1 ROI 控制修复

- 机器人 `config.detect_region` 不再过滤托盘或纸箱；
- ROI 仅由 VisionOps Web / Runtime ROI 文件控制；
- `detect_region` 仍兼容接收，但只记录在诊断中；
- `pallet_place_target` 与 `held_box_pose` 均使用 Runtime 已输出的 ROI 内检测结果；
- 解决机器人下发 640×480 ROI 后，把 1280×720 画面中的有效托盘过滤为空的问题。
