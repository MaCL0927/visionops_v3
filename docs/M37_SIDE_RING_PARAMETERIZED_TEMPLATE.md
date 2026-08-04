# M37：侧躺圆环参数化 3D 模板拟合离线验证

## 目标

M37 是独立于 M36 开口可见分支的新一级流程。它处理只有 `foam_ring`、没有可靠 `ring_mouth` 的侧躺圆环。

圆环使用已知尺寸的短空心圆柱模型：

- 外半径 42.5 mm；
- 内半径 30 mm；
- 轴向长度 70 mm。

M37 第一阶段只验证：

1. 圆柱轴线；
2. 圆环中心和两个端面中心；
3. 轴线正负方向；
4. 距离深度相机更近一侧的圆柱弧面抓取点。

不启用箱壁碰撞、机器人坐标转换或正式抓取通信。

## 轴线方向规则

外圆柱侧面只能确定无向轴线。M37 将两个模板端面中心分别计算到相机原点的距离，选择距离较小的一端作为近端开口，并将轴线定义为：

`far_opening_center -> near_opening_center`

因此 `axis_toward_camera` 始终朝向更靠近深度相机的一侧。

## 抓取点（M37.1 已修正）

M37 初版曾在近端开口平面内建立壁厚中线圆，并取投影纵坐标最小的圆周点。该定义对应开口边缘最高点，不符合实际侧躺抓取位置。

M37.1 改为：从近端开口沿轴线向内部缩进 11 mm，使用该局部轴向带内的真实圆柱径向内点恢复相机可见角区间，再在名义外表面半径 42.5 mm 上计算抓取点。

输出同时保留：

- `near_side_crown`：M37.1 可见外圆柱弧面抓取点；
- `near_opening_rim_top_diagnostic`：M37 旧开口边缘点，仅用于对照；
- `top_arc.grasp_point_*`：兼容别名，指向 M37.1 新点；
- 对应的 `uv` 投影。

详细定义见 `docs/M37.1_NEAR_VISIBLE_CROWN_GRASP_POINT.md`。

## M36.5 调试包验证

```bash
cd /opt/visionops_v3

bash production/foam_ring_grasp/scripts/run_m37_side_ring_offline.sh \
  /opt/visionops_v3/data/foam_ring_online_geometry/1785833810687
```

默认只处理没有匹配 `ring_mouth` 的 `foam_ring`。需要同时观察已匹配目标时：

```bash
bash production/foam_ring_grasp/scripts/run_m37_side_ring_offline.sh \
  /path/to/bundle \
  --include-mouth-matched
```

限制单个实例：

```bash
bash production/foam_ring_grasp/scripts/run_m37_side_ring_offline.sh \
  /path/to/bundle \
  --instance-id 7
```

## 旧 raw 数据集验证

PT 模型方式：

```bash
python3 -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate \
  --data-root /home/pc/桌面/visionops_v3/server_data/batches/rk3576-001_ring_20260728_132732/raw \
  --all \
  --model /path/to/best.pt \
  --output /home/pc/桌面/m37_side_ring_results
```

人工标签方式：

```bash
python3 -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate \
  --data-root /path/to/raw \
  --all \
  --labels-dir /path/to/labels \
  --output /path/to/output
```

## 输出

每个 capture 目录包含：

- `side_ring_template_result.json`；
- `side_ring_template_summary.csv`；
- `side_ring_template_overlay.jpg`；
- 每个实例的可见点云 PLY；
- 每个实例的近/远端模板曲线 PLY。

叠加图含义：

- 洋红线：拟合轴线；
- 红色实点和箭头：近端及朝相机方向；
- 红色方框：M37.1 近端可见弧面抓取点；
- 白色斜十字：旧开口边缘投影最高点；
- 蓝色点：可见弧面角区间两端；
- 黄色/青色曲线：近端外圆和内圆模板；
- 橙色曲线：向内部缩进后的抓取截面；
- 青色实例轮廓：当前自动选中的侧躺目标。

## 质量门限

`eligible` 由以下条件共同决定：

- 圆柱径向内点率；
- 径向残差中位数和 P90；
- 可见轴向跨度；
- 轴线与相机视线夹角；
- 是否已有可靠 `ring_mouth` 配对。

M37 离线阶段应优先检查叠加图和点云，不应只依赖一个总分。
