# M14 设置中心：SDK Bridge 相机设置调整

## 修改目标

在 `visionops_v3_m14_settings_ui.tar.gz` 基础上继续优化设置中心，重点调整“相机设置”页：

- 设置中心弹窗上下占比加长，更接近全屏设置面板。
- 相机取流方式统一命名为 SDK Bridge，不再在相机设置页使用 HP60C Bridge 作为通用名称。
- 固定的 SDK Bridge URL、Snapshot Path 不再作为用户可编辑字段展示。
- 预览刷新间隔与快照刷新间隔合并为一个“画面帧率 FPS”设置，保存后同步换算为 `preview_refresh_interval_ms` 与 `snapshot_refresh_interval_ms`。
- RGB 采集分辨率与帧率合并为 profile 下拉框，避免随意输入不受支持的组合。
- 新增 Depth / 深度图 profile 下拉框，并提供 RGB / Depth profile 匹配提示。
- 补充 HP60C / Orbbec 336L SDK Bridge 当前 env 中可配置的 JPEG、翻转、RGB 顺序、深度单位、Orbbec 序列号等入口。

## 参考的 Bridge 配置

HP60C SDK Bridge 当前支持：

- HTTP host / port
- JPEG quality
- MJPEG FPS
- vertical flip
- prefer MJPEG
- RGB order
- snapshot / depth / mjpeg / status 固定路径

Orbbec 336L SDK Bridge 当前支持：

- HTTP host / port
- color width / height
- depth width / height
- FPS
- JPEG quality
- MJPEG FPS
- vertical / horizontal flip
- depth unit
- serial
- snapshot / depth / depth_vis / depth_meta / mjpeg / status 固定路径

## 说明

本次仍然是前端临时设置，不写入 `/opt/visionops/edge/robot_gateway/*_bridge/*.env`，也不重启 systemd 服务。后续接入真实设置 API 后，可把这些字段映射到对应 env 并重启：

- `/opt/visionops_v3/edge/robot_gateway/hp60c_sdk_bridge/hp60c_sdk_bridge.env`
- `/opt/visionops_v3/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env`

