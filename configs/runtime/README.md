# Runtime 配置目录

该目录只保存运行时配置的格式说明和无敏感信息示例。真实生成的 `.env` 与 `generated/` 内容由部署工具创建，不提交 Git。

`active_camera.example.json` 是多相机选择文件的模板。Collector Web“设置 → 相机设置”保存后，默认写入：

`configs/runtime/generated/active_camera.json`

视觉盒子设置默认写入：

`configs/runtime/generated/vision_box_settings.json`

也可以通过 `VISIONOPS_CAMERA_SELECTION_FILE` 指向 `/etc/visionops_v3/` 等设备本地路径。运行态文件不要提交到 Git。

配置分层与生成规则见 `docs/architecture/config_design.md`。
