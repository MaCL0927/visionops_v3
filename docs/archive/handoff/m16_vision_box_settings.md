# M16 视觉盒子设置

本次在 M15.2 基础上接入视觉盒子设置页面与后端设置 API。

## 配置文件位置

视觉盒子设置持久化到：

```text
/opt/visionops_v3/config/vision_box_settings.json
```

可用环境变量覆盖：

```text
VISIONOPS_VISION_BOX_SETTINGS_FILE=/path/to/vision_box_settings.json
```

该文件保存 Web 可调整但不属于具体模型或相机 SDK 的边缘端配置，例如默认启动模式、状态刷新频率、磁盘告警阈值和服务端上传参数。

## 新增 API

```text
GET  /api/settings/vision_box
POST /api/settings/vision_box
```

可编辑字段：

- default_mode: factory / production
- status_refresh_fps
- disk_warning_percent
- upload.server_ip
- upload.ssh_user
- upload.ssh_password
- upload.ssh_port
- upload.remote_dir
- upload.timeout_s

只读展示字段：

- Runtime URL
- Gateway URL
- Business App URL
- Device ID
- Runtime / Collector 端口
- models/data/log 目录

## 页面行为

- 状态刷新间隔由 ms 改为刷新 FPS。
- 默认启动模式会在 Web 加载后生效；选择 production 会默认进入生产模式。
- 磁盘告警阈值会保存并用于视觉盒子设置 API 返回的 storage warning 判断。
- 服务端上传配置只保存，后续采集上传确认页再读取使用。
- 双网口配置区域目前是预留展示，不写入系统网络配置。
