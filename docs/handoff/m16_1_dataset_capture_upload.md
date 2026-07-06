# M16.1 数据集保存、打包与上传

本次在 M16 视觉盒子设置基础上接入采集上传闭环：

- 图片保存目录：`/opt/visionops_v3/data/images`
- 上传包目录：`/opt/visionops_v3/data/upload_packages`
- 上传配置来源：`/opt/visionops_v3/config/vision_box_settings.json` 中的 `upload` 字段

## 后端 API

- `GET /api/dataset/images?offset=0&limit=24`：分页读取采集图片，避免一次性加载大量图片导致 Web 卡顿。
- `GET /api/dataset/images/<filename>/content`：读取单张采集图片。
- `POST /api/dataset/images/capture`：从 Runtime snapshot 保存当前图片到边缘端。
- `DELETE /api/dataset/images/<filename>`：删除单张采集图片。
- `POST /api/dataset/packages/create`：将本地图片打包为 tar.gz。
- `GET /api/dataset/packages`：读取本地历史打包文件。
- `POST /api/dataset/upload`：先创建 tar.gz，再按视觉盒子设置里的 SSH 配置上传到服务端。

上传失败时接口会保留本地 tar.gz，并返回本地路径，便于后续重试或手动拷贝。

## 上传实现

优先使用 `paramiko`。如果当前 Python 环境没有 paramiko，则回退到系统 `ssh/scp`；使用 SSH 密码时需要系统安装 `sshpass`。

建议现场环境二选一：

```bash
pip install paramiko
```

或：

```bash
sudo apt-get install -y sshpass openssh-client
```

## Web 端

采集上传页已改为服务端持久化模式：

- 拍照采集：保存到 `/opt/visionops_v3/data/images`
- 采集记录：分页显示，每页 24 张，图片 lazy loading
- 支持预览、单张删除、当前页批量删除
- 支持打包采集包
- 支持打包后上传服务端

M16.1 不做缩略图生成，主要通过分页和懒加载避免 v2 中图片过多导致页面卡顿的问题。后续如图片量非常大，可增加 thumbnail 缓存目录。
