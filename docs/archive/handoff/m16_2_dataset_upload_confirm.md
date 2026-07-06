# M16.2 数据集预览、确认上传与打包命名

本次在 M16.1 的采集保存/打包上传基础上继续完善采集上传流程。

## 路径

- 图片保存目录：`/opt/visionops_v3/data/images`
- 上传包目录：`/opt/visionops_v3/data/upload_packages`

`upload_packges` 为早期拼写错误，已统一修正为 `upload_packages`。

## 图片预览与删除

采集上传页的图片列表仍采用分页读取，避免图片数量较多时一次性加载导致页面卡顿。点击图片缩略图或“预览”按钮会打开放大预览弹窗；弹窗中可以直接删除当前图片。

## 上传确认

“上传服务器”按钮现在集成了打包采集包功能。点击后会弹出上传确认界面，需要填写：

- 设备 ID，必填，默认使用 Web 配置中的 Device ID。
- 客户 ID，必填，默认 `CUST-001`。
- 联系方式，选填。
- 备注，选填。

点击“确认上传”后，Collector 会先在本地生成 tar.gz，再读取 `vision_box_settings.json` 中的 upload 配置执行上传。无论上传成功或失败，都会弹出明显结果提示。

## 压缩包命名

上传包按以下规则命名：

```text
<device_id>_<customer_id>_<YYYYMMDD_HHMMSS>.tar.gz
```

示例：

```text
rk3576-001_bag-test_20260629_094629.tar.gz
```

如果同名文件已经存在，会追加 `_01`、`_02` 后缀，避免覆盖。

## manifest.json

压缩包内包含 `manifest.json` 和 `images/` 目录，manifest 格式：

```json
{
  "device_id": "rk3576-001",
  "customer_id": "bag-test",
  "contact_info": "",
  "remark": "",
  "created_at": "2026-06-29T09:46:29",
  "counts": {
    "all": 50
  },
  "package_name": "rk3576-001_bag-test_20260629_094629.tar.gz"
}
```

## 上传失败处理

上传失败时，本地压缩包仍保留在 `/opt/visionops_v3/data/upload_packages`，Web 弹窗会显示本地包路径和失败原因，便于后续重试或手动拷贝。
