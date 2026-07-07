# VisionOps v3 服务端工作流

## 1. 数据进入服务端

边缘端 Collector Web 打包得到 `*.tar.gz` 后，将文件复制或同步到服务端 `incoming_root`，默认是：

```text
server_data/incoming/
```

服务端 Web 第 1 步会扫描该目录，显示所有尚未处理的 tar.gz。用户勾选一个包时按单包处理；勾选多个包时合并为一个 batch。

处理后，服务端会：

```text
server_data/batches/<batch_id>/raw/
```

里面保存解压后的 `manifest.json`、`images/`、`labels/` 等内容。原始压缩包会移动到：

```text
server_data/incoming/processed/
```

## 2. 标注与审核

第 1 步不确认任务类型。第 2 步由标注人员根据 batch manifest 和数据内容判断任务类型，然后选择：

- `detection`
- `classification`
- `obb_detection`
- `segmentation`

当前 MVP 只有 accept / reject 状态，后续可接入 v2 标注器或 X-AnyLabeling。batch 详情中会显示源 manifest，便于判断数据来源、图片数量、设备和用户信息。

## 3. 数据集构建

从选中的 extracted/accepted batch 构建 dataset。构建时写入任务类型，并生成：

```text
server_data/datasets/<dataset_id>/dataset.json
server_data/datasets/<dataset_id>/batches.json
```

当前 MVP 只生成数据集清单，不复制大文件。真实训练接入时，可以根据 batch 的 `raw_path`、`images_path`、`labels_path` 生成 YOLO 或分类训练目录。

## 4. 训练任务

当前 training job 是 mock runner，用来验证：任务创建、状态流转、日志查看、模型包生成。真实训练接入时替换 `training/pipeline/stages` 下的占位实现。

## 5. 模型包发布

服务端完整保留训练指标和报告，发布到边缘端或 Syncthing 共享目录时只复制：

```text
model.rknn
model.yaml
```

边缘端 Collector Web 扫描后即可切换模型。

## 6. 设备分发

当前只维护设备注册表和 `target_model`。后续可对接边缘端 Collector API，实现远程触发扫描、切换和回传部署结果。
