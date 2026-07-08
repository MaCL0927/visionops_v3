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
- `obb`
- `segmentation`

当前已接入内置标注器。标注器会保存 `labels/`、`labels_auto/`、`annotation_classes.json` 和 `annotation_task.json`，点击“确认审核完成”后会把 batch 标记为 accepted。

## 3. 数据集构建

从选中的 extracted/accepted batch 构建 dataset。构建时写入任务类型，并生成：

```text
server_data/datasets/<device_id>_<customer_id>_<task>_<yyyymmdd_hhmmss>/dataset.json
server_data/datasets/<device_id>_<customer_id>_<task>_<yyyymmdd_hhmmss>/batches.json
```

当前会从所有已审核且任务类型匹配的 batch 构建 dataset，并物化 YOLO 训练目录。未手动传入 batch_ids 时，扫描规则是：选择 `status=accepted` 且 `task_type` 等于当前选择任务类型的全部 batch。

## 4. 训练任务

当前 training job 已接入真实 stage 化 pipeline：`preprocess -> train -> evaluate -> export_onnx -> convert_rknn -> package_v3_model`。其中 train 使用当前服务端环境；export_onnx 默认进入 `pt2onnx`；convert_rknn 默认进入 `rknn311`。

## 5. 模型包发布

服务端完整保留训练指标和报告，发布到边缘端或 Syncthing 共享目录时只复制：

```text
model.rknn
model.yaml
```

边缘端 Collector Web 扫描后即可切换模型。

## 6. 设备分发

当前只维护设备注册表和 `target_model`。后续可对接边缘端 Collector API，实现远程触发扫描、切换和回传部署结果。

### 训练 pipeline 操作顺序

1. 在第二步标注器中完成标注，点击“确认审核完成”。
2. 回到服务端控制台第三步，选择任务类型，点击“从已审核数据构建数据集”。
3. 在 dataset 下拉框中选择刚生成的数据集。
4. 根据任务类型确认预训练权重，例如：
   - detection: `models/pretrained/yolov8n.pt`
   - obb: `models/pretrained/yolov8n-obb.pt`
   - segmentation: `models/pretrained/yolov8n-seg.pt`
5. 点击“开始训练流水线”。
6. 在训练任务列表点击“日志”查看当前阶段。
7. 成功后模型包会出现在第四步“模型部署”的模型包列表中。
