# 服务端数据集存储策略

## 目标

服务端仍保留三个逻辑层级，但不再保存三份图片数据：

```text
batches/   标注与审核视图，保留原始图片和最终 labels
datasets/  固定训练版本，提供 Ultralytics train/val 目录
jobs/      训练日志、权重、中间模型和报告，不保存数据集副本
```

## 新数据的实际存储方式

标注审核完成后，`DatasetService` 会生成 `datasets/<dataset_id>`：

- `images/train`、`images/val` 优先使用硬链接指向 `batches` 中的原始图片；
- `labels/train`、`labels/val` 独立复制，因为标签文件很小，并且需要保持数据集版本不可变；
- classification 的类别目录图片同样优先使用硬链接；
- 如果源文件与 datasets 不在同一文件系统，自动回退为普通复制，不影响原有功能。

硬链接不是快捷方式。删除 batch 时，dataset 中的图片仍可正常使用；删除 dataset 时，batch 中的标注数据也不受影响。两条路径只是共享相同的数据块。

`dataset.json` 中会记录：

```json
{
  "storage_mode": "hardlink_images_copy_labels",
  "storage": {
    "hardlinked_images": 1000,
    "copied_images": 0,
    "estimated_saved_bytes": 2147483648
  }
}
```

## 训练任务

训练任务的 preprocess 阶段直接使用：

```text
server_data/datasets/<dataset_id>/yolo_dataset
server_data/datasets/<dataset_id>/cls_dataset
```

不再创建：

```text
server_data/jobs/<job_id>/work/yolo_dataset
server_data/jobs/<job_id>/work/cls_dataset
```

`jobs` 只保留训练产生的 `runs/`、权重、ONNX、RKNN、日志和报告。数据集正在被 pending/running 任务使用时，服务端会阻止删除该 dataset。

## 迁移已有数据

先停止新的训练任务，再执行 dry-run：

```bash
cd /opt/visionops_v3
python3 -m tools.storage.optimize_server_data \
  --data-root /home/pc/桌面/visionops_v3/server_data
```

确认输出后应用：

```bash
python3 -m tools.storage.optimize_server_data \
  --data-root /home/pc/桌面/visionops_v3/server_data \
  --apply
```

工具会：

1. 把已有 datasets 图片副本替换成指向 batches 原图的硬链接；
2. 跳过无法可靠匹配的文件；
3. 删除非运行状态 job 下的 `work/yolo_dataset` 或 `work/cls_dataset`；
4. 更新已有 `preprocess_report.json` 的数据集路径；
5. 跳过 pending/running 训练任务。

只处理 datasets，不处理旧 job 副本：

```bash
python3 -m tools.storage.optimize_server_data \
  --data-root /home/pc/桌面/visionops_v3/server_data \
  --apply \
  --skip-jobs
```

## 查看真实空间收益

分别执行 `du -sh batches` 和 `du -sh datasets` 时，硬链接可能在两次独立统计中都显示文件大小；这不表示磁盘真的存了两份。应查看整个 `server_data` 或文件系统剩余空间：

```bash
du -sh /home/pc/桌面/visionops_v3/server_data
df -h /home/pc/桌面/visionops_v3/server_data
```

也可以检查两条路径是否共享 inode：

```bash
ls -li batch_image.jpg dataset_image.jpg
```

inode 编号相同且链接数大于 1，说明图片数据只占一份磁盘块。
