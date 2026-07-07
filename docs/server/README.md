# VisionOps v3 服务端说明

服务端的定位是“数据中心 + 训练中心 + 模型包中心 + 设备分发中心”。它不替代边缘端 Collector Web，也不执行边缘端实时推理。

## 当前 MVP 已支持

- 服务健康检查：`GET /api/server/health`
- incoming 上传包扫描：`GET /api/server/incoming-packages`
- 处理 incoming 目录下的 `*.tar.gz`：`POST /api/server/batches/process-incoming`
- 批次列表、详情、accept、reject
- 第二步确认任务类型，并从 extracted/accepted batch 构建数据集清单
- 创建训练任务 mock runner
- 自动生成 v3 标准模型包
- 模型包列表、详情、发布到同步目录
- 设备注册表与目标模型分配
- 简单 Web 控制台：`http://<host>:18100/`

## Web 控制台与 v2 功能映射

当前 v3 服务端 Web 参考 v2 控制台的现场流程组织为 4 个步骤：

```text
1. 接收并处理上传包  -> 扫描 incoming_root/*.tar.gz，解压为 v3 batch
2. 标注与审核        -> 查看 batch manifest，确认任务类型，accept/reject，构建 dataset
3. 训练与模型状态    -> v3 training job + model package
4. 模型部署          -> v3 model publish + device registry / target_model
```

底层仍保留 5 类服务端对象：

- `batch`：一次上传包解压后的数据批次；第一步不确认任务类型，默认 `task_type=unassigned`。
- `dataset`：由一个或多个 batch 组成的数据集版本；第二步才确认 detection/classification/OBB/segmentation。
- `training job`：训练流水线任务；MVP 阶段为 mock runner。
- `model package`：服务端生成并管理的 v3 标准模型包。
- `device`：设备注册表和目标模型分配记录。

因此，UI 操作流程可以参考 v2，但后端实现必须使用 v3 的 batch/dataset/job/model-package/device 分层，不能恢复 v2 的旧目录强绑定和边缘端 Python 推理链路。

## incoming 上传包目录

默认目录：

```text
server_data/incoming/
```

也可以启动时指定：

```bash
python3 -m apps.server_api.backend.main \
  --host 0.0.0.0 \
  --port 18100 \
  --data-root /opt/visionops_v3/server_data \
  --incoming-root /opt/visionops_v3/server_data/incoming \
  --publish-root /opt/visionops_v3/server_data/published_models
```

边缘端 Web 打包后的 `tar.gz` 文件先复制或通过 Syncthing 同步到 `incoming_root`。服务端第一步只扫描该目录，不要求人工填写 `device_id` 或 `task_type`。

包名推荐格式：

```text
rk3576-001_package-test_20260707_085333.tar.gz
```

服务端会自动推导：

```text
batch_id    = rk3576-001_package-test_20260707_085333
device_id   = rk3576-001
customer_id = package-test
captured_at = 20260707_085333
```

处理成功后：

```text
server_data/batches/<batch_id>/raw/     # 解压后的 manifest、images、labels 等
server_data/incoming/processed/         # 已处理的原始 tar.gz
```

## 当前尚未接入

- 真实 YOLO/分类/OBB/分割训练
- MLflow 真实 run 记录
- ONNX 导出
- RKNN 转换
- 远程调用边缘端 Collector API 完成模型热切换
- 数据标注审核器

这些能力在目录和 API 上已经预留，但第一版不假装生产可用。
