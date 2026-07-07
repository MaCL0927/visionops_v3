# VisionOps v3 服务端 API

## 健康检查

```bash
curl http://127.0.0.1:18100/api/server/health | python3 -m json.tool
```

## 扫描 incoming 上传包

```bash
curl http://127.0.0.1:18100/api/server/incoming-packages | python3 -m json.tool
```

返回 `incoming_root` 和尚未处理的 `*.tar.gz` 列表。第一步不需要手动传 `device_id` 或 `task_type`。

## 处理 incoming 上传包

单包处理：

```bash
curl -X POST http://127.0.0.1:18100/api/server/batches/process-incoming \
  -H 'Content-Type: application/json' \
  -d '{"packages":["rk3576-001_package-test_20260707_085333.tar.gz"]}' \
  | python3 -m json.tool
```

多包合并处理：

```bash
curl -X POST http://127.0.0.1:18100/api/server/batches/process-incoming \
  -H 'Content-Type: application/json' \
  -d '{"packages":["A.tar.gz","B.tar.gz"]}' \
  | python3 -m json.tool
```

处理成功后生成：

```text
server_data/batches/<batch_id>/batch.json
server_data/batches/<batch_id>/raw/
```

原始包移动到：

```text
server_data/incoming/processed/
```

## 批次管理

```bash
curl http://127.0.0.1:18100/api/server/batches | python3 -m json.tool
curl http://127.0.0.1:18100/api/server/batches/<batch_id> | python3 -m json.tool
```

第二步确认任务类型并 accept：

```bash
curl -X POST http://127.0.0.1:18100/api/server/batches/<batch_id>/accept \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"detection"}'
```

reject：

```bash
curl -X POST http://127.0.0.1:18100/api/server/batches/<batch_id>/reject
```

## 构建数据集

指定 batch 构建：

```bash
curl -X POST http://127.0.0.1:18100/api/server/datasets/build \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"detection","batch_ids":["<batch_id>"]}' \
  | python3 -m json.tool
```

不指定 batch 时，会使用已 accept 且任务类型匹配的 batch。

## 创建训练任务

```bash
curl -X POST http://127.0.0.1:18100/api/server/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"<dataset_id>","task_type":"detection","epochs":50,"batch_size":16,"imgsz":640}' \
  | python3 -m json.tool
```

## 模型包

```bash
curl http://127.0.0.1:18100/api/server/model-packages | python3 -m json.tool
curl -X POST http://127.0.0.1:18100/api/server/model-packages/<model_id>/publish \
  -H 'Content-Type: application/json' \
  -d '{"publish_root":"/tmp/visionops_publish"}'
```

## 设备管理

```bash
curl -X POST http://127.0.0.1:18100/api/server/devices \
  -H 'Content-Type: application/json' \
  -d '{"device_id":"lb3576-dev","device_type":"lb3576","ip":"192.168.1.100"}'

curl -X POST http://127.0.0.1:18100/api/server/devices/lb3576-dev/assign-model \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"<model_id>"}'
```
