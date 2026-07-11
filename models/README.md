# Local model packages

模型文件不进入 Git。边缘端模型包格式：

```text
models/<task>/<version>/
├── model.rknn
└── model.yaml
```

纸箱产线默认使用：

```text
models/carton_partition_check/current/
models/carton_tube_check/current/
```
