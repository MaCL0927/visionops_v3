# Production solutions

`production/` 只保存实际产线方案，不保存平台通用能力。

组织规则：

```text
production/<line_id>/
├── config/        # 该产线唯一主配置
├── gateway/       # 该产线的触发、业务编排和通信协议
├── tasks/         # 按任务拆分的算法与静态标定资产
├── scripts/       # 该产线启动入口
├── deploy/        # systemd 与安装脚本
└── README.md
```

新增任务时，优先放入现有产线的 `tasks/<task_id>/`；只有 PLC 协议、相机组合或部署拓扑独立时，才新建另一条产线目录。

平台通用的 Runtime、Collector、Camera Bridge、Modbus Server 和接口定义仍放在 `edge/`、`apps/` 和 `interfaces/`，禁止复制到产线目录。

当前独立生产方案：

- `carton_line/`：纸隔板、纸筒检测和纸筒抓取；
- `carton_palletizing/`：纸箱码垛与箱体抓取；
- `detergent_grasp/`：俯拍 OBB 洗衣液抓取和目标纸箱定位（M30.1）。
