# carton_partition_check Mock 契约

## 边界

该 App 消费标准 `inference_result`，输出 `AppDecision`、`gateway_message` 和 `200..219` 业务寄存器。M6 只是业务 Mock，不连接真实 PLC，不包含模板标定、图像几何或模型推理。

## 统一状态码

| Code | Label |
| ---: | --- |
| 0 | `OK` |
| 1 | `NG` |
| 2 | `NO_TARGET` |
| 3 | `MULTI_TARGET` |
| 4 | `LOW_CONFIDENCE` |
| 5 | `OUT_OF_ROI` |
| 6 | `SIZE_OUT_OF_RANGE` |
| 7 | `STRUCTURE_ABNORMAL` |
| 8 | `UPSTREAM_NO_RESULT` |
| 9 | `INTERNAL_ERROR` |

## 决策顺序

1. 有效 defect 目标最高优先级，输出 `STRUCTURE_ABNORMAL`。
2. 没有 cell 候选时输出 `NO_TARGET`。
3. cell 候选全部低于阈值时输出 `LOW_CONFIDENCE`。
4. `expected_cell_count` 已配置时优先校验精确数量，再校验 `min/max_cell_count`。
5. 其他情况输出 `OK`。

## HTTP

- `GET /health`
- `GET /api/app/status`
- `POST /api/app/evaluate_once`
- `GET /api/app/latest_decision`
- `GET /api/app/registers`
- `GET /api/app/register_map`

后续接入真实业务时只替换 upstream 为 Collector/Runtime 的真实 `latest_result`，保持 AppDecision 与寄存器契约稳定。
