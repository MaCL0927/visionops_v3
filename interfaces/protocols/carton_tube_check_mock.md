# carton_tube_check Mock 契约

## 边界

该 App 消费标准 `inference_result`，输出 `AppDecision`、`gateway_message` 和 `100..119` 业务寄存器。M6 不连接真实设备，不在 Runtime 或 Collector 中实现业务决策。

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

按目标类别筛选后，依次判断无目标、低置信度、多目标、ROI/中心偏差和包围框尺寸。全部通过时输出 `OK`。

`offset_x_signed` 和 `offset_y_signed` 采用 int16 二进制补码放入 uint16 Holding Register：`encoded = signed & 0xFFFF`，读取值大于等于 `32768` 时用 `value - 65536` 解码。

## HTTP

- `GET /health`
- `GET /api/app/status`
- `POST /api/app/evaluate_once`
- `GET /api/app/latest_decision`
- `GET /api/app/registers`
- `GET /api/app/register_map`
