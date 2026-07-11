# Tube Pick Vision 自定义 TCP JSON 协议

## 1. 传输

- 调度系统：TCP Server，默认监听 `10000`。
- VisionOps：TCP Client，主动连接并保持长连接。
- 编码：UTF-8 JSON。
- 帧格式：`*<json>#`。
- 支持分包和粘包。

## 2. 调度触发

沿用机器人侧触发结构，VisionOps 至少使用以下字段：

```json
{
  "function": "vision0",
  "timestamp": [1752135960, 123456789],
  "triggerpos": 1752135960,
  "triggerindex": 7,
  "camera": "cam_1",
  "task_id": "tube_pick_vision"
}
```

`timestamp`、`triggerpos`、`triggerindex`、`camera` 和 `task_id` 会回传，用于关联请求和响应。

## 3. 检测响应

```json
{
  "schema_version": "1.0",
  "message_type": "vision_detection_result",
  "function": "tube_pick_result",
  "timestamp": [1752135960, 123456789],
  "response_timestamp": [1752135960, 200000000],
  "triggerpos": 1752135960,
  "triggerindex": 7,
  "camera": "cam_1",
  "task_id": "tube_pick_vision",
  "result": 0,
  "status": "ok",
  "result_text": "success",
  "coordinate_frame": "image_depth_aligned",
  "coordinate_units": {
    "x": "pixel",
    "y": "pixel",
    "z": "mm"
  },
  "image": {
    "width": 1280,
    "height": 720
  },
  "depth": {
    "width": 1280,
    "height": 720,
    "encoding": "16UC1",
    "unit": "mm",
    "aligned_to": "color",
    "sampling": "roi_percentile",
    "roi_radius_px": 4,
    "percentile": 50
  },
  "product_detected": true,
  "separator_detected": true,
  "product_count": 2,
  "separator_count": 1,
  "invalid_depth_count": 0,
  "products": [
    {
      "class_id": 0,
      "class_name": "tube_product",
      "score": 0.946,
      "center": {
        "x": 532.4,
        "y": 281.7,
        "z": 842
      },
      "depth_valid": true
    }
  ],
  "separators": [
    {
      "class_id": 1,
      "class_name": "large_separator",
      "score": 0.923
    }
  ],
  "types": [],
  "poses": []
}
```

## 4. 坐标语义

- `products[].center.x`：产品框中心在彩色图像中的横坐标，像素。
- `products[].center.y`：产品框中心在彩色图像中的纵坐标，像素。
- `products[].center.z`：同一中心映射到 D2C 对齐深度图后，邻域有效深度的中位数，毫米。
- `separators[]`：只包含类别和置信度，不包含框、中心和深度。
- 不输出 `base_link`、机器人 TCP 或机械臂坐标。

## 5. result 约定

| result | status | 含义 |
|---:|---|---|
| `0` | `ok` | 推理完成，结果有效；允许没有检测到任何目标 |
| `2` | `partial` | 检测到产品，但至少一个产品中心没有有效深度 |
| `1001` | `error` | Runtime 或 Camera Bridge HTTP 请求失败 |
| `1002` | `error` | 深度图无效、过旧或解码失败 |
| `1003` | `error` | 请求、模型任务类型或结果格式错误 |
| `1004` | `error` | 服务正在处理上一条触发 |

## 6. 与原 VisionInterfacer 的兼容边界

响应保留 `types: []` 和 `poses: []`，因此原有调度端能识别为“视觉已响应，但没有机器人位姿”，不会把像素坐标误当成 `base_link` 坐标。

但是，原 `VisionInterfacer` 只消费 `types/poses`，不会自动使用新增的 `products/separators`。机器人调度程序必须增加对这两个自定义字段的解析，才能得到图像 `x/y` 和深度 `z`。

## 7. 调度端解析建议（nlohmann::json）

```cpp
if (json.contains("products") && json["products"].is_array()) {
    for (const auto &item : json["products"]) {
        const auto &center = item.at("center");
        const double image_x_px = center.value("x", 0.0);
        const double image_y_px = center.value("y", 0.0);
        const bool depth_valid = item.value("depth_valid", false);
        const int depth_z_mm = depth_valid && !center["z"].is_null()
            ? center["z"].get<int>() : 0;
        // image_x_px / image_y_px / depth_z_mm are not base_link coordinates.
    }
}

const bool separator_detected = json.value("separator_detected", false);
if (separator_detected && json.contains("separators")) {
    // Only class_id, class_name and score are supplied for separators.
}
```

## 8. TCP 帧与 HTTP 调试接口

协议中的 `*` 和 `#` 是 **TCP 字节流的帧界定符**，不是 JSON 对象内部字段。
因此完整线上消息是：

```text
*{"function":"tube_pick_result","triggerindex":7,"result":0,"products":[],"separators":[]}#
```

而本机 HTTP 调试接口：

```text
POST /api/app/evaluate_once
```

为了保持 `Content-Type: application/json`，只返回中间的 JSON 对象；通过
`curl ... | python3 -m json.tool` 看不到 `*` 和 `#` 属于预期行为，并不代表真实 TCP
发送缺少帧界定符。

用于检查实际编码格式的调试接口为：

```text
POST /api/tcp/evaluate_once_frame
```

它调用与真实 TCP 发送相同的 `StarHashJsonCodec.encode()`，以 `text/plain` 返回
完整的 `*<JSON>#` 帧。
