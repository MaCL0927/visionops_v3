# carton_tube_check

负责纸筒站立/倒伏分类、5×8 槽位分配、RGB 到 Depth 坐标缩放、深度采样和高度异常判断。

- 算法：`algorithm.py`
- 参数：`../../config/line.yaml` 中的 `tube.algorithm`
- 深度来源：`../../config/line.yaml` 中的 `camera_bridge.depth_url`
