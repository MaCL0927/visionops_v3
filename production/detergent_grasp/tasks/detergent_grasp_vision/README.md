# detergent_grasp_vision

俯拍场景下使用 OBB 模型检测大瓶、小瓶、瓶口/抓取点 `head` 和目标纸箱。算法把 `head` 与对应瓶体关联后，通过统一 WebSocket JSON 协议输出抓取像素、瓶体角度和纸箱位置。

核心文件：

- `algorithm.py`：OBB 解析、长轴角度和瓶体/抓取点一对一匹配；
- `service.py`：Runtime 调用、HTTP、WebSocket、故障码和生产 FPS；
- `PROTOCOL.md`：机器人端字段契约；
- `mock_robot_client.py`：无第三方依赖的联调客户端。
