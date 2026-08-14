# Foam Ring Grasp Production Architecture

## 1. 当前冻结链路

```text
Orbbec 336L RGB-D
        │
        ├── RGB shared memory ──> RKNN Runtime :28081
        │                              │
        └── Depth shared memory        │ segmentation
                                       ▼
                               foam_ring + ring_mouth
                                       │
                                       ▼
                              strict ring-mouth match
                                       │
                                       ▼
                               M38.1 front annulus
                                       │
                                       ▼
                         M39.3.1 ring-aware tilt evidence
                                       │
                                       ▼
                         M39.3.4 analytic conic surface
                              ┌────────┼────────┐
                              │        │        │
                            FLAT    TILTED  UNCERTAIN
                              │        │        │
                              │        │        └── reject
                              │        ▼
                              │   M39.3.4.1 tilted routing
                              │   real tilted PlaneModel
                              │        │
                              └────────┴──────┐
                                              ▼
                                     clock 3 candidate only
                                              │
                                     geometry/collision gates
                                              │
                                              ▼
                                    T_camera_visual_grasp
                                              │
                                      hand-eye transform
                                              │
                                      hand_tcp transform
                                              │
                                              ▼
                                  LEFT_LINK7 pregrasp/grasp
```

## 2. 生产硬约束

### 2.1 Visible-mouth only

M39.4 开始前，生产路由只接受存在可靠 `ring_mouth` 匹配的目标。旧 M38 branch B/D、M37 side-ring fallback 均不参与当前生产选择。

无 `ring_mouth` 的侧躺目标由 branch C 统一终止，避免旧算法猜测隐藏开口后输出机器人位姿。

### 2.2 Clock 3 only

`line.yaml` 中普通抓取和 tilted reroute 均固定 3 点钟：

```yaml
clock_search:
  preferred_clock_hours: [3]
  primary_clock_hours: [3]
  fallback_to_remaining: false

m39_3_4_1_tilted_production_routing:
  preferred_clock_hours: [3]
  fallback_enabled: false
  fallback_clock_hours: []
  maximum_clock_candidates: 1
```

M38.1 wrapper 的 `compare_top_clock_candidates` / `maximum_full_clock_candidates` 也固定为 1，防止内部再次扩展方向。

### 2.3 Tilted pose authenticity

TILTED 结果必须满足：

```text
production_surface_route = M39.3.4.1_TILTED
visual grasp +Z approach ≈ -analytic surface normal
approach-normal error <= configured threshold
```

不满足时不允许进入 robot pose transform。

### 2.4 Robot frame contract

- camera: `color_camera` / `camera_color_optical_frame`
- base: `robot_default_base`
- visual grasp: `m38_6_visual_grasp`
- TCP: `hand_tcp_link`
- flange: `left_link7`

机器人 SDK `move_l/get_pose` 使用默认 frame（不传 `frame="base"`）。

## 3. task 模块整理原则

只删除了确定不在在线服务依赖图中的离线入口、历史 replay 和阶段性验证程序。

仍保留的 `partial_opening_*` / `side_ring_template.py` / `side_surface_outer_contact_*` 是现有 `hybrid_grasp.py` 的内部依赖，也可作为 M39.4 设计参考；它们当前通过配置被禁用，不具有生产路由权。

历史 `side_ring_offline_validate.py` 已拆除，只提取在线 overlay 所需的小型 `side_ring_overlay.py`。

M39.3.2 ring-prior 在线诊断链已从 `online_processor.py` 删除；M39.3.3 独立在线诊断也已删除。M39.3.4 仍复用 `conic_ring_surface.py` 中的数学基础，因此该模块保留。
