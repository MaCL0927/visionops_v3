# carton_bundle_grasp — M41

M41 is the first production version for grasping the **top bundle of tied cartons** from an oblique Orbbec 336L robot-eye view.

## Frozen M41 geometry contract

One bundle is treated as a rectangular top surface with known physical size:

- length: **715 mm**
- width: **525 mm**
- height: variable / not required by the visual algorithm
- camera height: variable / not compensated in the visual algorithm

The segmentation model supplies the visible top-surface mask. M41 then performs:

1. perspective quadrilateral extraction from the proto mask;
2. 96 distributed depth samples from an eroded mask interior;
3. RANSAC + SVD top-plane fitting in the color-camera coordinate frame;
4. SDK camera-ray / plane intersections for the four observed corners;
5. 715 x 525 mm fixed-size rectangular regularisation in the fitted 3-D plane;
6. output of the **3-D midpoints of the two 525 mm width edges**.

This fixes the perspective error of the old `box_grasp_vision` path where a 2-D edge midpoint was sampled first and only then deprojected.

## Camera pitch and robot waist Z

M41 never converts depth with a separate camera pitch angle. The two output points are always:

```text
position_camera = [Xc, Yc, Zc]  # mm, color camera frame
```

Camera tilt and camera-height changes belong to the camera-to-robot transform. The robot side should apply the normal hand-eye transform to each 3-D point.

Changing the robot waist height changes the observed camera-frame XYZ automatically because every frame is reconstructed from its own RGB-D observation. Therefore **reading waist Z is not required for M41 geometry**.

`algorithm.robot_state.sdk_read_enabled` is intentionally `false`. If later the App should log or use `waist_z_mm`, provide the robot Python SDK connection/login details and the exact API method; it can be attached without changing the M41 geometry contract.

## Required model before online production

The code is complete, but the uploaded M40 repository did not contain the new deployed segmentation package for this task. Before `runtime` can start, provide a package containing:

```text
model.rknn
model.yaml
```

at:

```text
/opt/visionops_v3/models/carton_bundle_grasp/current
```

or set:

```bash
export VISIONOPS_CARTON_BUNDLE_GRASP_MODEL_DIR=/path/to/model_package
```

Default target class is `class_id=0`; accepted names are configurable in `config/line.yaml`.

## Ports

- Runtime: `28089`
- App HTTP: `19215`
- Collector: `18098`
- Robot WebSocket: `9001/vision`
- Orbbec 336L bridge: existing `18182`

As with the other production tasks, WebSocket `9001` is intended for one active production profile at a time.

## Start manually

```bash
cd /opt/visionops_v3

production/carton_bundle_grasp/scripts/start_runtime.sh
production/carton_bundle_grasp/scripts/start_app.sh
production/carton_bundle_grasp/scripts/start_collector.sh
```

Health / latest decision:

```bash
curl -s http://127.0.0.1:19215/health | python3 -m json.tool
curl -s http://127.0.0.1:19215/api/app/latest_decision | python3 -m json.tool
```

## Performance architecture retained from the validated v3 path

- one inference producer thread;
- capacity-1 latest-only inference-result queue;
- explicit trigger packets are never overwritten by continuous frames;
- CPU geometry runs in the postprocess thread and overlaps with the next Runtime request;
- localhost Runtime uses raw socket HTTP, `TCP_NODELAY`, one `sendall`;
- Runtime JSON decoding is deferred to postprocess;
- Orbbec RGB/depth shared-memory fast paths remain enabled;
- no separate 5 Hz WebSocket throttle: every completed result is pushed;
- Runtime HTTP worker default for deployment remains `1`.

## Geometry gates

Default gates are deliberately production-safe but not excessively strict:

```yaml
top_plane:
  sample_count: 96
  ransac_threshold_mm: 5.0
  min_valid_samples: 36
  min_inlier_ratio: 0.70
  max_rms_mm: 6.0

bundle_prior:
  length_mm: 715.0
  width_mm: 525.0
  length_tolerance_mm: 80.0
  width_tolerance_mm: 70.0
```

The size tolerance is wider than the physical prior because mask labels can terminate slightly inside the physical cardboard boundary. Valid observations are still regularised to the exact 715 x 525 mm rectangle.

## Future M42/M43 path

M41 is intentionally `FULL_TOP_FIXED_SIZE`. Later partial-view versions can reuse the same output and communication contract:

- `PARTIAL_TOP_EDGE`: one reliable corner + one visible edge direction + plane + fixed L/W;
- `PARTIAL_TOP_PRIOR_YAW`: one corner + plane + known conveyor/carton yaw + fixed L/W.

One isolated corner with no direction/yaw information is geometrically insufficient to determine a unique rectangular top surface.
