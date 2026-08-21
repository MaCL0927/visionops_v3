# carton_bundle_grasp — M41.2

Production task for grasping the **top tied bundle of cartons** from an oblique Orbbec Gemini 336L robot-eye view.

## Geometry contract

- segmentation class: `surface` (`class_id=0`)
- bundle physical size: **715 mm × 525 mm**
- bundle height: variable
- camera height: variable with robot waist Z
- visual output frame: Orbbec color-camera frame, millimetres
- robot output: two 3-D midpoints of the two **525 mm width edges**
- robot performs the normal camera-to-robot hand-eye transform
- visual geometry does **not** apply a separate camera-pitch or waist-Z correction

A successful target always has valid finite `position_camera=[X,Y,Z]` for both grasp points. M41.2 does not query depth at the final grasp pixels. If the top surface cannot support a reliable 3-D plane, the target is rejected instead of emitting `[0,0,0]`.

## M41.2 geometry path

```text
surface proto mask
      ↓
perspective quadrilateral
      ↓
96 distributed interior pixels
      ↓
Shared Depth: stable target-ROI snapshot
      ↓
vectorized 5×5 depth sampling
      ↓
vectorized pinhole XYZ for 96 points
      ↓
RANSAC + SVD top plane
      ↓
4 camera rays from shared color intrinsics
      ↓
ray / top-plane intersections
      ↓
observed 3-D rectangle
      ↓
715×525 mm fixed-size regularization
      ↓
2 width-edge 3-D midpoints
```

### Why the final grasp pixel does not need depth

Depth is used to reconstruct the **whole top plane**, not to look up the final grasp pixels. Once the plane, bundle center and 3-D axes are known, the two points are computed directly as:

```text
G_A = C - 715/2 × e_length
G_B = C + 715/2 × e_length
```

Therefore a local depth hole, strap or edge discontinuity exactly at `G_A/G_B` does not zero the camera coordinates. Enough valid interior depth samples are still required to fit the plane.

## M41.2 performance path

M41.1 failed to use shared depth efficiently because it kept a direct mmap view while processing all sample patches. At ~30 Hz the shared sequence changed during the ~59 ms Python sampling loop, so four retries were wasted before falling back to HTTP.

M41.2 changes this to:

1. read the shared-depth header;
2. copy only the target depth ROI while the sequence is stable;
3. re-check sequence immediately after the copy;
4. perform all depth filtering/percentile calculations on the private ROI snapshot;
5. process all 96 5×5 patches with vectorized NumPy operations;
6. compute the four corner rays directly from shared color intrinsics;
7. use **96 depth points total**; no 16 corner-ray depth probes and no second corner deprojection request.

Expected normal diagnostics:

```text
depth_point_count       = 96
corner_ray_probe_count  = 0
corner_ray_point_count  = 4
corner_ray_mode         = intrinsics
depth_transport         = posix_shared_memory
snapshot_attempts       ≈ 1
```

HTTP `sample_deproject` remains a fallback. If it is used, M41.2 still reads the shared-memory header for intrinsics so the corner rays do not need a second depth/deprojection request.

## Model

Default deployment path:

```text
/opt/visionops_v3/models/carton_bundle_grasp/current/
  model.rknn
  model.yaml
```

The package contains one class: `surface`.

## Ports

- Runtime: `28089`
- App HTTP: `19215`
- Collector: `18098`
- Robot WebSocket: `9001/vision`
- Orbbec 336L bridge: `18182`

Only one production profile should own WebSocket port `9001` at a time.

## Start

```bash
cd /opt/visionops_v3
production/carton_bundle_grasp/scripts/start_runtime.sh
production/carton_bundle_grasp/scripts/start_app.sh
production/carton_bundle_grasp/scripts/start_collector.sh
```

Or use the installed systemd services.

## Unified test tool

Small-version updates reuse one test script instead of adding version-specific scripts.

### Live performance

```bash
cd /opt/visionops_v3
python3 production/carton_bundle_grasp/scripts/test_m41.py performance
```

### App/shared-depth status

```bash
python3 production/carton_bundle_grasp/scripts/test_m41.py status
```

### Shared-memory ROI snapshot self-test

```bash
python3 production/carton_bundle_grasp/scripts/test_m41.py selftest
```

### Saved RGB-D dataset regression

```bash
python3 production/carton_bundle_grasp/scripts/test_m41.py offline /path/to/dataset
```

The dataset folder should contain `images/`, `depth/`, `labels/`, and `meta/`.

## Production architecture retained

- inference producer thread + postprocess/geometry thread;
- capacity-1 latest-only continuous-result queue;
- explicit robot `trigger/request_id` packets are never overwritten;
- raw localhost HTTP with `TCP_NODELAY` for Runtime/fallback paths;
- Runtime JSON decode stays in postprocess;
- no extra fixed 5 Hz WebSocket throttle;
- `debug.save_every_trigger: false` for continuous performance tests;
- Runtime HTTP worker default remains `1`.

## Main geometry gates

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

## Robot waist Z

The robot waist Z may later be read through the Python SDK for logging, transform selection, stationary-state checking or multi-frame depth fusion. It is intentionally **not required** by the M41.2 camera-frame XYZ reconstruction. The exact SDK connection/method can be integrated later without changing the visual geometry or robot protocol.

## Future partial-view mode

M41.2 freezes the `FULL_TOP_FIXED_SIZE` case. A later partial-view mode can reuse the same output protocol by combining a reliable 3-D corner, top-plane normal, one edge/yaw direction and the known 715×525 mm dimensions.
