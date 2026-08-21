# carton_bundle_grasp development history

This is the single rolling development-history file for the task. Small versions should update this file instead of adding new version-specific Markdown files.

## M41 — initial 3-D top-plane version

- created `production/carton_bundle_grasp/` from the validated v3 production architecture;
- one-class `surface` segmentation;
- 96 distributed top-surface depth samples;
- robust 3-D plane fitting;
- 715×525 mm fixed-size rectangle prior;
- output two 3-D midpoints of the 525 mm width edges;
- camera-frame XYZ only; no separate camera-pitch/waist-height correction;
- inherited latest-only continuous pipeline and reliable explicit trigger/request_id handling.

Initial five-dataset regression showed approximately 692–698 mm × 506–515 mm observed top surfaces before fixed-size regularization, with plane RMS roughly 1–2 mm.

## M41.1 — first performance refinement

- replaced full-frame mask allocation/erosion/`np.where` with compact contour-grid interior sampling;
- vectorized RANSAC hypothesis scoring;
- removed the second explicit four-corner `/deproject` request;
- temporarily used one combined request containing 96 plane samples + 16 corner-ray depth probes;
- added fine-grained timing.

On RK3576 the mask/classification preparation dropped from about 95 ms to about 5–10 ms, but the total depth stage remained about 200–280 ms. Timing showed `depth_transport=raw_socket` despite shared depth being enabled.

Root cause was then identified: the M41.1 shared-memory reader processed all 112 points directly against the live mmap and only checked the shared `sequence` afterwards. One attempt took roughly 59 ms while the 336L publishes a new depth frame about every 33 ms. Sequence therefore changed on nearly every attempt; four failed shared-memory retries were followed by the fast HTTP fallback.

This was a shared-memory consistency/read-architecture issue, **not** the 336L's millimetre depth noise.

## M41.2 — ROI snapshot + vectorized depth + intrinsics rays

Frozen geometry and robot protocol remain unchanged.

Changes:

1. **Stable ROI snapshot**
   - derive one depth ROI covering all 96 plane sample patches;
   - copy the ROI from the active shared-depth buffer;
   - verify sequence/active-buffer immediately after the copy;
   - all slower NumPy work happens on the private snapshot, so later camera frames cannot invalidate it.

2. **Vectorized 96-point depth sampling**
   - one compact `N×K` matrix (normally `96×25` for 5×5 patches);
   - vectorized valid-range mask and valid-pixel counts;
   - row-wise sorted percentile interpolation matching the Bridge rule;
   - vectorized pinhole XYZ calculation.

3. **96 depth points only**
   - removed all 16 M41.1 corner-ray depth probes;
   - `depth_point_count=96` for one target.

4. **Intrinsics corner rays**
   - use published shared color intrinsics `fx/fy/cx/cy` and image/depth dimensions;
   - no corner depth lookup;
   - no second corner deprojection request;
   - final corner/grasp pixels can have invalid local depth without zeroing the final XYZ.

5. **Fallback behavior**
   - HTTP `sample_deproject` remains available if shared ROI snapshot fails;
   - the shared header is still read for corner-ray intrinsics;
   - geometry is rejected if reliable intrinsics/plane data are unavailable instead of emitting fake zero coordinates.

6. **Diagnostics**
   - added snapshot-copy, vectorized-sample, vectorized-deprojection, ROI bytes and snapshot-attempt timings/status;
   - expected mode: `posix_shared_memory`, one snapshot attempt in normal operation.

7. **Project cleanup**
   - removed `M41.1_PERFORMANCE_NOTES.md`;
   - removed version-specific `watch_m41_1_performance.py` and separate offline validation script;
   - all tests now enter through `scripts/test_m41.py`;
   - future small versions should keep updating this file and that test entry point.

Validation before packaging:

- Python syntax checks pass;
- synthetic double-buffer shared-memory test: 96/96 depths valid with one ROI snapshot attempt;
- five historical RGB-D datasets: 5/5 pass;
- observed dimensions and plane RMS remain effectively unchanged from M41.1.
