# M34_new.4 Rim-Pinch, Calibrated 3-D Box, and Depth-Based Neighbor Collision

M34_new.4 replaces the old hard 2-D neighbor-mask overlap rejection with aligned-depth, per-instance neighbor point clouds and 3-D finger swept-volume checks.

See `production/foam_ring_grasp/README.md` for:

- point-cloud ownership and target-surface suppression;
- six finger motion stages;
- 3-D collision status and thresholds;
- 2-D warning-only compatibility fields;
- offline commands and robot-interface output.
