# M41 development notes

## Dataset verification supplied with M40

The five supplied RGB-D captures were re-evaluated with the M41 plane-sampling path (96 distributed interior samples, 5 mm RANSAC threshold).

| capture | observed L (mm) | observed W (mm) | plane RMS (mm) | inlier ratio |
|---|---:|---:|---:|---:|
| 072604 | 696.7 | 514.9 | 2.06 | 0.979 |
| 072625 | 698.4 | 515.0 | 1.69 | 1.000 |
| 072851 | 697.5 | 507.9 | 1.42 | 0.990 |
| 074126 | 696.7 | 514.7 | 1.00 | 0.989 |
| 074302 | 692.3 | 506.5 | 1.38 | 0.979 |

The physical dimensions later supplied by the user are 715 x 525 mm. The systematic under-size in the observed quadrilateral is therefore treated as mask/annotation boundary bias rather than a reason to abandon the cuboid prior. M41 regularises the valid observed plane to the exact physical size.

The same five captures span a large camera/top-surface Z change (roughly 0.55--0.89 m for the reconstructed edge midpoints), which validates the decision not to hard-code camera height into the geometry.

## M41 scope

Frozen now:

- full top face mostly visible;
- one top bundle returned;
- fixed 715 x 525 mm top rectangle;
- variable bundle stack height;
- variable robot-eye camera height;
- camera-frame 3-D output only;
- robot performs hand-eye conversion.

Not implemented yet because interface details are not available / not needed for geometry:

- direct robot Python SDK login and waist-Z reading;
- partial-view / one-corner reconstruction;
- explicit multi-candidate topmost ranking when several complete top faces are simultaneously detected.
