# M41 Carton Bundle Grasp Robot Protocol

Robot WebSocket endpoint: `ws://<vision-box>:9001/vision`.

M41 keeps the same point-level contract used by `box_grasp_vision`: one physical bundle produces **two elements in `items[]` with the same `id`**.

Example:

```json
{
  "type": "detection",
  "frame_id": 1234,
  "timestamp": 1787210763.599,
  "request_id": "robot-17",
  "items": [
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.96,
      "position_camera": [-314.2, -20.1, 888.8],
      "center_px": [426.0, 351.1]
    },
    {
      "id": 0,
      "class_id": 0,
      "confidence": 0.96,
      "position_camera": [399.4, 19.8, 870.4],
      "center_px": [921.0, 378.7]
    }
  ],
  "fault_code": 0,
  "fault_type": "NONE"
}
```

`position_camera`:

- unit: mm;
- frame: Orbbec color-camera coordinate frame;
- points: exact M41 fixed-size 3-D midpoints of the two 525 mm width edges;
- no camera-pitch or robot-waist correction has been applied.

Robot integration should apply its hand-eye transform to each `position_camera` point. The two points are sorted deterministically by `center_px.x`; no left/right semantic field is required.

## Trigger

```json
{"type":"control","command":"trigger","request_id":"robot-17"}
```

The App first acknowledges the trigger and later returns the detection carrying the same `request_id`. Trigger results are protected from latest-only continuous-frame replacement.

## Continuous mode

```json
{"type":"control","command":"start","request_id":"ctl-1"}
{"type":"control","command":"stop","request_id":"ctl-2"}
```

There is no independent WebSocket push Hz. Every completed production inference is pushed while continuous mode is enabled.
