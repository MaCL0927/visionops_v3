# M16.4 Production Mode Live View

## Scope

Updated Collector Web production mode so the default production screen is a full-page live inference view. The previous JSON/status dashboard is retained as a secondary view reachable through the `消息状态` button.

## Behavior

- Entering production mode starts a realtime loop that calls the same Runtime API used by model validation:
  - `POST /api/runtime/infer_once`
  - `GET /api/runtime/snapshot.jpg`
- The active Runtime model is used, so production view follows the model selected in the model validation page / Runtime model switch.
- The realtime loop interval uses the same frontend setting as model validation: `config.inference_interval_ms`.
- Overlay rendering reuses `drawInferenceOverlay`, so Detection / OBB / Segmentation visualization settings stay consistent with model validation.
- The production page shows current model, task/result count, total latency, and actual/configured FPS.
- Status cards, latest result JSON, Gateway/Business App messages, and registers are moved to the secondary status view.

## Admin guard

Returning from production mode to factory mode now requires administrator authentication.

Current fixed test credentials:

- username: `admin`
- password: `admin`

The guard applies to both the top-right `返回工厂模式` button and direct clicks on factory tabs while production mode is active.

## Changed files

- `apps/collector_web/frontend/index.html`
- `apps/collector_web/frontend/static/js/main.js`
- `apps/collector_web/frontend/static/js/pages/production.js`
- `apps/collector_web/frontend/static/css/main.css`
