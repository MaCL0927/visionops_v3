# M15.1 Algorithm Settings UI and YAML Apply

## Scope

This change refines the algorithm settings panel and connects model threshold settings to the selected model package YAML.

## Main changes

- The algorithm panel is split into public settings and four task-specific sections:
  - Classification
  - Detection
  - OBB
  - Segmentation
- Automatic inference interval is displayed as inference FPS and stored as `inference_interval_ms` in frontend local settings.
- Preprocess backend preference is no longer shown. The production default is RGA and Runtime is still controlled by its startup command/systemd config.
- Task view preference is removed from the UI.
- The selected model is loaded from the standard M15 model package list.
- Only the selected model task section is editable; other task sections are disabled and greyed out.
- Confidence and NMS thresholds are initialized from `/opt/visionops_v3/models/<model>/model.yaml` and saved back to that file.
- If the edited model is the active Runtime model, Collector asks Runtime to reload the same `model_dir` after writing YAML.
- Model card platform display now uses `dataset.device_id` when `target_platform/platform` is absent.

## Backend API

New API:

```text
GET  /api/settings/algorithm
POST /api/settings/algorithm
```

`model.yaml` remains the single source of truth for model metadata and algorithm thresholds.

## Notes

- M15 model package format remains strict: `model.rknn + model.yaml` only.
- No `manifest.json` or `labels.txt` compatibility was added.
- Threshold keys are updated in place when aliases are already used, for example `conf_threshold`; otherwise the canonical `score_threshold` and `nms_threshold` keys are written.
