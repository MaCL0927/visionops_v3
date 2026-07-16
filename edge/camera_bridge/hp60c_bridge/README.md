# VisionOps v3 HP60C SDK Bridge

HP60C/HP60CN uses the Angstrong SDK and runs independently from the Orbbec Gemini 336L Bridge.

| Camera | Service | Port |
|---|---|---:|
| HP60C / HP60CN | `visionops-hp60c-sdk-bridge.service` | `18181` |
| Orbbec Gemini 336L | `visionops-orbbec336l-bridge.service` | `18182` |

Both services may run concurrently. `config/active_camera.json` determines which bridge the Runtime and Web pages consume.

Endpoints: `/health`, `/stream/profiles`, `/stream/snapshot.jpg`, `/stream/depth.png`, `/stream/depth_vis.jpg`, `/stream/depth_meta`, `/stream.mjpeg`, `/stream/camera_info`.

The bridge monitors both RGB and depth freshness. When either stream is stale it clears the cached frames, disconnects MJPEG clients, destroys the old SDK listener/camera handles, and re-enumerates with exponential backoff. The external systemd watchdog restarts the service if the vendor SDK blocks.

The encrypted Angstrong configuration file controls the true sensor profile/exposure. Web settings can select the config file, RGB source, channel order, JPEG quality, display FPS and flips. Width/height/FPS fields describe the expected profile and are reported to VisionOps.

`/api/coordinate/deproject` is available only after setting `VISIONOPS_HP60C_FX/FY/CX/CY`; otherwise it returns HTTP 503 rather than inventing intrinsics.
