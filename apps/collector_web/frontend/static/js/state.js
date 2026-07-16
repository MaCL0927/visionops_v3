const LOCAL_CONFIG_KEY = "visionops_v3_frontend_config";

const defaultOverlay = {
  show_labels: true,
  show_centers: true,
  show_detection_bbox: true,
  show_obb_rotated: true,
  show_obb_bbox: false,
  show_segmentation_bbox: true,
  show_segmentation_mask: true,
  mask_opacity: 0.28,
};

const state = {
  activePage: "calibration",
  productionMode: false,
  config: {
    runtime_url: "",
    gateway_url: "",
    business_app_url: "",
    production_inference_source: "runtime",
    device_id: "",
    preview_refresh_interval_ms: 200,
    inference_interval_ms: 500,
    snapshot_refresh_interval_ms: 200,
    status_refresh_interval_ms: 2000,
    camera_type: "sdk_bridge",
    camera_model: "orbbec336l",
    hp60c_url: "http://127.0.0.1:18181",
    hp60c_snapshot_path: "/stream/snapshot.jpg",
    display_fps: 5,
    camera_read_fps: 30,
    camera_resolution: "1280x720",
    rgb_profile: "orbbec:1280x720@30",
    depth_profile: "orbbec:1280x720@30",
    depth_unit: "mm",
    rgb_source_preference: "auto",
    flip_vertical: "false",
    flip_horizontal: "false",
    rgb_order: "bgr",
    orbbec_serial: "",
    hp60c_config_path: "",
    hp60c_fx: 0,
    hp60c_fy: 0,
    hp60c_cx: 0,
    hp60c_cy: 0,
    camera_rotation: "0",
    camera_jpeg_quality: 100,
    camera_brightness: 50,
    camera_contrast: 50,
    camera_exposure_mode: "auto",
    default_mode: "factory",
    models_root: "/opt/visionops_v3/models",
    data_root: "/opt/visionops_v3/data",
    log_root: "/opt/visionops_v3/logs",
    disk_warning_percent: 85,
    upload: { server_ip: "", ssh_user: "", ssh_password: "", ssh_port: 22, remote_dir: "/opt/visionops_uploads", timeout_s: 60 },
    runtime_port: 28081,
    collector_port: 18091,
    preprocess_backend_preference: "auto",
    task_view_preference: "auto",
    overlay: { ...defaultOverlay },
  },
  savedConfig: null,
  latestResult: null,
  captureRecords: [],
};
const listeners = new Set();

function clampMs(value, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(100, Math.round(number));
}

function clampNumber(value, fallback, min = -Infinity, max = Infinity) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function fpsToIntervalMs(fps, fallbackMs = 200) {
  const number = clampNumber(fps, 1000 / fallbackMs, 1, 60);
  return Math.max(100, Math.round(1000 / number));
}

function normalizeOverlay(overlay = {}) {
  return {
    ...defaultOverlay,
    ...overlay,
    show_labels: overlay.show_labels ?? defaultOverlay.show_labels,
    show_centers: overlay.show_centers ?? defaultOverlay.show_centers,
    show_detection_bbox: overlay.show_detection_bbox ?? defaultOverlay.show_detection_bbox,
    show_obb_rotated: overlay.show_obb_rotated ?? defaultOverlay.show_obb_rotated,
    show_obb_bbox: overlay.show_obb_bbox ?? defaultOverlay.show_obb_bbox,
    show_segmentation_bbox: overlay.show_segmentation_bbox ?? defaultOverlay.show_segmentation_bbox,
    show_segmentation_mask: overlay.show_segmentation_mask ?? defaultOverlay.show_segmentation_mask,
    mask_opacity: clampNumber(overlay.mask_opacity, defaultOverlay.mask_opacity, 0, 1),
  };
}

export function normalizeConfig(config = {}) {
  const displayFps = clampNumber(config.display_fps ?? config.camera_read_fps, 5, 1, 30);
  const previewRefresh = fpsToIntervalMs(displayFps, 200);
  const overlay = normalizeOverlay(config.overlay || {});
  return {
    ...state.config,
    ...config,
    camera_type: "sdk_bridge",
    camera_model: config.camera_model || state.config.camera_model,
    display_fps: displayFps,
    preview_refresh_interval_ms: previewRefresh,
    inference_interval_ms: clampMs(config.inference_interval_ms, 500),
    snapshot_refresh_interval_ms: previewRefresh,
    status_refresh_interval_ms: clampMs(config.status_refresh_interval_ms, 2000),
    camera_read_fps: clampNumber(config.camera_read_fps, 30, 1, 60),
    rgb_profile: config.rgb_profile || state.config.rgb_profile,
    depth_profile: config.depth_profile || state.config.depth_profile,
    depth_unit: config.depth_unit || state.config.depth_unit,
    rgb_source_preference: config.rgb_source_preference || state.config.rgb_source_preference,
    flip_vertical: String(config.flip_vertical ?? state.config.flip_vertical),
    flip_horizontal: String(config.flip_horizontal ?? state.config.flip_horizontal),
    rgb_order: config.rgb_order || state.config.rgb_order,
    orbbec_serial: config.orbbec_serial || "",
    hp60c_config_path: config.hp60c_config_path || "",
    hp60c_fx: clampNumber(config.hp60c_fx, 0, 0),
    hp60c_fy: clampNumber(config.hp60c_fy, 0, 0),
    hp60c_cx: clampNumber(config.hp60c_cx, 0),
    hp60c_cy: clampNumber(config.hp60c_cy, 0),
    camera_jpeg_quality: clampNumber(config.camera_jpeg_quality, 100, 10, 100),
    camera_brightness: clampNumber(config.camera_brightness, 50, 0, 100),
    camera_contrast: clampNumber(config.camera_contrast, 50, 0, 100),
    disk_warning_percent: clampNumber(config.disk_warning_percent, 85, 50, 99),
    default_mode: ["factory", "production"].includes(config.default_mode) ? config.default_mode : "factory",
    production_inference_source: ["runtime", "app"].includes(config.production_inference_source)
      ? config.production_inference_source
      : "runtime",
    upload: {
      ...state.config.upload,
      ...(config.upload || {}),
      ssh_port: clampNumber(config.upload?.ssh_port, 22, 1, 65535),
      timeout_s: clampNumber(config.upload?.timeout_s, 60, 5, 3600),
    },
    runtime_port: clampNumber(config.runtime_port, 28081, 1, 65535),
    collector_port: clampNumber(config.collector_port, 18091, 1, 65535),
    overlay,
  };
}

export function loadPersistedConfig() {
  try {
    const raw = window.localStorage.getItem(LOCAL_CONFIG_KEY);
    if (!raw) return null;
    return normalizeConfig(JSON.parse(raw));
  } catch (_error) {
    return null;
  }
}

export function persistConfig(config) {
  const next = normalizeConfig(config);
  window.localStorage.setItem(LOCAL_CONFIG_KEY, JSON.stringify({
    preview_refresh_interval_ms: next.preview_refresh_interval_ms,
    snapshot_refresh_interval_ms: next.snapshot_refresh_interval_ms,
    inference_interval_ms: next.inference_interval_ms,
    status_refresh_interval_ms: next.status_refresh_interval_ms,
    camera_type: next.camera_type,
    camera_model: next.camera_model,
    hp60c_url: next.hp60c_url,
    hp60c_snapshot_path: next.hp60c_snapshot_path,
    display_fps: next.display_fps,
    camera_read_fps: next.camera_read_fps,
    camera_resolution: next.camera_resolution,
    rgb_profile: next.rgb_profile,
    depth_profile: next.depth_profile,
    depth_unit: next.depth_unit,
    rgb_source_preference: next.rgb_source_preference,
    flip_vertical: next.flip_vertical,
    flip_horizontal: next.flip_horizontal,
    rgb_order: next.rgb_order,
    orbbec_serial: next.orbbec_serial,
    hp60c_config_path: next.hp60c_config_path,
    hp60c_fx: next.hp60c_fx,
    hp60c_fy: next.hp60c_fy,
    hp60c_cx: next.hp60c_cx,
    hp60c_cy: next.hp60c_cy,
    camera_rotation: next.camera_rotation,
    camera_jpeg_quality: next.camera_jpeg_quality,
    camera_brightness: next.camera_brightness,
    camera_contrast: next.camera_contrast,
    camera_exposure_mode: next.camera_exposure_mode,
    default_mode: next.default_mode,
    models_root: next.models_root,
    data_root: next.data_root,
    log_root: next.log_root,
    disk_warning_percent: next.disk_warning_percent,
    upload: next.upload,
    runtime_port: next.runtime_port,
    collector_port: next.collector_port,
    preprocess_backend_preference: next.preprocess_backend_preference,
    task_view_preference: next.task_view_preference,
    overlay: next.overlay,
  }));
}

export function getState() { return state; }
export function updateState(patch) { Object.assign(state, patch); listeners.forEach((listener) => listener(state)); }
export function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
export { defaultOverlay };
