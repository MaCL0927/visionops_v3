const LOCAL_CONFIG_KEY = "visionops_v3_frontend_config";

const state = {
  activePage: "calibration",
  productionMode: false,
  config: {
    runtime_url: "",
    gateway_url: "",
    business_app_url: "",
    device_id: "",
    preview_refresh_interval_ms: 200,
    inference_interval_ms: 500,
    snapshot_refresh_interval_ms: 200,
    status_refresh_interval_ms: 2000,
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

export function normalizeConfig(config = {}) {
  const previewRefresh = clampMs(
    config.preview_refresh_interval_ms ?? config.snapshot_refresh_interval_ms,
    200,
  );
  return {
    ...state.config,
    ...config,
    preview_refresh_interval_ms: previewRefresh,
    inference_interval_ms: clampMs(config.inference_interval_ms, 500),
    snapshot_refresh_interval_ms: previewRefresh,
    status_refresh_interval_ms: clampMs(config.status_refresh_interval_ms, 2000),
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
    inference_interval_ms: next.inference_interval_ms,
    status_refresh_interval_ms: next.status_refresh_interval_ms,
  }));
}

export function getState() { return state; }
export function updateState(patch) { Object.assign(state, patch); listeners.forEach((listener) => listener(state)); }
export function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
