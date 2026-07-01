const state = {
  activePage: "calibration",
  productionMode: false,
  config: {
    runtime_url: "",
    gateway_url: "",
    business_app_url: "",
    device_id: "",
    snapshot_refresh_interval_ms: 1000,
    status_refresh_interval_ms: 2000,
  },
  savedConfig: null,
  latestResult: null,
  captureRecords: [],
};
const listeners = new Set();

export function getState() { return state; }
export function updateState(patch) { Object.assign(state, patch); listeners.forEach((listener) => listener(state)); }
export function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
