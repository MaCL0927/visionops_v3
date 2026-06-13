const state = {
  activePage: "capture",
  config: { snapshot_refresh_interval_ms: 1000, status_refresh_interval_ms: 2000 },
  latestResult: null,
};
const listeners = new Set();

export function getState() { return state; }
export function updateState(patch) { Object.assign(state, patch); listeners.forEach((listener) => listener(state)); }
export function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
