export class ApiError extends Error {
  constructor(message, status, body) { super(message); this.status = status; this.body = body; }
}

export async function requestJson(path, options = {}) {
  let response;
  try { response = await fetch(path, { cache: "no-store", ...options }); }
  catch (error) { throw new ApiError(error.message, 0, { status: "unreachable" }); }
  let body;
  try { body = await response.json(); }
  catch (_error) { throw new ApiError(`响应不是 JSON: ${path}`, response.status, null); }
  if (!response.ok) throw new ApiError(`HTTP ${response.status}`, response.status, body);
  return body;
}

export function postJson(path) {
  return requestJson(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
}

export async function requestBlob(path) {
  let response;
  try { response = await fetch(path, { cache: "no-store" }); }
  catch (error) { throw new ApiError(error.message, 0, { status: "unreachable" }); }
  if (!response.ok) {
    let body = null;
    try { body = await response.json(); } catch (_error) { /* 保留空错误体 */ }
    throw new ApiError(`HTTP ${response.status}`, response.status, body);
  }
  return response.blob();
}

export const endpoints = {
  frontendConfig: "/api/collector/config", collectorStatus: "/api/collector/status",
  runtimeStatus: "/api/runtime/status", inferOnce: "/api/runtime/infer_once",
  latestResult: "/api/runtime/latest_result", snapshot: "/api/runtime/snapshot.jpg",
  gatewayStatus: "/api/gateway/status", gatewayRegisters: "/api/gateway/registers",
  appStatus: "/api/app/status", appRegisters: "/api/app/registers",
};
