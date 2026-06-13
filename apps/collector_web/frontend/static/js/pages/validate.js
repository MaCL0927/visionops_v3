import { endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { updateState } from "../state.js";
import { clearOverlay, drawInferenceOverlay } from "../render/overlay.js";

const image = document.getElementById("validate-image"), canvas = document.getElementById("validate-overlay"), empty = document.getElementById("validate-empty"), output = document.getElementById("validate-result");
let snapshotUrl = null, currentResult = null, realtimeTimer = null, realtimeBusy = false;

function showResult(result) {
  currentResult = result; updateState({ latestResult: result }); output.textContent = JSON.stringify(result, null, 2);
  const timing = result.timing || {};
  for (const key of ["preprocess", "inference", "postprocess", "total"]) document.getElementById(`timing-${key}`).textContent = timing[`${key}_ms`] == null ? "--" : `${timing[`${key}_ms`]} ms`;
  document.getElementById("model-name").textContent = result.model?.model_name || "--";
  document.getElementById("model-version").textContent = result.model?.model_version || "--";
  document.getElementById("model-task").textContent = result.task_type || "--";
}

async function refreshImage() {
  const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`); if (snapshotUrl) URL.revokeObjectURL(snapshotUrl); snapshotUrl = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = snapshotUrl; }); empty.classList.add("hidden"); drawInferenceOverlay(canvas, image, currentResult);
}

export async function inferOnce() {
  try { const result = await postJson(endpoints.inferOnce); showResult(result); await refreshImage(); return result; }
  catch (error) { output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2); empty.classList.remove("hidden"); empty.textContent = "Runtime infer_once unavailable"; clearOverlay(canvas); return null; }
}
export async function refreshLatestResult() { try { const result = await requestJson(endpoints.latestResult); showResult(result); await refreshImage(); } catch (error) { output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2); } }
export async function refreshRuntimeStatus() {
  try { const status = await requestJson(endpoints.runtimeStatus); document.getElementById("validate-runtime-status").textContent = JSON.stringify(status, null, 2); const model = status.loaded_model || {}; document.getElementById("model-name").textContent = model.model_name || model.name || "未加载 / Mock"; document.getElementById("model-version").textContent = model.model_version || model.version || "--"; }
  catch (error) { document.getElementById("validate-runtime-status").textContent = JSON.stringify(error.body || { status: "unreachable" }, null, 2); }
}
function toggleRealtime() {
  const button = document.getElementById("validate-realtime");
  if (realtimeTimer) { clearInterval(realtimeTimer); realtimeTimer = null; button.textContent = "实时检测"; button.setAttribute("aria-pressed", "false"); return; }
  button.textContent = "停止实时检测"; button.setAttribute("aria-pressed", "true");
  realtimeTimer = setInterval(async () => { if (realtimeBusy) return; realtimeBusy = true; try { await inferOnce(); } finally { realtimeBusy = false; } }, 1500);
  inferOnce();
}

export function initValidate() {
  document.getElementById("validate-infer").addEventListener("click", inferOnce);
  document.getElementById("validate-photo").addEventListener("click", inferOnce);
  document.getElementById("validate-realtime").addEventListener("click", toggleRealtime);
  document.getElementById("validate-refresh").addEventListener("click", refreshLatestResult);
  document.getElementById("validate-runtime-refresh").addEventListener("click", refreshRuntimeStatus);
  window.addEventListener("resize", () => currentResult && drawInferenceOverlay(canvas, image, currentResult));
  refreshRuntimeStatus();
}
