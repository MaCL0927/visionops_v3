import { endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { getState, updateState } from "../state.js";
import { clearOverlay, drawInferenceOverlay } from "../render/overlay.js";

const image = document.getElementById("production-image");
const canvas = document.getElementById("production-overlay");
const empty = document.getElementById("production-empty");
const liveStatus = document.getElementById("production-live-status");
const currentModel = document.getElementById("production-current-model");
const resultBrief = document.getElementById("production-result-brief");
const timingTotal = document.getElementById("production-timing-total");
const fpsText = document.getElementById("production-fps");
const liveView = document.getElementById("production-live-view");
const statusView = document.getElementById("production-status-view");
const viewToggle = document.getElementById("production-view-toggle");
const refreshButton = document.getElementById("production-refresh");

let snapshotUrl = null;
let latestResult = null;
let liveTimer = null;
let liveBusy = false;
let liveEnabled = false;
let activeView = "live";
let lastLoopFinishedAt = null;

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${number.toFixed(number >= 100 ? 1 : 2)} ms`;
}

function formatFps(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "--";
  return `${number.toFixed(number >= 10 ? 1 : 2)} FPS`;
}

function countResults(result) {
  const detections = Array.isArray(result?.detections) ? result.detections.length : 0;
  const classifications = Array.isArray(result?.classifications) ? result.classifications.length : 0;
  return detections || classifications || 0;
}

function modelDisplay(result) {
  const model = result?.model || {};
  return model.model_name || model.model_id || "--";
}

function updateLiveSummary(result, elapsedMs = null) {
  latestResult = result;
  updateState({ latestResult: result });
  const task = result?.task_type || "--";
  const count = countResults(result);
  const total = result?.timing?.total_ms ?? result?.timing_detail?.total_ms;
  currentModel.textContent = modelDisplay(result);
  resultBrief.textContent = `${task} / ${count} 个结果`;
  timingTotal.textContent = formatMs(total);
  const configuredFps = 1000 / Math.max(100, Number(getState().config.inference_interval_ms || 500));
  const actualFps = elapsedMs ? 1000 / elapsedMs : configuredFps;
  fpsText.textContent = `${formatFps(actualFps)} / 设定 ${formatFps(configuredFps)}`;
  liveStatus.textContent = `实时检测中 · ${task}`;
}

async function displaySnapshot() {
  const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
  if (snapshotUrl) URL.revokeObjectURL(snapshotUrl);
  snapshotUrl = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = snapshotUrl;
  });
  empty.classList.add("hidden");
  if (latestResult) drawInferenceOverlay(canvas, image, latestResult);
}

export async function productionInferOnce() {
  const startedAt = performance.now();
  try {
    const result = await postJson(endpoints.inferOnce);
    const finishedAt = performance.now();
    const elapsedMs = lastLoopFinishedAt ? finishedAt - lastLoopFinishedAt : finishedAt - startedAt;
    lastLoopFinishedAt = finishedAt;
    updateLiveSummary(result, elapsedMs);
    await displaySnapshot();
    return result;
  } catch (error) {
    empty.classList.remove("hidden");
    empty.textContent = error.body?.error?.message || error.message || "生产实时检测失败";
    liveStatus.textContent = "实时检测异常";
    resultBrief.textContent = "检测失败";
    timingTotal.textContent = "--";
    clearOverlay(canvas);
    return null;
  }
}

function scheduleLiveLoop(delayMs = 0) {
  if (liveTimer) clearTimeout(liveTimer);
  liveTimer = setTimeout(runLiveLoop, Math.max(0, delayMs));
}

async function runLiveLoop() {
  if (!liveEnabled || activeView !== "live") return;
  if (liveBusy) {
    scheduleLiveLoop(getState().config.inference_interval_ms);
    return;
  }
  liveBusy = true;
  try {
    await productionInferOnce();
  } finally {
    liveBusy = false;
    if (liveEnabled && activeView === "live") scheduleLiveLoop(getState().config.inference_interval_ms);
  }
}

function startLive() {
  if (liveEnabled) return;
  liveEnabled = true;
  liveStatus.textContent = "实时检测启动中";
  lastLoopFinishedAt = null;
  scheduleLiveLoop(0);
}

function stopLive() {
  liveEnabled = false;
  liveBusy = false;
  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }
}

function renderStatus(id, badgeId, value) {
  document.getElementById(id).textContent = JSON.stringify(value, null, 2);
  const health = value?.health || value?.status || (value?.reachable === false ? "unreachable" : "ok");
  const badge = document.getElementById(badgeId); badge.textContent = health; badge.className = `badge ${health}`;
}

function renderRegisters(id, payload) {
  const target = document.getElementById(id), registers = payload?.registers;
  if (!Array.isArray(registers)) { target.className = "register-table empty-copy"; target.textContent = "unreachable / no data"; return; }
  target.className = "register-table"; target.replaceChildren();
  const table = document.createElement("table"), head = document.createElement("thead"), headerRow = document.createElement("tr"), body = document.createElement("tbody");
  for (const label of ["地址", "名称", "值", "类型"]) { const cell = document.createElement("th"); cell.textContent = label; headerRow.append(cell); }
  head.append(headerRow); table.append(head, body);
  for (const item of registers) { const row = document.createElement("tr"); for (const value of [item.address, item.name, item.value, item.type]) { const cell = document.createElement("td"); cell.textContent = String(value ?? ""); row.append(cell); } body.append(row); }
  target.append(table);
}

async function safe(path) { try { return await requestJson(path); } catch (error) { return error.body || { status: "unreachable", reachable: false, error: { message: error.message } }; } }

export async function refreshProductionStatus() {
  const [collector, runtime, gateway, app, latestRuntimeResult] = await Promise.all([safe(endpoints.collectorStatus), safe(endpoints.runtimeStatus), safe(endpoints.gatewayStatus), safe(endpoints.appStatus), safe(endpoints.latestResult)]);
  renderStatus("collector-status", "collector-badge", collector.collector || collector); renderStatus("runtime-status", "runtime-badge", collector.runtime?.status_response || runtime); renderStatus("gateway-status", "gateway-badge", gateway); renderStatus("app-status", "app-badge", app);
  document.getElementById("production-result-summary").textContent = JSON.stringify(latestRuntimeResult, null, 2);
  document.getElementById("production-gateway-summary").textContent = JSON.stringify(gateway.latest_gateway_message || { status: gateway.status || "no_message" }, null, 2);
  document.getElementById("production-app-summary").textContent = JSON.stringify(app.latest_decision || { status: app.status || "no_decision" }, null, 2);
  const [gatewayRegisters, appRegisters] = await Promise.all([gateway.reachable === false ? null : safe(endpoints.gatewayRegisters), app.reachable === false ? null : safe(endpoints.appRegisters)]);
  renderRegisters("gateway-registers", gatewayRegisters); renderRegisters("app-registers", appRegisters);
}

export async function refreshProduction() {
  if (activeView === "status") await refreshProductionStatus();
}

function showProductionView(view) {
  activeView = view === "status" ? "status" : "live";
  liveView.classList.toggle("active", activeView === "live");
  statusView.classList.toggle("active", activeView === "status");
  viewToggle.textContent = activeView === "live" ? "消息状态" : "生产画面";
  refreshButton.textContent = activeView === "live" ? "手动检测" : "手动刷新";
  if (activeView === "status") {
    stopLive();
    refreshProductionStatus();
  } else {
    startLive();
    setTimeout(() => latestResult && drawInferenceOverlay(canvas, image, latestResult), 0);
  }
}

export function setProductionActive(active) {
  if (active) {
    showProductionView("live");
    startLive();
  } else {
    stopLive();
  }
}

export function initProduction() {
  refreshButton.addEventListener("click", async () => {
    if (activeView === "status") await refreshProductionStatus();
    else await productionInferOnce();
  });
  viewToggle.addEventListener("click", () => showProductionView(activeView === "live" ? "status" : "live"));
  window.addEventListener("resize", () => latestResult && activeView === "live" && drawInferenceOverlay(canvas, image, latestResult));
  window.addEventListener("visionops:settings-saved", () => latestResult && activeView === "live" && drawInferenceOverlay(canvas, image, latestResult));
  const evaluateBtn = document.getElementById("production-app-evaluate");
  if (evaluateBtn) {
    evaluateBtn.addEventListener("click", async () => {
      const summary = document.getElementById("production-app-summary");
      try {
        const decision = await postJson(endpoints.appEvaluate);
        summary.textContent = JSON.stringify(decision, null, 2);
      } catch (error) {
        summary.textContent = JSON.stringify(error.body || { status: "error", error: { message: error.message } }, null, 2);
      }
      if (activeView === "status") await refreshProductionStatus();
    });
  }
}
