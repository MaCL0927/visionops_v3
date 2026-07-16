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
const liveStage = image?.parentElement;

let snapshotUrl = null;
let latestResult = null;
let liveTimer = null;
let liveBusy = false;
let liveEnabled = false;
let activeView = "live";
let lastLoopFinishedAt = null;
let overlayResizeFrame = null;

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

function placementSummary(result) {
  const placement = result?.placement;
  if (!placement || Number(placement.layer) !== 1) return null;
  const occupied = Number(placement.occupied_count || 0);
  const total = Number(placement.slot_count || placement.slots?.length || 0);
  if (placement.state === "WAIT_TRAY") return "等待托盘";
  if (placement.complete) return `第一层已放满 ${occupied}/${total}`;
  const next = placement.next_slot_id ? ` · 下一位置 ${placement.next_slot_id}` : "";
  return `第一层 ${occupied}/${total}${next}`;
}

function updateLiveSummary(result, elapsedMs = null) {
  latestResult = result;
  updateState({ latestResult: result });
  const task = result?.task_type || "--";
  const count = countResults(result);
  const total = result?.timing?.total_ms ?? result?.timing_detail?.total_ms;
  currentModel.textContent = modelDisplay(result);
  resultBrief.textContent = placementSummary(result) || `${task} / ${count} 个结果`;
  timingTotal.textContent = formatMs(total);
  const configuredFps = 1000 / Math.max(100, Number(getState().config.inference_interval_ms || 500));
  const actualFps = elapsedMs ? 1000 / elapsedMs : configuredFps;
  fpsText.textContent = `${formatFps(actualFps)} / 设定 ${formatFps(configuredFps)}`;
  liveStatus.textContent = `实时检测中 · ${task}`;
}

/**
 * 将图像元素本身缩放到舞台内能容纳的最大尺寸。
 *
 * 不能只依赖 img { width:auto; max-width:100% }，因为浏览器不会把图像
 * 从固有尺寸主动放大。这里显式计算 contain 尺寸，使图像保持完整、保持
 * 长宽比，并在当前舞台内尽可能铺满。Canvas 随后直接对齐该图像元素。
 */
function fitProductionImage() {
  if (!liveStage || !image?.naturalWidth || !image?.naturalHeight) return false;
  const availableWidth = Math.max(1, liveStage.clientWidth);
  const availableHeight = Math.max(1, liveStage.clientHeight);
  const scale = Math.min(
    availableWidth / image.naturalWidth,
    availableHeight / image.naturalHeight,
  );
  if (!Number.isFinite(scale) || scale <= 0) return false;
  const width = Math.max(1, Math.floor(image.naturalWidth * scale));
  const height = Math.max(1, Math.floor(image.naturalHeight * scale));
  image.style.width = `${width}px`;
  image.style.height = `${height}px`;
  return true;
}

function redrawOverlayAfterLayout() {
  if (overlayResizeFrame) cancelAnimationFrame(overlayResizeFrame);
  overlayResizeFrame = requestAnimationFrame(() => {
    overlayResizeFrame = requestAnimationFrame(() => {
      overlayResizeFrame = null;
      fitProductionImage();
      if (latestResult && image.complete && image.naturalWidth) {
        drawInferenceOverlay(canvas, image, latestResult);
      }
    });
  });
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
  redrawOverlayAfterLayout();
}

export async function productionInferOnce() {
  const startedAt = performance.now();
  try {
    const source = getState().config.production_inference_source || "runtime";
    const response = source === "app"
      ? await postJson(endpoints.appEvaluate)
      : await postJson(endpoints.inferOnce);
    const result = response?.visualization_result || response?.runtime_result || response;
    if (!result || result.message_type !== "inference_result") {
      throw new Error("生产业务应用未返回 visualization_result/inference_result");
    }
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
    redrawOverlayAfterLayout();
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
  window.addEventListener("resize", redrawOverlayAfterLayout);
  window.addEventListener("visionops:settings-saved", redrawOverlayAfterLayout);
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

if (typeof ResizeObserver !== "undefined" && liveStage) {
  const productionResizeObserver = new ResizeObserver(redrawOverlayAfterLayout);
  productionResizeObserver.observe(liveStage);
}
