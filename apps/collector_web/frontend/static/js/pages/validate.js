import { ApiError, endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { updateState } from "../state.js";
import { clearOverlay, drawInferenceOverlay } from "../render/overlay.js";

const image = document.getElementById("validate-image");
const canvas = document.getElementById("validate-overlay");
const empty = document.getElementById("validate-empty");
const output = document.getElementById("validate-result");
const modelList = document.getElementById("model-list");
const modelCatalogStatus = document.getElementById("model-catalog-status");

let snapshotUrl = null;
let currentResult = null;
let realtimeTimer = null;
let realtimeBusy = false;
let currentCatalog = null;
let switchingModelId = null;

function showResult(result) {
  currentResult = result;
  updateState({ latestResult: result });
  output.textContent = JSON.stringify(result, null, 2);
  const timing = result.timing || {};
  for (const key of ["preprocess", "inference", "postprocess", "total"]) {
    document.getElementById(`timing-${key}`).textContent = timing[`${key}_ms`] == null ? "--" : `${timing[`${key}_ms`]} ms`;
  }
  document.getElementById("model-name").textContent = result.model?.model_name || "--";
  document.getElementById("model-version").textContent = result.model?.model_version || "--";
  document.getElementById("model-task").textContent = result.task_type || "--";
}

function showModelStatus(model) {
  document.getElementById("model-name").textContent = model?.model_name || model?.name || "--";
  document.getElementById("model-version").textContent = model?.model_version || model?.version || "--";
  document.getElementById("model-task").textContent = model?.task_type || "--";
}

function formatBytes(value) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function catalogMessage(text, kind = "") {
  modelCatalogStatus.textContent = text;
  modelCatalogStatus.className = "inline-note";
  if (kind) modelCatalogStatus.classList.add(kind);
}

function renderModelList() {
  const models = currentCatalog?.models || [];
  if (!models.length) {
    modelList.innerHTML = '<div class="empty-copy">models_root 下暂无可识别的模型包</div>';
    return;
  }
  modelList.innerHTML = "";
  for (const model of models) {
    const card = document.createElement("article");
    card.className = `model-list-card${model.active ? " active" : ""}${model.valid ? "" : " invalid"}`;
    const statusText = model.active ? "当前使用中" : (model.valid ? "可切换" : "模型包无效");
    const statusClass = model.active ? "active" : (model.valid ? "" : "error");
    const switchDisabled = !model.valid || model.active || switchingModelId === model.model_id;
    card.innerHTML = `
      <div class="model-list-head">
        <div>
          <b>${model.model_name || model.package_dir}</b>
          <small>${model.model_version || "--"} / ${model.package_dir}</small>
        </div>
        <span class="status-pill ${statusClass}">${statusText}</span>
      </div>
      <div class="model-meta">
        <div><span>任务类型</span><b>${model.task_type || "--"}</b></div>
        <div><span>目标平台</span><b>${model.target_platform || "--"}</b></div>
        <div><span>输入尺寸</span><b>${Array.isArray(model.input_size) ? model.input_size.join(" x ") : "--"}</b></div>
        <div><span>类别数量</span><b>${model.labels_count ?? "--"}</b></div>
        <div><span>模型大小</span><b>${formatBytes(model.rknn_size_bytes)}</b></div>
        <div><span>模型标识</span><b>${model.model_id || "--"}</b></div>
      </div>
      <div class="model-card-status"></div>
    `;
    const footer = card.querySelector(".model-card-status");
    if (model.error) {
      const error = document.createElement("span");
      error.className = "status-pill error";
      error.textContent = model.error;
      footer.appendChild(error);
    }
    if (!model.active) {
      const button = document.createElement("button");
      button.textContent = switchingModelId === model.model_id ? "切换中..." : "切换到该模型";
      button.disabled = switchDisabled;
      button.addEventListener("click", () => switchModel(model.model_id));
      footer.appendChild(button);
    }
    modelList.appendChild(card);
  }
}

async function refreshImage() {
  const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
  if (snapshotUrl) URL.revokeObjectURL(snapshotUrl);
  snapshotUrl = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = snapshotUrl;
  });
  empty.classList.add("hidden");
  drawInferenceOverlay(canvas, image, currentResult);
}

export async function inferOnce() {
  try {
    const result = await postJson(endpoints.inferOnce);
    showResult(result);
    await refreshImage();
    return result;
  } catch (error) {
    output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2);
    empty.classList.remove("hidden");
    empty.textContent = "Runtime infer_once unavailable";
    clearOverlay(canvas);
    return null;
  }
}

export async function refreshLatestResult() {
  try {
    const result = await requestJson(endpoints.latestResult);
    showResult(result);
    await refreshImage();
  } catch (error) {
    output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2);
  }
}

export async function refreshRuntimeStatus() {
  try {
    const status = await requestJson(endpoints.runtimeStatus);
    document.getElementById("validate-runtime-status").textContent = JSON.stringify(status, null, 2);
    showModelStatus(status.loaded_model || {});
  } catch (error) {
    document.getElementById("validate-runtime-status").textContent = JSON.stringify(error.body || { status: "unreachable" }, null, 2);
  }
}

export async function refreshModelCatalog() {
  try {
    const catalog = await requestJson(endpoints.models);
    currentCatalog = catalog;
    showModelStatus(catalog.current_model || {});
    renderModelList();
    const modelsRoot = catalog.models_root || "--";
    const count = Array.isArray(catalog.models) ? catalog.models.length : 0;
    catalogMessage(`已扫描 ${count} 个模型包，models_root=${modelsRoot}`, "ok");
  } catch (error) {
    currentCatalog = null;
    renderModelList();
    catalogMessage(`扫描模型失败: ${error.body?.error?.message || error.message}`, "error");
  }
}

export async function switchModel(modelId) {
  switchingModelId = modelId;
  renderModelList();
  catalogMessage(`正在切换模型: ${modelId}`, "");
  try {
    const status = await postJson(endpoints.switchModel, { model_id: modelId });
    showModelStatus(status.loaded_model || {});
    document.getElementById("validate-runtime-status").textContent = JSON.stringify(status, null, 2);
    catalogMessage(`模型切换成功: ${status.loaded_model?.model_name || modelId}`, "ok");
    await refreshModelCatalog();
  } catch (error) {
    const detail = error instanceof ApiError ? (error.body?.error?.message || error.message) : String(error);
    catalogMessage(`模型切换失败: ${detail}`, "error");
  } finally {
    switchingModelId = null;
    renderModelList();
  }
}

function toggleRealtime() {
  const button = document.getElementById("validate-realtime");
  if (realtimeTimer) {
    clearInterval(realtimeTimer);
    realtimeTimer = null;
    button.textContent = "实时检测";
    button.setAttribute("aria-pressed", "false");
    return;
  }
  button.textContent = "停止实时检测";
  button.setAttribute("aria-pressed", "true");
  realtimeTimer = setInterval(async () => {
    if (realtimeBusy) return;
    realtimeBusy = true;
    try { await inferOnce(); } finally { realtimeBusy = false; }
  }, 1500);
  inferOnce();
}

export function initValidate() {
  document.getElementById("validate-infer").addEventListener("click", inferOnce);
  document.getElementById("validate-photo").addEventListener("click", inferOnce);
  document.getElementById("validate-realtime").addEventListener("click", toggleRealtime);
  document.getElementById("validate-refresh").addEventListener("click", refreshLatestResult);
  document.getElementById("validate-runtime-refresh").addEventListener("click", refreshRuntimeStatus);
  document.getElementById("validate-model-scan").addEventListener("click", refreshModelCatalog);
  window.addEventListener("resize", () => currentResult && drawInferenceOverlay(canvas, image, currentResult));
  refreshRuntimeStatus();
  refreshModelCatalog();
}
