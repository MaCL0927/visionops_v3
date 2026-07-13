import { ApiError, endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { getState, updateState } from "../state.js";
import { clearOverlay, drawInferenceOverlay } from "../render/overlay.js";

const image = document.getElementById("validate-image");
const canvas = document.getElementById("validate-overlay");
const empty = document.getElementById("validate-empty");
const modelList = document.getElementById("model-list");
const modelCatalogStatus = document.getElementById("model-catalog-status");
const targetSummary = document.getElementById("validate-target-summary");
const resultBrief = document.getElementById("validate-result-brief");
const capturePickerPanel = document.getElementById("validate-picker-panel");
const captureList = document.getElementById("validate-capture-list");
const roiModal = document.getElementById("validate-roi-modal");
const roiImage = document.getElementById("validate-roi-image");
const roiCanvas = document.getElementById("validate-roi-canvas");
const roiEmpty = document.getElementById("validate-roi-empty");
const roiCoordinates = document.getElementById("validate-roi-coordinates");

let snapshotUrl = null;
let currentResult = null;
let realtimeTimer = null;
let realtimeBusy = false;
let currentCatalog = null;
let switchingModelId = null;
let selectedCaptureRecord = null;
let realtimeEnabled = false;
let roiSnapshotUrl = null;
let roiDraft = { enabled: false, x1: 0, y1: 0, x2: 1, y2: 1 };
let roiDrawing = false;
let roiStartPoint = null;

function formatBytes(value) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function formatNumber(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function catalogMessage(text, kind = "") {
  modelCatalogStatus.textContent = text;
  modelCatalogStatus.className = "inline-note";
  if (kind) modelCatalogStatus.classList.add(kind);
}

function renderEmptySummary(text) {
  targetSummary.innerHTML = `<div class="empty-copy">${text}</div>`;
  resultBrief.textContent = "尚无结果";
}

function setRealtimeButtonState(running) {
  const button = document.getElementById("validate-realtime");
  button.textContent = running ? "停止实时" : "实时检测";
  button.setAttribute("aria-pressed", running ? "true" : "false");
  button.classList.toggle("active", running);
}

function stopRealtimeLoop() {
  realtimeEnabled = false;
  if (realtimeTimer) {
    clearTimeout(realtimeTimer);
    realtimeTimer = null;
  }
  setRealtimeButtonState(false);
}

function hidePicker() {
  capturePickerPanel.classList.add("hidden");
}

function renderCapturePicker() {
  const records = getState().captureRecords || [];
  captureList.replaceChildren();
  if (!records.length) {
    captureList.innerHTML = '<div class="empty-copy">请先到“采集上传”页面拍照采集</div>';
    return;
  }
  for (const record of records) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `capture-choice-item${selectedCaptureRecord?.id === record.id ? " active" : ""}`;
    item.innerHTML = `<img src="${record.url}" alt="采集记录缩略图"><span>${record.time}</span>`;
    item.addEventListener("click", () => selectCaptureAndInfer(record));
    captureList.appendChild(item);
  }
}

function renderResultSummary(result) {
  const detections = Array.isArray(result?.detections) ? result.detections : [];
  const classifications = Array.isArray(result?.classifications) ? result.classifications : [];
  const taskType = result?.task_type || "--";

  if (!detections.length && !classifications.length) {
    resultBrief.textContent = `${taskType} / 0 个目标`;
    renderEmptySummary("当前结果未检测到目标");
    return;
  }

  resultBrief.textContent = `${taskType} / ${detections.length || classifications.length} 项结果`;
  targetSummary.innerHTML = "";

  if (detections.length) {
    detections.forEach((item, index) => {
      const bbox = Array.isArray(item.bbox_xyxy) ? item.bbox_xyxy.map((value) => formatNumber(value, 1)).join(", ") : "--";
      const center = Array.isArray(item.center_xy) ? item.center_xy.map((value) => formatNumber(value, 1)).join(", ") : "--";
      const obbPoints = Array.isArray(item.obb?.points) ? item.obb.points.length : 0;
      const mask = item.mask?.encoding || "--";
      const card = document.createElement("article");
      card.className = `target-item${index > 0 ? " secondary" : ""}`;
      card.innerHTML = `
        <div class="target-item-head">
          <b>${item.class_name || `目标 ${index + 1}`}</b>
          <span>${formatNumber((item.score || 0) * 100, 1)}%</span>
        </div>
        <div class="target-item-meta">
          <div><span>类别 ID</span><b>${item.class_id ?? "--"}</b></div>
          <div><span>中心点</span><b>${center}</b></div>
          <div><span>BBox</span><b>${bbox}</b></div>
          <div><span>结果类型</span><b>${item.obb ? `OBB ${obbPoints} 点` : (item.mask ? `Mask ${mask}` : "Detection")}</b></div>
        </div>
      `;
      targetSummary.appendChild(card);
    });
    return;
  }

  classifications.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = `target-item${index > 0 ? " secondary" : ""}`;
    card.innerHTML = `
      <div class="target-item-head">
        <b>${item.class_name || `分类 ${index + 1}`}</b>
        <span>${formatNumber((item.score || 0) * 100, 1)}%</span>
      </div>
      <div class="target-item-meta">
        <div><span>类别 ID</span><b>${item.class_id ?? "--"}</b></div>
        <div><span>标签</span><b>${item.label || item.class_name || "--"}</b></div>
        <div><span>任务类型</span><b>${taskType}</b></div>
        <div><span>状态</span><b>${result?.status || "ok"}</b></div>
      </div>
    `;
    targetSummary.appendChild(card);
  });
}

function showResult(result) {
  currentResult = result;
  updateState({ latestResult: result });
  const timing = result.timing || {};
  for (const key of ["preprocess", "inference", "postprocess", "total"]) {
    document.getElementById(`timing-${key}`).textContent = timing[`${key}_ms`] == null ? "--" : `${timing[`${key}_ms`]} ms`;
  }
  renderResultSummary(result);
}

function showRoiModal() {
  roiModal?.classList.add("active");
  roiModal?.setAttribute("aria-hidden", "false");
}

function hideRoiModal() {
  roiModal?.classList.remove("active");
  roiModal?.setAttribute("aria-hidden", "true");
}

function normalizeRoi(payload) {
  const roi = payload?.roi || payload || {};
  const values = Array.isArray(roi.normalized_xyxy)
    ? roi.normalized_xyxy.map(Number)
    : [Number(roi.x1), Number(roi.y1), Number(roi.x2), Number(roi.y2)];
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    return { enabled: false, x1: 0, y1: 0, x2: 1, y2: 1 };
  }
  return {
    enabled: roi.enabled === true,
    x1: Math.max(0, Math.min(1, values[0])),
    y1: Math.max(0, Math.min(1, values[1])),
    x2: Math.max(0, Math.min(1, values[2])),
    y2: Math.max(0, Math.min(1, values[3])),
  };
}

function updateRoiButton(roi) {
  const button = document.getElementById("validate-roi");
  if (!button) return;
  button.classList.toggle("active", roi?.enabled === true);
  button.textContent = roi?.enabled === true ? "ROI 已启用" : "绘制 ROI";
}

function sizeRoiCanvas() {
  if (!roiImage?.naturalWidth || !roiImage?.naturalHeight || !roiCanvas) return false;
  const rect = roiImage.getBoundingClientRect();
  roiCanvas.width = Math.max(1, Math.round(rect.width));
  roiCanvas.height = Math.max(1, Math.round(rect.height));
  roiCanvas.style.left = `${roiImage.offsetLeft}px`;
  roiCanvas.style.top = `${roiImage.offsetTop}px`;
  roiCanvas.style.width = `${rect.width}px`;
  roiCanvas.style.height = `${rect.height}px`;
  return true;
}

function renderRoiEditor() {
  if (!sizeRoiCanvas()) return;
  const ctx = roiCanvas.getContext("2d");
  const width = roiCanvas.width;
  const height = roiCanvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!roiDraft.enabled) {
    if (roiCoordinates) roiCoordinates.textContent = "未设置，使用全画面输出";
    return;
  }
  const x1 = roiDraft.x1 * width;
  const y1 = roiDraft.y1 * height;
  const x2 = roiDraft.x2 * width;
  const y2 = roiDraft.y2 * height;
  ctx.fillStyle = "rgba(2, 6, 23, .42)";
  ctx.fillRect(0, 0, width, y1);
  ctx.fillRect(0, y2, width, height - y2);
  ctx.fillRect(0, y1, x1, y2 - y1);
  ctx.fillRect(x2, y1, width - x2, y2 - y1);
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = Math.max(2, Math.round(width / 400));
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(250, 204, 21, .95)";
  ctx.fillRect(x1, Math.max(0, y1 - 28), 54, 26);
  ctx.fillStyle = "#111827";
  ctx.font = "bold 15px system-ui";
  ctx.fillText("ROI", x1 + 10, Math.max(18, y1 - 9));
  if (roiCoordinates) {
    const px = [
      Math.round(roiDraft.x1 * roiImage.naturalWidth),
      Math.round(roiDraft.y1 * roiImage.naturalHeight),
      Math.round(roiDraft.x2 * roiImage.naturalWidth),
      Math.round(roiDraft.y2 * roiImage.naturalHeight),
    ];
    roiCoordinates.textContent = `像素 [${px.join(", ")}]；归一化 [${roiDraft.x1.toFixed(4)}, ${roiDraft.y1.toFixed(4)}, ${roiDraft.x2.toFixed(4)}, ${roiDraft.y2.toFixed(4)}]`;
  }
}

function roiPointerPosition(event) {
  const rect = roiCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1))),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(rect.height, 1))),
  };
}

function beginRoiDraw(event) {
  if (!roiCanvas.width || !roiCanvas.height) return;
  roiDrawing = true;
  roiStartPoint = roiPointerPosition(event);
  roiDraft = { enabled: true, x1: roiStartPoint.x, y1: roiStartPoint.y, x2: roiStartPoint.x, y2: roiStartPoint.y };
  roiCanvas.setPointerCapture?.(event.pointerId);
  event.preventDefault();
}

function updateRoiDraw(event) {
  if (!roiDrawing || !roiStartPoint) return;
  const current = roiPointerPosition(event);
  roiDraft = {
    enabled: true,
    x1: Math.min(roiStartPoint.x, current.x),
    y1: Math.min(roiStartPoint.y, current.y),
    x2: Math.max(roiStartPoint.x, current.x),
    y2: Math.max(roiStartPoint.y, current.y),
  };
  renderRoiEditor();
}

function endRoiDraw(event) {
  if (!roiDrawing) return;
  roiDrawing = false;
  roiCanvas.releasePointerCapture?.(event.pointerId);
  if ((roiDraft.x2 - roiDraft.x1) < 0.005 || (roiDraft.y2 - roiDraft.y1) < 0.005) {
    roiDraft.enabled = false;
  }
  renderRoiEditor();
}

async function openRoiEditor() {
  stopRealtimeLoop();
  showRoiModal();
  roiEmpty?.classList.remove("hidden");
  if (roiEmpty) roiEmpty.textContent = "正在读取当前相机图像";
  try {
    const [config, blob] = await Promise.all([
      requestJson(endpoints.runtimeRoi),
      requestBlob(`${endpoints.snapshot}?t=${Date.now()}`),
    ]);
    roiDraft = normalizeRoi(config);
    updateRoiButton(roiDraft);
    if (roiSnapshotUrl) URL.revokeObjectURL(roiSnapshotUrl);
    roiSnapshotUrl = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      roiImage.onload = resolve;
      roiImage.onerror = reject;
      roiImage.src = roiSnapshotUrl;
    });
    roiEmpty?.classList.add("hidden");
    renderRoiEditor();
  } catch (error) {
    if (roiEmpty) {
      roiEmpty.textContent = error.body?.error?.message || error.message || "读取 ROI 编辑图像失败";
      roiEmpty.classList.remove("hidden");
    }
  }
}

async function saveRoi() {
  if (!roiDraft.enabled) {
    if (roiCoordinates) roiCoordinates.textContent = "请先在图像中拖动绘制 ROI。";
    return;
  }
  try {
    const payload = await postJson(endpoints.runtimeRoi, roiDraft);
    const saved = normalizeRoi(payload);
    roiDraft = saved;
    updateRoiButton(saved);
    if (currentResult) {
      currentResult = { ...currentResult, roi: payload.roi || saved };
      drawInferenceOverlay(canvas, image, currentResult);
    }
    hideRoiModal();
  } catch (error) {
    if (roiCoordinates) roiCoordinates.textContent = error.body?.error?.message || error.message || "保存 ROI 失败";
  }
}

async function disableRoi() {
  try {
    const payload = await postJson(endpoints.runtimeRoi, {
      enabled: false,
      x1: roiDraft.x1,
      y1: roiDraft.y1,
      x2: roiDraft.x2,
      y2: roiDraft.y2,
    });
    roiDraft = normalizeRoi(payload);
    updateRoiButton(roiDraft);
    if (currentResult) {
      currentResult = { ...currentResult, roi: payload.roi || roiDraft };
      drawInferenceOverlay(canvas, image, currentResult);
    }
    hideRoiModal();
  } catch (error) {
    if (roiCoordinates) roiCoordinates.textContent = error.body?.error?.message || error.message || "关闭 ROI 失败";
  }
}

async function refreshRoiState() {
  try {
    const payload = await requestJson(endpoints.runtimeRoi);
    updateRoiButton(normalizeRoi(payload));
  } catch (_error) {
    updateRoiButton({ enabled: false });
  }
}

function renderModelList() {
  const models = [...(currentCatalog?.models || [])].sort((left, right) => Number(right.mtime_ms || 0) - Number(left.mtime_ms || 0));
  if (!models.length) {
    modelList.innerHTML = '<div class="empty-copy">models_root 下暂无可识别的模型包</div>';
    return;
  }
  modelList.innerHTML = "";
  for (const model of models) {
    const card = document.createElement("article");
    card.className = `model-list-card${model.active ? " active" : ""}${model.valid ? "" : " invalid"}${switchingModelId === model.model_id ? " switching" : ""}`;
    const statusText = model.active ? "当前使用中" : (model.valid ? "点击切换" : "模型包无效");
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
        <div><span>任务</span><b>${model.task_type || "--"}</b></div>
        <div><span>平台</span><b>${model.target_platform || "--"}</b></div>
        <div><span>输入</span><b>${Array.isArray(model.input_size) ? model.input_size.join("x") : "--"}</b></div>
        <div><span>类别</span><b>${model.labels_count ?? "--"}</b></div>
        <div><span>大小</span><b>${formatBytes(model.rknn_size_bytes)}</b></div>
        <div><span>ID</span><b>${model.model_id || "--"}</b></div>
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
    if (!model.active && model.valid) {
      const hint = document.createElement("span");
      hint.className = "status-pill soft";
      hint.textContent = switchingModelId === model.model_id ? "切换中..." : "点击卡片切换";
      footer.appendChild(hint);
    }
    if (!switchDisabled) {
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `切换到模型 ${model.model_name || model.package_dir}`);
      card.addEventListener("click", () => switchModel(model.model_id));
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          switchModel(model.model_id);
        }
      });
    }
    modelList.appendChild(card);
  }
}

async function displayImageSource(sourceUrl) {
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = sourceUrl;
  });
  empty.classList.add("hidden");
  drawInferenceOverlay(canvas, image, currentResult);
}

async function refreshImage() {
  if (selectedCaptureRecord?.url) {
    await displayImageSource(selectedCaptureRecord.url);
    return;
  }
  const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
  if (snapshotUrl) URL.revokeObjectURL(snapshotUrl);
  snapshotUrl = URL.createObjectURL(blob);
  await displayImageSource(snapshotUrl);
}

export async function inferOnce() {
  try {
    const result = await postJson(endpoints.inferOnce);
    showResult(result);
    await refreshImage();
    return result;
  } catch (error) {
    empty.classList.remove("hidden");
    empty.textContent = "Runtime infer_once unavailable";
    clearOverlay(canvas);
    resultBrief.textContent = "检测失败";
    renderEmptySummary(error.body?.error?.message || error.message || "Runtime infer_once unavailable");
    return null;
  }
}

async function selectCaptureAndInfer(record) {
  if (realtimeTimer) {
    stopRealtimeLoop();
  }
  selectedCaptureRecord = record;
  hidePicker();
  await inferOnce();
  renderCapturePicker();
}

export async function refreshLatestResult() {
  try {
    const result = await requestJson(endpoints.latestResult);
    showResult(result);
    await refreshImage();
  } catch (error) {
    resultBrief.textContent = "读取失败";
    renderEmptySummary(error.body?.error?.message || error.message || "读取最新结果失败");
  }
}

export async function refreshRuntimeStatus() {
  try {
    const status = await requestJson(endpoints.runtimeStatus);
    const model = status.loaded_model || {};
    const displayName = [model.model_name || "--", model.model_version || "--", model.task_type || "--"].join(" / ");
    resultBrief.textContent = currentResult ? resultBrief.textContent : displayName;
  } catch (_error) {
    // 验证页不再单独展示 Runtime JSON，失败时不阻塞页面。
  }
}

export async function refreshModelCatalog() {
  try {
    const catalog = await requestJson(endpoints.models);
    currentCatalog = catalog;
    renderModelList();
    const modelsRoot = catalog.models_root || "--";
    const count = Array.isArray(catalog.models) ? catalog.models.length : 0;
    const active = catalog.models?.find((item) => item.active);
    catalogMessage(`已扫描 ${count} 个模型，当前 ${active?.model_name || "未识别"}，models_root=${modelsRoot}`, "ok");
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
    await postJson(endpoints.switchModel, { model_id: modelId });
    catalogMessage(`模型切换成功: ${modelId}`, "ok");
    await refreshRuntimeStatus();
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
  if (realtimeEnabled) {
    stopRealtimeLoop();
    return;
  }
  selectedCaptureRecord = null;
  hidePicker();
  setRealtimeButtonState(true);
  realtimeEnabled = true;
  const runLoop = async () => {
    if (!realtimeEnabled) return;
    if (realtimeBusy) {
      realtimeTimer = setTimeout(runLoop, getState().config.inference_interval_ms);
      return;
    }
    realtimeBusy = true;
    try {
      await inferOnce();
    } finally {
      realtimeBusy = false;
      if (realtimeEnabled) {
        realtimeTimer = setTimeout(runLoop, getState().config.inference_interval_ms);
      }
    }
  };
  runLoop();
}

export function initValidate() {
  document.getElementById("validate-infer").addEventListener("click", () => {
    stopRealtimeLoop();
    renderCapturePicker();
    capturePickerPanel.classList.toggle("hidden");
  });
  document.getElementById("validate-photo").addEventListener("click", async () => {
    selectedCaptureRecord = null;
    hidePicker();
    stopRealtimeLoop();
    await inferOnce();
  });
  document.getElementById("validate-realtime").addEventListener("click", toggleRealtime);
  document.getElementById("validate-roi")?.addEventListener("click", openRoiEditor);
  document.getElementById("validate-roi-close")?.addEventListener("click", hideRoiModal);
  document.getElementById("validate-roi-cancel")?.addEventListener("click", hideRoiModal);
  document.getElementById("validate-roi-save")?.addEventListener("click", saveRoi);
  document.getElementById("validate-roi-disable")?.addEventListener("click", disableRoi);
  roiCanvas?.addEventListener("pointerdown", beginRoiDraw);
  roiCanvas?.addEventListener("pointermove", updateRoiDraw);
  roiCanvas?.addEventListener("pointerup", endRoiDraw);
  roiCanvas?.addEventListener("pointercancel", endRoiDraw);
  document.getElementById("validate-model-scan").addEventListener("click", refreshModelCatalog);
  window.addEventListener("resize", () => {
    if (currentResult) drawInferenceOverlay(canvas, image, currentResult);
    if (roiModal?.classList.contains("active")) renderRoiEditor();
  });
  window.addEventListener("visionops:settings-saved", () => currentResult && drawInferenceOverlay(canvas, image, currentResult));
  renderEmptySummary("执行检测后显示目标摘要");
  setRealtimeButtonState(false);
  renderCapturePicker();
  refreshRuntimeStatus();
  refreshModelCatalog();
  refreshRoiState();
}
