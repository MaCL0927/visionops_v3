import { endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { getState, updateState } from "../state.js";

let currentBlob = null;
let currentUrl = null;
let records = [];
let offset = 0;
const limit = 24;
let total = 0;
let busy = false;
let previewRecord = null;
let timedStatusTimer = null;
let lastTimedCaptureCount = -1;
let timedCaptureEnabled = false;
let timedCaptureStateKnown = false;
let timedToggleBusy = false;
let latestTimedStatus = null;
let captureRoi = { enabled: false, x1: 0, y1: 0, x2: 1, y2: 1, sourceWidth: 0, sourceHeight: 0 };
let captureRoiDraft = { ...captureRoi };
let captureRoiSnapshotUrl = null;
let captureRoiDrawing = false;
let captureRoiStartPoint = null;
let captureRoiResizeObserver = null;

const image = document.getElementById("capture-image");
const empty = document.getElementById("capture-empty");
const message = document.getElementById("capture-message");

const previewModal = document.getElementById("capture-preview-modal");
const previewImage = document.getElementById("capture-preview-image");
const previewMeta = document.getElementById("capture-preview-meta");
const uploadModal = document.getElementById("capture-upload-modal");
const resultModal = document.getElementById("capture-result-modal");
const timedModal = document.getElementById("capture-timed-modal");
const timedStatusNode = document.getElementById("capture-timed-status");
const timedIntervalInput = document.getElementById("capture-timed-interval");
const captureRoiButton = document.getElementById("capture-roi-btn");
const captureRoiOverlay = document.getElementById("capture-roi-overlay");
const captureStage = document.querySelector(".capture-stage");
const captureRoiModal = document.getElementById("capture-roi-modal");
const captureRoiImage = document.getElementById("capture-roi-image");
const captureRoiCanvas = document.getElementById("capture-roi-canvas");
const captureRoiStage = document.querySelector(".capture-roi-editor-stage");
const captureRoiEmpty = document.getElementById("capture-roi-empty");
const captureRoiCoordinates = document.getElementById("capture-roi-coordinates");

function formatBytes(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function setMessage(text, kind = "") {
  if (!message) return;
  message.textContent = text;
  message.dataset.kind = kind;
}

function updateStatusCards(extra = {}) {
  const countNode = document.getElementById("capture-count");
  const exportNode = document.getElementById("capture-export-status");
  const uploadNode = document.getElementById("capture-upload-status");
  if (countNode) countNode.textContent = String(total || records.length);
  if (exportNode && extra.exportText) exportNode.textContent = extra.exportText;
  if (uploadNode && extra.uploadText) uploadNode.textContent = extra.uploadText;
}

function setBusy(nextBusy) {
  busy = nextBusy;
  for (const id of ["capture-shoot-btn", "capture-roi-btn", "capture-refresh-list", "capture-prev-page", "capture-next-page", "capture-upload-server", "capture-upload-confirm"]) {
    const node = document.getElementById(id);
    if (node) node.disabled = busy;
  }
}

function toCaptureRecord(imageRecord) {
  return {
    id: imageRecord.id || imageRecord.filename,
    filename: imageRecord.filename,
    url: imageRecord.url,
    time: imageRecord.mtime_text || "--",
    size_bytes: imageRecord.size_bytes,
    server_saved: true,
  };
}

function syncSharedRecords() {
  updateState({ captureRecords: records.map(toCaptureRecord) });
}

function showModal(modal) {
  if (!modal) return;
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
}

function hideModal(modal) {
  if (!modal) return;
  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
}

function normalizeCaptureRoi(payload) {
  const roi = payload?.capture_roi || payload || {};
  const values = Array.isArray(roi.normalized_xyxy)
    ? roi.normalized_xyxy.map(Number)
    : [Number(roi.x1), Number(roi.y1), Number(roi.x2), Number(roi.y2)];
  const source = roi.source_resolution || {};
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    return { enabled: false, x1: 0, y1: 0, x2: 1, y2: 1, sourceWidth: 0, sourceHeight: 0 };
  }
  return {
    enabled: roi.enabled === true,
    x1: Math.max(0, Math.min(1, values[0])),
    y1: Math.max(0, Math.min(1, values[1])),
    x2: Math.max(0, Math.min(1, values[2])),
    y2: Math.max(0, Math.min(1, values[3])),
    sourceWidth: Number(source.width || 0),
    sourceHeight: Number(source.height || 0),
    imageCount: Number(payload?.image_count || 0),
    batchLocked: payload?.batch_locked === true,
  };
}

function captureRoiPayload(roi, extra = {}) {
  return {
    enabled: roi.enabled === true,
    normalized_xyxy: [roi.x1, roi.y1, roi.x2, roi.y2],
    source_resolution: {
      width: Number(roi.sourceWidth || captureRoiImage?.naturalWidth || image?.naturalWidth || 0),
      height: Number(roi.sourceHeight || captureRoiImage?.naturalHeight || image?.naturalHeight || 0),
    },
    ...extra,
  };
}

function updateCaptureRoiButton() {
  if (!captureRoiButton) return;
  captureRoiButton.classList.toggle("active", captureRoi.enabled === true);
  captureRoiButton.textContent = captureRoi.enabled ? "采集 ROI 已启用" : "采集 ROI";
  captureRoiButton.title = captureRoi.enabled
    ? "手动采集、定时采图和本地下载将只保存蓝色 ROI 区域"
    : "绘制只影响数据采集保存的 ROI；不会修改模型验证结果 ROI";
}

function sizeCanvasOverImage(canvas, targetImage, stage) {
  if (!canvas || !targetImage?.naturalWidth || !targetImage?.naturalHeight || !stage) return false;
  const stageRect = stage.getBoundingClientRect();
  const imageRect = targetImage.getBoundingClientRect();
  if (imageRect.width <= 0 || imageRect.height <= 0) return false;
  canvas.width = Math.max(1, Math.round(imageRect.width));
  canvas.height = Math.max(1, Math.round(imageRect.height));
  canvas.style.left = `${imageRect.left - stageRect.left}px`;
  canvas.style.top = `${imageRect.top - stageRect.top}px`;
  canvas.style.width = `${imageRect.width}px`;
  canvas.style.height = `${imageRect.height}px`;
  return true;
}

function drawCaptureRoiOverlay() {
  if (!sizeCanvasOverImage(captureRoiOverlay, image, captureStage)) return;
  const ctx = captureRoiOverlay.getContext("2d");
  const width = captureRoiOverlay.width;
  const height = captureRoiOverlay.height;
  ctx.clearRect(0, 0, width, height);
  if (!captureRoi.enabled) return;
  const x1 = captureRoi.x1 * width;
  const y1 = captureRoi.y1 * height;
  const x2 = captureRoi.x2 * width;
  const y2 = captureRoi.y2 * height;
  ctx.fillStyle = "rgba(2, 6, 23, .18)";
  ctx.fillRect(0, 0, width, y1);
  ctx.fillRect(0, y2, width, height - y2);
  ctx.fillRect(0, y1, x1, y2 - y1);
  ctx.fillRect(x2, y1, width - x2, y2 - y1);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = Math.max(2, Math.round(width / 500));
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(14, 165, 233, .95)";
  ctx.fillRect(x1, Math.max(0, y1 - 28), 92, 26);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 14px system-ui";
  ctx.fillText("采集 ROI", x1 + 9, Math.max(18, y1 - 9));
}

function sizeCaptureRoiEditorCanvas() {
  if (!captureRoiImage?.naturalWidth || !captureRoiImage?.naturalHeight || !captureRoiCanvas || !captureRoiStage) return false;
  const stageWidth = Math.max(1, captureRoiStage.clientWidth);
  const stageHeight = Math.max(1, captureRoiStage.clientHeight);
  const scale = Math.min(stageWidth / captureRoiImage.naturalWidth, stageHeight / captureRoiImage.naturalHeight);
  const displayWidth = Math.max(1, Math.floor(captureRoiImage.naturalWidth * scale));
  const displayHeight = Math.max(1, Math.floor(captureRoiImage.naturalHeight * scale));
  captureRoiImage.style.width = `${displayWidth}px`;
  captureRoiImage.style.height = `${displayHeight}px`;
  return sizeCanvasOverImage(captureRoiCanvas, captureRoiImage, captureRoiStage);
}

function renderCaptureRoiEditor() {
  if (!sizeCaptureRoiEditorCanvas()) return;
  const ctx = captureRoiCanvas.getContext("2d");
  const width = captureRoiCanvas.width;
  const height = captureRoiCanvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!captureRoiDraft.enabled) {
    if (captureRoiCoordinates) captureRoiCoordinates.textContent = "未设置，采集时保存完整画面";
    return;
  }
  const x1 = captureRoiDraft.x1 * width;
  const y1 = captureRoiDraft.y1 * height;
  const x2 = captureRoiDraft.x2 * width;
  const y2 = captureRoiDraft.y2 * height;
  ctx.fillStyle = "rgba(2, 6, 23, .45)";
  ctx.fillRect(0, 0, width, y1);
  ctx.fillRect(0, y2, width, height - y2);
  ctx.fillRect(0, y1, x1, y2 - y1);
  ctx.fillRect(x2, y1, width - x2, y2 - y1);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = Math.max(2, Math.round(width / 400));
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(14, 165, 233, .95)";
  ctx.fillRect(x1, Math.max(0, y1 - 30), 98, 28);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 15px system-ui";
  ctx.fillText("采集 ROI", x1 + 9, Math.max(19, y1 - 10));
  if (captureRoiCoordinates) {
    const naturalWidth = captureRoiImage.naturalWidth;
    const naturalHeight = captureRoiImage.naturalHeight;
    const px = [
      Math.round(captureRoiDraft.x1 * naturalWidth),
      Math.round(captureRoiDraft.y1 * naturalHeight),
      Math.round(captureRoiDraft.x2 * naturalWidth),
      Math.round(captureRoiDraft.y2 * naturalHeight),
    ];
    const cropWidth = Math.max(0, px[2] - px[0]);
    const cropHeight = Math.max(0, px[3] - px[1]);
    captureRoiCoordinates.textContent = `像素 [${px.join(", ")}]，裁剪尺寸 ${cropWidth}×${cropHeight}；归一化 [${captureRoiDraft.x1.toFixed(4)}, ${captureRoiDraft.y1.toFixed(4)}, ${captureRoiDraft.x2.toFixed(4)}, ${captureRoiDraft.y2.toFixed(4)}]`;
  }
}

function captureRoiPointerPosition(event) {
  const rect = captureRoiCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1))),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(rect.height, 1))),
  };
}

function beginCaptureRoiDraw(event) {
  if (!captureRoiCanvas.width || !captureRoiCanvas.height) return;
  captureRoiDrawing = true;
  captureRoiStartPoint = captureRoiPointerPosition(event);
  captureRoiDraft = {
    enabled: true,
    x1: captureRoiStartPoint.x,
    y1: captureRoiStartPoint.y,
    x2: captureRoiStartPoint.x,
    y2: captureRoiStartPoint.y,
    sourceWidth: captureRoiImage.naturalWidth,
    sourceHeight: captureRoiImage.naturalHeight,
  };
  captureRoiCanvas.setPointerCapture?.(event.pointerId);
  event.preventDefault();
}

function updateCaptureRoiDraw(event) {
  if (!captureRoiDrawing || !captureRoiStartPoint) return;
  const current = captureRoiPointerPosition(event);
  captureRoiDraft = {
    ...captureRoiDraft,
    enabled: true,
    x1: Math.min(captureRoiStartPoint.x, current.x),
    y1: Math.min(captureRoiStartPoint.y, current.y),
    x2: Math.max(captureRoiStartPoint.x, current.x),
    y2: Math.max(captureRoiStartPoint.y, current.y),
  };
  renderCaptureRoiEditor();
}

function endCaptureRoiDraw(event) {
  if (!captureRoiDrawing) return;
  captureRoiDrawing = false;
  captureRoiCanvas.releasePointerCapture?.(event.pointerId);
  if ((captureRoiDraft.x2 - captureRoiDraft.x1) < 0.01 || (captureRoiDraft.y2 - captureRoiDraft.y1) < 0.01) {
    captureRoiDraft.enabled = false;
  }
  renderCaptureRoiEditor();
}

async function refreshCaptureRoiState() {
  try {
    const payload = await requestJson(endpoints.captureRoi);
    captureRoi = normalizeCaptureRoi(payload);
    updateCaptureRoiButton();
    drawCaptureRoiOverlay();
    return payload;
  } catch (error) {
    setMessage(error.body?.error?.message || error.message || "读取采集 ROI 失败", "error");
    return null;
  }
}

async function openCaptureRoiEditor() {
  showModal(captureRoiModal);
  captureRoiEmpty?.classList.remove("hidden");
  if (captureRoiEmpty) captureRoiEmpty.textContent = "正在读取当前相机图像";
  try {
    const [configPayload, blob] = await Promise.all([
      requestJson(endpoints.captureRoi),
      requestBlob(`${endpoints.snapshot}?t=${Date.now()}`),
    ]);
    captureRoi = normalizeCaptureRoi(configPayload);
    captureRoiDraft = { ...captureRoi };
    updateCaptureRoiButton();
    if (captureRoiSnapshotUrl) URL.revokeObjectURL(captureRoiSnapshotUrl);
    captureRoiSnapshotUrl = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      captureRoiImage.onload = resolve;
      captureRoiImage.onerror = reject;
      captureRoiImage.src = captureRoiSnapshotUrl;
    });
    captureRoiDraft.sourceWidth = captureRoiImage.naturalWidth;
    captureRoiDraft.sourceHeight = captureRoiImage.naturalHeight;
    captureRoiEmpty?.classList.add("hidden");
    window.requestAnimationFrame(() => window.requestAnimationFrame(renderCaptureRoiEditor));
  } catch (error) {
    if (captureRoiEmpty) {
      captureRoiEmpty.textContent = error.body?.error?.message || error.message || "读取采集 ROI 编辑图像失败";
      captureRoiEmpty.classList.remove("hidden");
    }
  }
}

async function postCaptureRoiWithBatchGuard(payload) {
  try {
    return await postJson(endpoints.captureRoi, payload);
  } catch (error) {
    if (error.status !== 409 || error.body?.error?.code !== "CAPTURE_ROI_BATCH_LOCKED") throw error;
    const confirmed = window.confirm(`${error.body?.error?.message || "已有采集图片，无法直接切换 ROI。"}\n\n确认后将删除当前采集目录中的全部图片，并应用新的采集 ROI。是否继续？`);
    if (!confirmed) return null;
    return postJson(endpoints.captureRoi, { ...payload, clear_existing_images: true });
  }
}

async function saveCaptureRoi() {
  if (!captureRoiDraft.enabled) {
    if (captureRoiCoordinates) captureRoiCoordinates.textContent = "请先在图像中拖动绘制采集 ROI。";
    return;
  }
  captureRoiDraft.sourceWidth = captureRoiImage.naturalWidth;
  captureRoiDraft.sourceHeight = captureRoiImage.naturalHeight;
  try {
    const result = await postCaptureRoiWithBatchGuard(captureRoiPayload(captureRoiDraft));
    if (!result) return;
    captureRoi = normalizeCaptureRoi(result);
    updateCaptureRoiButton();
    drawCaptureRoiOverlay();
    hideModal(captureRoiModal);
    if (Number(result.deleted_image_count || 0) > 0) await loadRecords(0);
    setMessage(`采集 ROI 已启用：${result.capture_roi?.crop_resolution?.width || "--"}×${result.capture_roi?.crop_resolution?.height || "--"}`, "ok");
  } catch (error) {
    if (captureRoiCoordinates) captureRoiCoordinates.textContent = error.body?.error?.message || error.message || "保存采集 ROI 失败";
  }
}

async function disableCaptureRoi() {
  const sourceWidth = captureRoiImage?.naturalWidth || image?.naturalWidth || captureRoi.sourceWidth || 0;
  const sourceHeight = captureRoiImage?.naturalHeight || image?.naturalHeight || captureRoi.sourceHeight || 0;
  try {
    const result = await postCaptureRoiWithBatchGuard({
      enabled: false,
      normalized_xyxy: [0, 0, 1, 1],
      source_resolution: { width: sourceWidth, height: sourceHeight },
    });
    if (!result) return;
    captureRoi = normalizeCaptureRoi(result);
    captureRoiDraft = { ...captureRoi };
    updateCaptureRoiButton();
    drawCaptureRoiOverlay();
    hideModal(captureRoiModal);
    if (Number(result.deleted_image_count || 0) > 0) await loadRecords(0);
    setMessage("采集 ROI 已关闭，后续采集将保存完整画面", "ok");
  } catch (error) {
    if (captureRoiCoordinates) captureRoiCoordinates.textContent = error.body?.error?.message || error.message || "关闭采集 ROI 失败";
  }
}

async function cropBlobForDownload(blob) {
  if (!captureRoi.enabled) return blob;
  const bitmap = await createImageBitmap(blob);
  try {
    const x1 = Math.max(0, Math.min(bitmap.width - 1, Math.round(captureRoi.x1 * bitmap.width)));
    const y1 = Math.max(0, Math.min(bitmap.height - 1, Math.round(captureRoi.y1 * bitmap.height)));
    const x2 = Math.max(x1 + 1, Math.min(bitmap.width, Math.round(captureRoi.x2 * bitmap.width)));
    const y2 = Math.max(y1 + 1, Math.min(bitmap.height, Math.round(captureRoi.y2 * bitmap.height)));
    const canvas = document.createElement("canvas");
    canvas.width = x2 - x1;
    canvas.height = y2 - y1;
    canvas.getContext("2d").drawImage(bitmap, x1, y1, x2 - x1, y2 - y1, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve, reject) => {
      canvas.toBlob((result) => result ? resolve(result) : reject(new Error("ROI 图片编码失败")), blob.type || "image/jpeg", 0.95);
    });
  } finally {
    bitmap.close?.();
  }
}

function renderRecords() {
  syncSharedRecords();
  updateStatusCards();
  const target = document.getElementById("capture-records");
  if (!target) return;
  target.replaceChildren();
  if (!records.length) {
    const emptyCopy = document.createElement("div");
    emptyCopy.className = "empty-copy";
    emptyCopy.textContent = "暂无采集图片，请先拍照采集";
    target.append(emptyCopy);
  } else {
    for (const record of records) {
      const card = document.createElement("article");
      const preview = document.createElement("img");
      const meta = document.createElement("div");
      const actions = document.createElement("div");
      const deleteBtn = document.createElement("button");
      const openBtn = document.createElement("button");
      card.className = "capture-record";
      preview.src = `${record.url}?thumb=1`;
      preview.loading = "lazy";
      preview.alt = record.filename || "采集图片";
      preview.addEventListener("click", () => previewSavedImage(record));
      meta.className = "capture-record-meta";
      meta.innerHTML = `<b>${record.filename || "采集图片"}</b><span>${record.mtime_text || "--"} · ${formatBytes(record.size_bytes)}</span>`;
      actions.className = "capture-record-actions";
      openBtn.type = "button";
      openBtn.textContent = "预览";
      openBtn.addEventListener("click", () => previewSavedImage(record));
      deleteBtn.type = "button";
      deleteBtn.className = "danger-soft";
      deleteBtn.textContent = "删除";
      deleteBtn.addEventListener("click", () => deleteSavedImage(record));
      actions.append(openBtn, deleteBtn);
      card.append(preview, meta, actions);
      target.append(card);
    }
  }
  const pageNode = document.getElementById("capture-page-info");
  if (pageNode) {
    const start = total === 0 ? 0 : offset + 1;
    const end = Math.min(total, offset + limit);
    pageNode.textContent = `${start}-${end} / ${total}`;
  }
  const prev = document.getElementById("capture-prev-page");
  const next = document.getElementById("capture-next-page");
  if (prev) prev.disabled = busy || offset <= 0;
  if (next) next.disabled = busy || offset + limit >= total;
}

async function loadRecords(nextOffset = offset) {
  try {
    const payload = await requestJson(`${endpoints.datasetImages}?offset=${nextOffset}&limit=${limit}`);
    records = Array.isArray(payload.images) ? payload.images : [];
    total = Number(payload.total || 0);
    offset = Number(payload.offset || 0);
    renderRecords();
    updateStatusCards({ exportText: "上传时自动打包", uploadText: "等待上传" });
  } catch (error) {
    setMessage(error.body?.error?.message || error.message || "读取采集图片失败", "error");
  }
}

function activateStep(kind) {
  document.querySelectorAll("[data-capture-step]").forEach((button) => button.classList.toggle("active", button.dataset.captureStep === kind));
  document.getElementById("capture-shoot")?.classList.toggle("active", kind === "shoot");
  document.getElementById("capture-upload")?.classList.toggle("active", kind === "upload");
  if (kind === "shoot") refreshCapture();
  loadRecords();
}

export async function refreshCapture() {
  try {
    currentBlob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(currentBlob);
    image.src = currentUrl;
    empty.classList.add("hidden");
    setMessage(`已刷新 ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    currentBlob = null;
    empty.classList.remove("hidden");
    empty.textContent = "Runtime snapshot unreachable";
    setMessage(error.body?.error?.message || error.message, "error");
  }
}

async function ensureBlob() {
  if (!currentBlob) await refreshCapture();
  return currentBlob;
}

async function downloadCapture() {
  const blob = await ensureBlob();
  if (!blob) return;
  try {
    const outputBlob = await cropBlobForDownload(blob);
    const url = URL.createObjectURL(outputBlob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `visionops-${captureRoi.enabled ? "roi-" : "snapshot-"}${Date.now()}.${outputBlob.type === "image/png" ? "png" : "jpg"}`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    setMessage(error.message || "本地下载 ROI 图片失败", "error");
  }
}

async function shoot() {
  setBusy(true);
  try {
    const payload = await postJson(endpoints.datasetCapture, {});
    const item = payload.image;
    const crop = payload.capture_roi?.crop_resolution;
    const suffix = payload.images_are_cropped ? `（ROI ${crop?.width || "--"}×${crop?.height || "--"}）` : "（完整画面）";
    setMessage(`已保存到边缘端：${item?.filename || "采集图片"}${suffix}`, "ok");
    await refreshCapture();
    await loadRecords(0);
  } catch (error) {
    setMessage(error.body?.error?.message || error.message || "保存采集图片失败", "error");
  } finally {
    setBusy(false);
    renderRecords();
  }
}

function previewSavedImage(record) {
  if (!record?.url) return;
  previewRecord = record;
  if (previewImage) previewImage.src = `${record.url}?t=${Date.now()}`;
  if (previewMeta) previewMeta.textContent = `${record.filename || "采集图片"} · ${record.mtime_text || "--"} · ${formatBytes(record.size_bytes)}`;
  showModal(previewModal);
}

async function deleteSavedImage(record, options = {}) {
  if (!record?.filename) return;
  if (!options.skipConfirm && !window.confirm(`确定删除图片 ${record.filename}？`)) return;
  setBusy(true);
  try {
    const response = await fetch(record.delete_url || `${endpoints.datasetImages}/${encodeURIComponent(record.filename)}`, { method: "DELETE", cache: "no-store" });
    if (!response.ok) {
      let body = null;
      try { body = await response.json(); } catch (_error) { /* ignore */ }
      throw new Error(body?.error?.message || `HTTP ${response.status}`);
    }
    setMessage(`已删除：${record.filename}`, "ok");
    if (previewRecord?.filename === record.filename) {
      hideModal(previewModal);
      previewRecord = null;
    }
    const nextOffset = Math.max(0, Math.min(offset, Math.max(0, total - 2)));
    await loadRecords(nextOffset);
  } catch (error) {
    setMessage(error.message || "删除失败", "error");
  } finally {
    setBusy(false);
    renderRecords();
  }
}

function formatTimestamp(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "--";
  return new Date(number).toLocaleString();
}

function renderTimedStatus(payload) {
  const enabled = payload?.enabled === true;
  const button = document.getElementById("capture-timed-btn");
  const count = Number(payload?.capture_count || 0);
  timedCaptureEnabled = enabled;
  timedCaptureStateKnown = true;
  latestTimedStatus = payload || {};
  if (button) {
    button.classList.toggle("active", enabled);
    button.setAttribute("aria-pressed", String(enabled));
    button.disabled = timedToggleBusy;
    button.title = enabled ? "点击立即停止定时采图" : "设置定时采图间隔";
    button.textContent = timedToggleBusy
      ? (enabled ? "正在停止定时采图…" : "正在处理…")
      : (enabled ? `停止定时采图 (${payload.interval_seconds}s)` : "定时采图");
  }
  if (timedIntervalInput && document.activeElement !== timedIntervalInput) {
    timedIntervalInput.value = String(payload?.interval_seconds ?? 10);
  }
  if (timedStatusNode) {
    if (enabled) {
      timedStatusNode.textContent = `运行中：已自动保存 ${count} 张；下次 ${formatTimestamp(payload.next_capture_at_ms)}；最近错误：${payload.last_error || "无"}`;
      timedStatusNode.dataset.kind = payload.last_error ? "error" : "ok";
    } else {
      timedStatusNode.textContent = `当前未启用。累计自动保存 ${count} 张；上次采图 ${formatTimestamp(payload.last_capture_at_ms)}。`;
      timedStatusNode.dataset.kind = "";
    }
  }
  if (lastTimedCaptureCount >= 0 && count > lastTimedCaptureCount) {
    setMessage(`定时采图已保存：${payload.last_image?.filename || `累计 ${count} 张`}`, "ok");
    loadRecords(0);
  }
  lastTimedCaptureCount = count;
}

async function refreshTimedStatus() {
  try {
    const payload = await requestJson(endpoints.timedCapture);
    renderTimedStatus(payload);
    return payload;
  } catch (error) {
    if (timedStatusNode) {
      timedStatusNode.textContent = error.body?.error?.message || error.message || "读取定时采图状态失败";
      timedStatusNode.dataset.kind = "error";
    }
    return null;
  }
}

async function openTimedCapture() {
  showModal(timedModal);
  await refreshTimedStatus();
}

async function toggleTimedCapture() {
  if (timedToggleBusy) return;
  if (!timedCaptureStateKnown) await refreshTimedStatus();
  if (timedCaptureEnabled) {
    await stopTimedCapture();
    return;
  }
  await openTimedCapture();
}

async function startTimedCapture() {
  const interval = Number(timedIntervalInput?.value || 10);
  if (!Number.isFinite(interval) || interval < 0.5 || interval > 86400) {
    timedStatusNode.textContent = "采图间隔必须位于 0.5 到 86400 秒。";
    timedStatusNode.dataset.kind = "error";
    return;
  }
  try {
    const payload = await postJson(endpoints.timedCapture, {
      enabled: true,
      interval_seconds: interval,
    });
    renderTimedStatus(payload);
    hideModal(timedModal);
    setMessage(`定时采图已启动，间隔 ${interval} 秒`, "ok");
  } catch (error) {
    timedStatusNode.textContent = error.body?.error?.message || error.message || "启动定时采图失败";
    timedStatusNode.dataset.kind = "error";
  }
}

async function stopTimedCapture() {
  timedToggleBusy = true;
  renderTimedStatus(latestTimedStatus || { enabled: true });
  try {
    const payload = await postJson(endpoints.timedCapture, { enabled: false });
    latestTimedStatus = payload;
    hideModal(timedModal);
    setMessage("定时采图已停止", "ok");
  } catch (error) {
    const messageText = error.body?.error?.message || error.message || "停止定时采图失败";
    if (timedStatusNode) {
      timedStatusNode.textContent = messageText;
      timedStatusNode.dataset.kind = "error";
    }
    setMessage(messageText, "error");
  } finally {
    timedToggleBusy = false;
    if (latestTimedStatus) renderTimedStatus(latestTimedStatus);
    else await refreshTimedStatus();
  }
}

function openUploadConfirm() {
  const state = getState();
  const config = state.config || {};
  const deviceInput = document.getElementById("upload-device-id");
  const customerInput = document.getElementById("upload-customer-id");
  const contactInput = document.getElementById("upload-contact-info");
  const remarkInput = document.getElementById("upload-remark");
  if (deviceInput) deviceInput.value = config.device_id || "rk3576-001";
  if (customerInput && !customerInput.value.trim()) customerInput.value = "CUST-001";
  if (contactInput) contactInput.value = contactInput.value || "";
  if (remarkInput) remarkInput.value = remarkInput.value || "";
  const msg = document.getElementById("capture-upload-confirm-message");
  if (msg) {
    msg.textContent = `当前本地图片 ${total} 张；确认后会先生成 tar.gz，再上传服务器。`;
    msg.dataset.kind = "";
  }
  showModal(uploadModal);
}

function collectUploadMetadata() {
  const deviceId = document.getElementById("upload-device-id")?.value.trim() || "";
  const customerId = document.getElementById("upload-customer-id")?.value.trim() || "";
  const contactInfo = document.getElementById("upload-contact-info")?.value.trim() || "";
  const remark = document.getElementById("upload-remark")?.value.trim() || "";
  if (!deviceId) throw new Error("设备 ID 为必填项");
  if (!customerId) throw new Error("客户 ID 为必填项");
  return { device_id: deviceId, customer_id: customerId, contact_info: contactInfo, remark };
}

function showUploadResult(payload, ok) {
  const title = document.getElementById("capture-result-title");
  const subtitle = document.getElementById("capture-result-subtitle");
  const body = document.getElementById("capture-result-body");
  const pack = payload?.package || {};
  if (title) title.textContent = ok ? "上传成功" : "上传失败";
  if (subtitle) subtitle.textContent = ok ? "采集包已上传到服务端。" : "本地压缩包已保留，可稍后重试或手动拷贝。";
  if (body) {
    const rows = [
      ["本地压缩包", pack.path || pack.filename || "--"],
      ["压缩包大小", formatBytes(pack.size_bytes)],
      ["远端路径", payload?.upload?.remote_path || "--"],
      ["图片数量", String(payload?.image_count ?? payload?.manifest?.counts?.all ?? "--")],
      ["耗时", payload?.elapsed_ms != null ? `${payload.elapsed_ms} ms` : "--"],
    ];
    if (!ok) rows.push(["失败原因", payload?.upload?.error || payload?.error?.message || payload?.message || "unknown"]);
    body.innerHTML = rows.map(([k, v]) => `<div><b>${k}</b><span>${v}</span></div>`).join("");
    body.dataset.kind = ok ? "ok" : "error";
  }
  showModal(resultModal);
}

async function confirmUpload() {
  let metadata;
  const confirmMessage = document.getElementById("capture-upload-confirm-message");
  try {
    metadata = collectUploadMetadata();
  } catch (error) {
    if (confirmMessage) {
      confirmMessage.textContent = error.message;
      confirmMessage.dataset.kind = "error";
    }
    return;
  }
  setBusy(true);
  updateStatusCards({ exportText: "打包中...", uploadText: "打包并上传中..." });
  if (confirmMessage) {
    confirmMessage.textContent = "正在打包并上传，请稍候...";
    confirmMessage.dataset.kind = "loading";
  }
  try {
    const payload = await postJson(endpoints.datasetUpload, metadata);
    const pack = payload.package || {};
    updateStatusCards({ exportText: `${pack.filename || "已生成"} · ${formatBytes(pack.size_bytes)}` });
    if (payload.upload_ok) {
      updateStatusCards({ uploadText: `上传成功：${payload.upload?.remote_path || "服务器"}` });
      setMessage(`上传成功：${pack.filename || "采集包"}`, "ok");
      hideModal(uploadModal);
      showUploadResult(payload, true);
    } else {
      updateStatusCards({ uploadText: "上传失败，压缩包已保留" });
      setMessage(`${payload.message || "上传失败"} 本地包：${pack.path || pack.filename || "--"}`, "error");
      hideModal(uploadModal);
      showUploadResult(payload, false);
    }
  } catch (error) {
    const payload = error.body || { error: { message: error.message } };
    updateStatusCards({ uploadText: "上传失败" });
    setMessage(payload.error?.message || error.message || "上传失败", "error");
    hideModal(uploadModal);
    showUploadResult(payload, false);
  } finally {
    setBusy(false);
    renderRecords();
  }
}

function clearRecords() {
  if (!window.confirm("该操作会删除当前页显示的采集图片，确定继续？")) return;
  Promise.all(records.map((record) => fetch(record.delete_url || `${endpoints.datasetImages}/${encodeURIComponent(record.filename)}`, { method: "DELETE", cache: "no-store" }).catch(() => null)))
    .then(() => loadRecords(0))
    .then(() => setMessage("已删除当前页采集图片", "ok"));
}

export function initCapture() {
  document.querySelectorAll("[data-capture-step]").forEach((button) => button.addEventListener("click", () => activateStep(button.dataset.captureStep)));
  document.getElementById("capture-refresh")?.addEventListener("click", refreshCapture);
  captureRoiButton?.addEventListener("click", openCaptureRoiEditor);
  document.getElementById("capture-shoot-btn")?.addEventListener("click", shoot);
  document.getElementById("capture-timed-btn")?.addEventListener("click", toggleTimedCapture);
  document.getElementById("capture-download")?.addEventListener("click", downloadCapture);
  document.getElementById("capture-clear")?.addEventListener("click", clearRecords);
  document.getElementById("capture-refresh-list")?.addEventListener("click", () => loadRecords(offset));
  document.getElementById("capture-prev-page")?.addEventListener("click", () => loadRecords(Math.max(0, offset - limit)));
  document.getElementById("capture-next-page")?.addEventListener("click", () => loadRecords(offset + limit));
  document.getElementById("capture-upload-server")?.addEventListener("click", openUploadConfirm);

  document.getElementById("capture-preview-close")?.addEventListener("click", () => hideModal(previewModal));
  document.getElementById("capture-preview-done")?.addEventListener("click", () => hideModal(previewModal));
  document.getElementById("capture-preview-delete")?.addEventListener("click", () => deleteSavedImage(previewRecord, { skipConfirm: false }));

  document.getElementById("capture-upload-close")?.addEventListener("click", () => hideModal(uploadModal));
  document.getElementById("capture-upload-cancel")?.addEventListener("click", () => hideModal(uploadModal));
  document.getElementById("capture-upload-confirm")?.addEventListener("click", confirmUpload);
  document.getElementById("capture-result-close")?.addEventListener("click", () => hideModal(resultModal));
  document.getElementById("capture-result-done")?.addEventListener("click", () => hideModal(resultModal));

  document.getElementById("capture-timed-close")?.addEventListener("click", () => hideModal(timedModal));
  document.getElementById("capture-timed-cancel")?.addEventListener("click", () => hideModal(timedModal));
  document.getElementById("capture-timed-confirm")?.addEventListener("click", startTimedCapture);

  document.getElementById("capture-roi-close")?.addEventListener("click", () => hideModal(captureRoiModal));
  document.getElementById("capture-roi-cancel")?.addEventListener("click", () => hideModal(captureRoiModal));
  document.getElementById("capture-roi-save")?.addEventListener("click", saveCaptureRoi);
  document.getElementById("capture-roi-disable")?.addEventListener("click", disableCaptureRoi);
  captureRoiCanvas?.addEventListener("pointerdown", beginCaptureRoiDraw);
  captureRoiCanvas?.addEventListener("pointermove", updateCaptureRoiDraw);
  captureRoiCanvas?.addEventListener("pointerup", endCaptureRoiDraw);
  captureRoiCanvas?.addEventListener("pointercancel", endCaptureRoiDraw);
  image?.addEventListener("load", drawCaptureRoiOverlay);
  window.addEventListener("resize", () => {
    drawCaptureRoiOverlay();
    if (captureRoiModal?.classList.contains("active")) window.requestAnimationFrame(renderCaptureRoiEditor);
  });
  if (captureRoiStage && "ResizeObserver" in window && !captureRoiResizeObserver) {
    captureRoiResizeObserver = new ResizeObserver(() => {
      if (captureRoiModal?.classList.contains("active")) window.requestAnimationFrame(renderCaptureRoiEditor);
    });
    captureRoiResizeObserver.observe(captureRoiStage);
  }

  window.addEventListener("visionops:camera-switched", () => {
    window.setTimeout(async () => {
      await refreshCaptureRoiState();
      await refreshCapture().catch(() => null);
    }, 1500);
  });
  loadRecords(0);
  refreshCaptureRoiState();
  refreshTimedStatus();
  if (!timedStatusTimer) timedStatusTimer = window.setInterval(refreshTimedStatus, 2000);
}
