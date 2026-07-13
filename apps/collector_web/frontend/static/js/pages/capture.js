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
  for (const id of ["capture-shoot-btn", "capture-refresh-list", "capture-prev-page", "capture-next-page", "capture-upload-server", "capture-upload-confirm"]) {
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
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `visionops-snapshot-${Date.now()}.jpg`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function shoot() {
  setBusy(true);
  try {
    const payload = await postJson(endpoints.datasetCapture, {});
    const item = payload.image;
    setMessage(`已保存到边缘端：${item?.filename || "采集图片"}`, "ok");
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

  loadRecords(0);
  refreshTimedStatus();
  if (!timedStatusTimer) timedStatusTimer = window.setInterval(refreshTimedStatus, 2000);
}
