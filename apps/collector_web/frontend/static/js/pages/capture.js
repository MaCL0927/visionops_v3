import { endpoints, requestBlob } from "../api.js";
import { updateState } from "../state.js";

let currentBlob = null, currentUrl = null;
const records = [];
const image = document.getElementById("capture-image"), empty = document.getElementById("capture-empty"), message = document.getElementById("capture-message");

function activateStep(kind) {
  document.querySelectorAll("[data-capture-step]").forEach((button) => button.classList.toggle("active", button.dataset.captureStep === kind));
  document.getElementById("capture-shoot").classList.toggle("active", kind === "shoot");
  document.getElementById("capture-upload").classList.toggle("active", kind === "upload");
  if (kind === "shoot") refreshCapture(); else renderRecords();
}

function renderRecords() {
  updateState({ captureRecords: [...records] });
  document.getElementById("capture-count").textContent = String(records.length);
  const target = document.getElementById("capture-records"); target.replaceChildren();
  if (!records.length) { const emptyCopy = document.createElement("div"); emptyCopy.className = "empty-copy"; emptyCopy.textContent = "暂无采集记录"; target.append(emptyCopy); return; }
  for (const record of records) {
    const card = document.createElement("article"), preview = document.createElement("img"), meta = document.createElement("div");
    card.className = "capture-record"; preview.src = record.url; preview.alt = "临时采集图片"; meta.textContent = record.time; card.append(preview, meta); target.append(card);
  }
}

export async function refreshCapture() {
  try {
    currentBlob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(currentBlob); image.src = currentUrl; empty.classList.add("hidden"); message.textContent = `已刷新 ${new Date().toLocaleTimeString()}`;
  } catch (error) { currentBlob = null; empty.classList.remove("hidden"); empty.textContent = "Runtime snapshot unreachable"; message.textContent = error.body?.error?.message || error.message; }
}

async function ensureBlob() { if (!currentBlob) await refreshCapture(); return currentBlob; }
async function downloadCapture() {
  const blob = await ensureBlob(); if (!blob) return;
  const url = URL.createObjectURL(blob), anchor = document.createElement("a"); anchor.href = url; anchor.download = `visionops-snapshot-${Date.now()}.jpg`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function shoot() {
  try {
    const blob = await requestBlob(`${endpoints.snapshot}?capture=${Date.now()}`);
    records.unshift({ id: `capture-${Date.now()}`, url: URL.createObjectURL(blob), time: new Date().toLocaleString() });
    message.textContent = `已加入临时采集记录，共 ${records.length} 张`; renderRecords();
  } catch (error) { message.textContent = error.body?.error?.message || error.message; }
}
function clearRecords() { records.forEach((record) => URL.revokeObjectURL(record.url)); records.length = 0; renderRecords(); }

export function initCapture() {
  document.querySelectorAll("[data-capture-step]").forEach((button) => button.addEventListener("click", () => activateStep(button.dataset.captureStep)));
  document.getElementById("capture-refresh").addEventListener("click", refreshCapture);
  document.getElementById("capture-shoot-btn").addEventListener("click", shoot);
  document.getElementById("capture-download").addEventListener("click", downloadCapture);
  document.getElementById("capture-clear").addEventListener("click", clearRecords);
  document.getElementById("capture-export").title = "采集包导出接口待接入";
  document.getElementById("capture-upload-server").title = "真实上传服务器待接入";
}
