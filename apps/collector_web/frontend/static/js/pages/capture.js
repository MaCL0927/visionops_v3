import { endpoints, requestBlob } from "../api.js";

let currentUrl = null;
const image = document.getElementById("capture-image");
const empty = document.getElementById("capture-empty");
const message = document.getElementById("capture-message");

export async function refreshCapture() {
  try {
    const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`);
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(blob); image.src = currentUrl; empty.classList.add("hidden");
    message.textContent = `快照已刷新：${new Date().toLocaleTimeString()}`;
  } catch (error) { empty.classList.remove("hidden"); empty.textContent = "Runtime snapshot unreachable"; message.textContent = error.body?.error?.message || error.message; }
}

async function downloadCapture() {
  try { const blob = await requestBlob(endpoints.snapshot); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `visionops-snapshot-${Date.now()}.jpg`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  catch (error) { message.textContent = error.body?.error?.message || error.message; }
}

export function initCapture() {
  document.getElementById("capture-refresh").addEventListener("click", refreshCapture);
  document.getElementById("capture-download").addEventListener("click", downloadCapture);
  document.getElementById("capture-export").title = "M7 仅预留入口，尚未实现真实上传服务器";
}
