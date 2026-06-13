import { getState, updateState } from "../state.js";

const fields = {
  runtime_url: "setting-runtime-url", gateway_url: "setting-gateway-url",
  business_app_url: "setting-business-app-url", device_id: "setting-device-id",
  snapshot_refresh_interval_ms: "setting-snapshot-interval",
  status_refresh_interval_ms: "setting-status-interval",
};

function fill(config) { for (const [key, id] of Object.entries(fields)) document.getElementById(id).value = config[key] ?? ""; }
function open() { fill(getState().config); const modal = document.getElementById("settings-modal"); modal.classList.add("active"); modal.setAttribute("aria-hidden", "false"); }
function close() { const modal = document.getElementById("settings-modal"); modal.classList.remove("active"); modal.setAttribute("aria-hidden", "true"); }

export function initSettings() {
  document.getElementById("open-settings").addEventListener("click", open);
  document.getElementById("close-settings").addEventListener("click", close);
  document.getElementById("settings-modal").addEventListener("click", (event) => { if (event.target.id === "settings-modal") close(); });
  document.getElementById("settings-reset").addEventListener("click", () => fill(getState().savedConfig || getState().config));
  document.getElementById("settings-save").addEventListener("click", () => {
    const config = { ...getState().config };
    config.snapshot_refresh_interval_ms = Math.max(100, Number(document.getElementById(fields.snapshot_refresh_interval_ms).value) || 1000);
    config.status_refresh_interval_ms = Math.max(100, Number(document.getElementById(fields.status_refresh_interval_ms).value) || 2000);
    updateState({ config });
    document.querySelector(".settings-notice").textContent = "临时设置已保存到当前浏览器页面状态；刷新页面后将重新读取 Collector 配置。";
    close();
  });
}
