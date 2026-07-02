import { getState, normalizeConfig, persistConfig, updateState } from "../state.js";

const fields = {
  runtime_url: "setting-runtime-url", gateway_url: "setting-gateway-url",
  business_app_url: "setting-business-app-url", device_id: "setting-device-id",
  preview_refresh_interval_ms: "setting-preview-interval",
  inference_interval_ms: "setting-inference-interval",
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
    const config = normalizeConfig({
      ...getState().config,
      preview_refresh_interval_ms: Number(document.getElementById(fields.preview_refresh_interval_ms).value) || 200,
      inference_interval_ms: Number(document.getElementById(fields.inference_interval_ms).value) || 500,
      snapshot_refresh_interval_ms: Number(document.getElementById(fields.snapshot_refresh_interval_ms).value) || 200,
      status_refresh_interval_ms: Number(document.getElementById(fields.status_refresh_interval_ms).value) || 2000,
    });
    persistConfig(config);
    updateState({ config });
    document.querySelector(".settings-notice").textContent = "当前已保存到浏览器 localStorage，并会覆盖 Collector 默认前端间隔设置；不会写入 .env 或源 YAML。";
    close();
  });
}
