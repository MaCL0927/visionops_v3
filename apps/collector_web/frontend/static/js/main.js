import { endpoints, requestJson, postJson } from "./api.js";
import { getState, updateState } from "./state.js";
import { initCalibration, refreshCalibration } from "./pages/calibration.js";
import { initCapture, refreshCapture } from "./pages/capture.js";
import { initValidate } from "./pages/validate.js";
import { initSettings } from "./pages/settings.js";
import { initProduction, refreshProduction } from "./pages/production.js";

function activateFactoryPage(name) {
  document.querySelectorAll(".top-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.page === name));
  document.querySelectorAll("#factory-mode .page").forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
  updateState({ activePage: name, productionMode: false });
  document.getElementById("factory-mode").classList.add("active");
  document.getElementById("production-mode").classList.remove("active");
  document.getElementById("mode-toggle").textContent = "切换生产模式";
  if (name === "calibration") refreshCalibration("position");
  if (name === "capture") refreshCapture();
}

function toggleProduction() {
  const entering = !getState().productionMode;
  updateState({ productionMode: entering });
  document.getElementById("factory-mode").classList.toggle("active", !entering);
  document.getElementById("production-mode").classList.toggle("active", entering);
  document.getElementById("mode-toggle").textContent = entering ? "返回工厂模式" : "切换生产模式";
  if (entering) refreshProduction();
}

async function loadConfig() {
  try { const config = await requestJson(endpoints.frontendConfig); updateState({ config, savedConfig: { ...config } }); }
  catch (_error) { updateState({ savedConfig: { ...getState().config } }); }
}

async function checkCollector() {
  const dot = document.getElementById("global-status-dot"), text = document.getElementById("global-status");
  try { const health = await requestJson("/health"); dot.className = "ok"; text.textContent = `${health.component} / online`; }
  catch (_error) { dot.className = "error"; text.textContent = "Collector unreachable"; }
}

function scheduleSnapshotRefresh() {
  setTimeout(async () => {
    const state = getState();
    if (!state.productionMode && state.activePage === "calibration") await refreshCalibration();
    if (!state.productionMode && state.activePage === "capture") await refreshCapture();
    scheduleSnapshotRefresh();
  }, getState().config.snapshot_refresh_interval_ms);
}

function scheduleStatusRefresh() {
  setTimeout(async () => {
    await checkCollector();
    if (getState().productionMode) await refreshProduction();
    scheduleStatusRefresh();
  }, getState().config.status_refresh_interval_ms);
}

async function main() {
  await loadConfig();
  initCalibration(); initCapture(); initValidate(); initSettings(); initProduction();

  try {
    await postJson(endpoints.startPreview);
  } catch (error) {
    console.warn("start_preview failed", error);
  }

  document.querySelectorAll(".top-tab").forEach((tab) => tab.addEventListener("click", () => activateFactoryPage(tab.dataset.page)));
  document.getElementById("mode-toggle").addEventListener("click", toggleProduction);
  await checkCollector(); await refreshCalibration("position");
  scheduleSnapshotRefresh();
  scheduleStatusRefresh();
}

main();
