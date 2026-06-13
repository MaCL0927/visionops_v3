import { endpoints, requestJson } from "./api.js";
import { getState, updateState } from "./state.js";
import { initCapture, refreshCapture } from "./pages/capture.js";
import { initValidate } from "./pages/validate.js";
import { initProduction, refreshProduction } from "./pages/production.js";

function activatePage(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.page === name));
  document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
  updateState({ activePage: name });
  if (name === "capture") refreshCapture(); if (name === "production") refreshProduction();
}

async function loadConfig() { try { updateState({ config: await requestJson(endpoints.frontendConfig) }); } catch (_error) { /* 使用前端安全默认值 */ } }
async function checkCollector() { const dot = document.getElementById("global-status-dot"), text = document.getElementById("global-status"); try { const health = await requestJson("/health"); dot.className = "status-dot ok"; text.textContent = `${health.component} / online`; } catch (_error) { dot.className = "status-dot error"; text.textContent = "Collector unreachable"; } }

async function main() {
  await loadConfig(); initCapture(); initValidate(); initProduction();
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activatePage(tab.dataset.page)));
  await checkCollector(); await refreshCapture();
  const config = getState().config;
  setInterval(() => { if (getState().activePage === "capture") refreshCapture(); }, config.snapshot_refresh_interval_ms);
  setInterval(() => { checkCollector(); if (getState().activePage === "production") refreshProduction(); }, config.status_refresh_interval_ms);
}

main();
