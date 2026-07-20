import { endpoints, postJson, requestJson } from "./api.js";
import { getState, loadPersistedConfig, normalizeConfig, updateState } from "./state.js";
import { initCalibration, refreshCalibration } from "./pages/calibration.js";
import { initCapture, refreshCapture } from "./pages/capture.js";
import { initValidate } from "./pages/validate.js";
import { initSettings } from "./pages/settings.js";
import { initProduction, refreshProduction, setProductionActive } from "./pages/production.js";

let pendingFactoryPage = null;

function showAdminAuth(targetPage = null) {
  pendingFactoryPage = targetPage;
  const modal = document.getElementById("admin-auth-modal");
  const user = document.getElementById("admin-auth-user");
  const password = document.getElementById("admin-auth-password");
  const message = document.getElementById("admin-auth-message");
  if (user) user.value = "admin";
  if (password) password.value = "";
  if (message) {
    message.textContent = "请输入管理员账号和密码。";
    message.className = "settings-inline-status warn";
  }
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
  setTimeout(() => password?.focus(), 50);
}

function hideAdminAuth() {
  const modal = document.getElementById("admin-auth-modal");
  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
}

function confirmAdminAuth() {
  const user = document.getElementById("admin-auth-user")?.value?.trim();
  const password = document.getElementById("admin-auth-password")?.value ?? "";
  const message = document.getElementById("admin-auth-message");
  if (user === "admin" && password === "admin") {
    const target = pendingFactoryPage || getState().activePage || "calibration";
    pendingFactoryPage = null;
    hideAdminAuth();
    setProductionMode(false, { authenticated: true });
    activateFactoryPage(target, { authenticated: true });
    return;
  }
  if (message) {
    message.textContent = "账号或密码错误。当前测试账号：admin，密码：admin。";
    message.className = "settings-inline-status error";
  }
}

function activateFactoryPage(name, options = {}) {
  if (getState().productionMode && !options.authenticated) {
    showAdminAuth(name);
    return;
  }
  document.querySelectorAll(".top-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.page === name));
  document.querySelectorAll("#factory-mode .page").forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
  updateState({ activePage: name, productionMode: false });
  document.getElementById("factory-mode").classList.add("active");
  document.getElementById("production-mode").classList.remove("active");
  document.getElementById("mode-toggle").textContent = "切换生产模式";
  setProductionActive(false);
  if (name === "calibration") refreshCalibration("position");
  if (name === "capture") refreshCapture();
}

function setProductionMode(entering, options = {}) {
  if (!entering && getState().productionMode && !options.authenticated) {
    showAdminAuth(getState().activePage || "calibration");
    return;
  }
  updateState({ productionMode: entering });
  document.getElementById("factory-mode").classList.toggle("active", !entering);
  document.getElementById("production-mode").classList.toggle("active", entering);
  document.getElementById("mode-toggle").textContent = entering ? "返回工厂模式" : "切换生产模式";
  setProductionActive(entering);
  if (entering) refreshProduction();
}

function toggleProduction() {
  setProductionMode(!getState().productionMode);
}

async function loadConfig() {
  try {
    const backendConfig = normalizeConfig(await requestJson(endpoints.frontendConfig));
    const persistedConfig = loadPersistedConfig();
    updateState({
      config: persistedConfig
        ? normalizeConfig({
            ...backendConfig,
            ...persistedConfig,
            // 生产推理来源属于部署拓扑，不能被浏览器旧缓存覆盖。
            production_inference_source: backendConfig.production_inference_source,
          })
        : backendConfig,
      savedConfig: { ...backendConfig },
    });
  } catch (_error) {
    const fallbackConfig = normalizeConfig(getState().config);
    const persistedConfig = loadPersistedConfig();
    updateState({
      config: persistedConfig ? normalizeConfig({ ...fallbackConfig, ...persistedConfig }) : fallbackConfig,
      savedConfig: { ...fallbackConfig },
    });
  }
}

async function checkCollector() {
  const dot = document.getElementById("global-status-dot"), text = document.getElementById("global-status");
  try { const health = await requestJson("/health"); dot.className = "ok"; text.textContent = `${health.component} / online`; }
  catch (_error) { dot.className = "error"; text.textContent = "Collector unreachable"; }
}

function scheduleSnapshotRefresh(delayMs = 0) {
  setTimeout(async () => {
    const startedAt = performance.now();
    const state = getState();
    if (!state.productionMode && state.activePage === "calibration") await refreshCalibration();
    if (!state.productionMode && state.activePage === "capture") await refreshCapture();
    const targetMs = Math.max(16, Number(getState().config.preview_refresh_interval_ms || 200));
    const remainingMs = Math.max(0, targetMs - (performance.now() - startedAt));
    scheduleSnapshotRefresh(remainingMs);
  }, Math.max(0, delayMs));
}

function scheduleStatusRefresh(delayMs = 0) {
  setTimeout(async () => {
    const startedAt = performance.now();
    await checkCollector();
    if (getState().productionMode) await refreshProduction();
    const targetMs = Math.max(16, Number(getState().config.status_refresh_interval_ms || 2000));
    const remainingMs = Math.max(0, targetMs - (performance.now() - startedAt));
    scheduleStatusRefresh(remainingMs);
  }, Math.max(0, delayMs));
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
  document.getElementById("admin-auth-confirm")?.addEventListener("click", confirmAdminAuth);
  document.getElementById("admin-auth-cancel")?.addEventListener("click", hideAdminAuth);
  document.getElementById("admin-auth-close")?.addEventListener("click", hideAdminAuth);
  document.getElementById("admin-auth-password")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") confirmAdminAuth();
  });
  window.addEventListener("visionops:settings-saved", (event) => {
    const mode = event.detail?.config?.default_mode;
    if (mode === "production") setProductionMode(true);
    if (mode === "factory") setProductionMode(false);
  });
  if (getState().config.default_mode === "production") setProductionMode(true);
  await checkCollector();
  if (!getState().productionMode) await refreshCalibration("position");
  scheduleSnapshotRefresh();
  scheduleStatusRefresh();
}

main();
