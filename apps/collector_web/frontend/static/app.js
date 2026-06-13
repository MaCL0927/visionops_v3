"use strict";

const collectorHealth = document.getElementById("collector-health");
const runtimeStatus = document.getElementById("runtime-status");
const latestResult = document.getElementById("latest-result");
const snapshot = document.getElementById("snapshot");

function show(target, value) {
  target.textContent = JSON.stringify(value, null, 2);
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) {
    throw Object.assign(new Error(`HTTP ${response.status}`), { body });
  }
  return body;
}

async function refreshStatus() {
  try {
    show(collectorHealth, await requestJson("/health"));
  } catch (error) {
    show(collectorHealth, error.body || { error: error.message });
  }

  try {
    show(runtimeStatus, await requestJson("/api/collector/status"));
  } catch (error) {
    show(runtimeStatus, error.body || { error: error.message });
  }

  try {
    show(latestResult, await requestJson("/api/runtime/latest_result"));
  } catch (error) {
    show(latestResult, error.body || { error: error.message });
  }

  snapshot.src = `/api/runtime/snapshot.jpg?t=${Date.now()}`;
}

async function postRuntime(path) {
  try {
    const result = await requestJson(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    show(runtimeStatus, result);
    await refreshStatus();
  } catch (error) {
    show(runtimeStatus, error.body || { error: error.message });
  }
}

document.getElementById("refresh-status").addEventListener("click", refreshStatus);
document.getElementById("start-preview").addEventListener("click", () => postRuntime("/api/runtime/start_preview"));
document.getElementById("stop-preview").addEventListener("click", () => postRuntime("/api/runtime/stop_preview"));
document.getElementById("infer-once").addEventListener("click", () => postRuntime("/api/runtime/infer_once"));

refreshStatus();
