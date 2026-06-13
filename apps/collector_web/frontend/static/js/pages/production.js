import { endpoints, requestJson } from "../api.js";

function renderStatus(id, badgeId, value) {
  document.getElementById(id).textContent = JSON.stringify(value, null, 2);
  const health = value?.health || value?.status || (value?.reachable === false ? "unreachable" : "ok");
  const badge = document.getElementById(badgeId); badge.textContent = health; badge.className = `badge ${health}`;
}

function renderRegisters(id, payload) {
  const target = document.getElementById(id), registers = payload?.registers;
  if (!Array.isArray(registers)) { target.className = "register-table empty-copy"; target.textContent = "unreachable / no data"; return; }
  target.className = "register-table"; target.replaceChildren();
  const table = document.createElement("table"), head = document.createElement("thead"), headerRow = document.createElement("tr"), body = document.createElement("tbody");
  for (const label of ["地址", "名称", "值", "类型"]) { const cell = document.createElement("th"); cell.textContent = label; headerRow.append(cell); }
  head.append(headerRow); table.append(head, body);
  for (const item of registers) { const row = document.createElement("tr"); for (const value of [item.address, item.name, item.value, item.type]) { const cell = document.createElement("td"); cell.textContent = String(value ?? ""); row.append(cell); } body.append(row); }
  target.append(table);
}

async function safe(path) { try { return await requestJson(path); } catch (error) { return error.body || { status: "unreachable", reachable: false, error: { message: error.message } }; } }

export async function refreshProduction() {
  const [collector, runtime, gateway, app, latestResult] = await Promise.all([safe(endpoints.collectorStatus), safe(endpoints.runtimeStatus), safe(endpoints.gatewayStatus), safe(endpoints.appStatus), safe(endpoints.latestResult)]);
  renderStatus("collector-status", "collector-badge", collector.collector || collector); renderStatus("runtime-status", "runtime-badge", collector.runtime?.status_response || runtime); renderStatus("gateway-status", "gateway-badge", gateway); renderStatus("app-status", "app-badge", app);
  document.getElementById("production-result-summary").textContent = JSON.stringify(latestResult, null, 2);
  document.getElementById("production-gateway-summary").textContent = JSON.stringify(gateway.latest_gateway_message || { status: gateway.status || "no_message" }, null, 2);
  document.getElementById("production-app-summary").textContent = JSON.stringify(app.latest_decision || { status: app.status || "no_decision" }, null, 2);
  const [gatewayRegisters, appRegisters] = await Promise.all([gateway.reachable === false ? null : safe(endpoints.gatewayRegisters), app.reachable === false ? null : safe(endpoints.appRegisters)]);
  renderRegisters("gateway-registers", gatewayRegisters); renderRegisters("app-registers", appRegisters);
}

export function initProduction() { document.getElementById("production-refresh").addEventListener("click", refreshProduction); }
