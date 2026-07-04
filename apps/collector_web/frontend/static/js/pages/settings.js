import { ApiError, endpoints, postJson, requestJson } from "../api.js";
import { getState, normalizeConfig, persistConfig, updateState } from "../state.js";

const fields = {
  runtime_url: "setting-runtime-url",
  gateway_url: "setting-gateway-url",
  business_app_url: "setting-business-app-url",
  device_id: "setting-device-id",
  display_fps: "setting-display-fps",
  inference_interval_ms: "setting-inference-interval",
  status_refresh_interval_ms: "setting-status-interval",
  camera_model: "setting-camera-model",
  rgb_profile: "setting-rgb-profile",
  depth_profile: "setting-depth-profile",
  depth_unit: "setting-depth-unit",
  rgb_source_preference: "setting-rgb-source-preference",
  flip_vertical: "setting-flip-vertical",
  flip_horizontal: "setting-flip-horizontal",
  rgb_order: "setting-rgb-order",
  orbbec_serial: "setting-orbbec-serial",
  camera_jpeg_quality: "setting-camera-jpeg-quality",
  default_mode: "setting-default-mode",
  models_root: "setting-models-root",
  data_root: "setting-data-root",
  log_root: "setting-log-root",
  disk_warning_percent: "setting-disk-warning",
  runtime_port: "setting-runtime-port",
  collector_port: "setting-collector-port",
  preprocess_backend_preference: "setting-preprocess-backend",
  task_view_preference: "setting-task-view",
};

const overlayFields = {
  show_labels: "setting-overlay-show-labels",
  show_centers: "setting-overlay-show-centers",
  show_detection_bbox: "setting-overlay-show-detection-bbox",
  show_obb_rotated: "setting-overlay-show-obb-rotated",
  show_obb_bbox: "setting-overlay-show-obb-bbox",
  show_segmentation_bbox: "setting-overlay-show-seg-bbox",
  show_segmentation_mask: "setting-overlay-show-seg-mask",
  mask_opacity: "setting-overlay-mask-opacity",
};

let latestBridgeSettings = null;
let bridgeSettingsLoading = false;

function element(id) { return document.getElementById(id); }

function setValue(id, value) {
  const node = element(id);
  if (!node) return;
  node.value = value ?? "";
}

function setChecked(id, value) {
  const node = element(id);
  if (!node) return;
  node.checked = Boolean(value);
}

function getValue(id, fallback = "") {
  const node = element(id);
  if (!node) return fallback;
  return node.value;
}

function getNumber(id, fallback) {
  const value = Number(getValue(id, fallback));
  return Number.isFinite(value) ? value : fallback;
}

function getChecked(id, fallback = false) {
  const node = element(id);
  if (!node) return fallback;
  return Boolean(node.checked);
}

function intervalMsToFps(ms, fallback = 5) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return fallback;
  return Math.max(1, Math.min(30, Math.round(1000 / value)));
}

function parseProfile(profile, fallback = { resolution: "1280x720", fps: 30 }) {
  const raw = String(profile || "");
  const match = raw.match(/:(\d+)x(\d+)@(\d+)$/);
  if (!match) return fallback;
  return { resolution: `${match[1]}x${match[2]}`, fps: Number(match[3]), width: Number(match[1]), height: Number(match[2]) };
}

function showSaveStatus(text, kind = "") {
  const saveStatus = element("settings-save-status");
  if (!saveStatus) return;
  saveStatus.textContent = text;
  saveStatus.dataset.kind = kind;
}

function setNotice(text, kind = "") {
  const notice = document.querySelector(".settings-notice");
  if (!notice) return;
  notice.textContent = text;
  notice.dataset.kind = kind;
}

function updateDepthMatchHint() {
  const hint = element("setting-depth-match-hint");
  if (!hint) return;
  const rgb = getValue(fields.rgb_profile, "");
  const depth = getValue(fields.depth_profile, "");
  const rgbParsed = parseProfile(rgb, null);
  const depthParsed = parseProfile(depth, null);
  if (!rgbParsed || !depthParsed) {
    hint.textContent = "等待从 SDK Bridge 读取 RGB / Depth profile。";
    hint.className = "profile-match-hint warn";
    return;
  }
  const sameResolution = rgbParsed.resolution === depthParsed.resolution;
  const sameFps = rgbParsed.fps === depthParsed.fps;
  if (sameResolution && sameFps) {
    hint.textContent = `匹配：${rgbParsed.resolution} @ ${rgbParsed.fps} FPS`;
    hint.className = "profile-match-hint ok";
  } else if (sameFps) {
    hint.textContent = `FPS 匹配，分辨率不同：RGB ${rgbParsed.resolution}，Depth ${depthParsed.resolution}`;
    hint.className = "profile-match-hint warn";
  } else {
    hint.textContent = `FPS 不匹配，无法写入当前 Bridge env：RGB ${rgbParsed.fps} FPS，Depth ${depthParsed.fps} FPS`;
    hint.className = "profile-match-hint warn";
  }
}

function profileOptionLabel(profile) {
  if (!profile) return "unknown";
  return profile.label || `${profile.width}×${profile.height} @ ${profile.fps} FPS`;
}

function populateProfileSelect(id, profiles, selected, emptyText) {
  const node = element(id);
  if (!node) return;
  const currentValue = selected || node.value;
  node.innerHTML = "";
  if (!profiles || profiles.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyText || "没有可用 profile";
    node.appendChild(option);
    return;
  }
  profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profileOptionLabel(profile);
    option.dataset.width = profile.width;
    option.dataset.height = profile.height;
    option.dataset.fps = profile.fps;
    node.appendChild(option);
  });
  if (currentValue && Array.from(node.options).some((option) => option.value === currentValue)) {
    node.value = currentValue;
  } else {
    node.value = node.options[0]?.value || "";
  }
}

function updateBridgeStatus(settings) {
  const serviceNode = element("setting-bridge-service-status");
  if (serviceNode) {
    const service = settings?.service || {};
    const active = service.active || "unknown";
    serviceNode.textContent = `${service.name || "visionops-orbbec336l-bridge.service"}: ${active}`;
    serviceNode.className = active === "active" ? "profile-match-hint ok" : "profile-match-hint warn";
  }
  const sourceNode = element("setting-bridge-profile-source");
  if (sourceNode) {
    const profiles = settings?.profiles || {};
    if (profiles.source === "bridge_api") {
      sourceNode.textContent = `Profile 来源：SDK Bridge 实时枚举 (${profiles.profile_url || ""})`;
      sourceNode.className = "settings-inline-status";
    } else {
      sourceNode.textContent = profiles.warning || "Profile 来源：当前 env 回退值；请升级 / 重启 Orbbec Bridge 以启用 SDK 实时枚举。";
      sourceNode.className = "settings-inline-status warn";
    }
  }
}

async function loadOrbbecSettings() {
  if (bridgeSettingsLoading) return;
  bridgeSettingsLoading = true;
  showSaveStatus("正在读取 Orbbec 336L SDK Bridge 设置...", "loading");
  try {
    const settings = await requestJson(endpoints.orbbecSettings);
    latestBridgeSettings = settings;
    const current = settings.settings || {};
    populateProfileSelect(fields.rgb_profile, settings.profiles?.color || [], current.rgb_profile, "未读取到 RGB profile");
    populateProfileSelect(fields.depth_profile, settings.profiles?.depth || [], current.depth_profile, "未读取到 Depth profile");
    setValue(fields.camera_model, "orbbec336l");
    setValue(fields.display_fps, current.display_fps ?? getState().config.display_fps);
    setValue(fields.camera_jpeg_quality, current.camera_jpeg_quality ?? getState().config.camera_jpeg_quality);
    setValue(fields.flip_vertical, String(current.flip_vertical ?? getState().config.flip_vertical));
    setValue(fields.flip_horizontal, String(current.flip_horizontal ?? getState().config.flip_horizontal));
    setValue(fields.depth_unit, current.depth_unit ?? getState().config.depth_unit);
    setValue(fields.orbbec_serial, current.orbbec_serial ?? "");
    updateDepthMatchHint();
    updateBridgeStatus(settings);
    const source = settings.profiles?.source === "bridge_api" ? "SDK 实时枚举" : "env 回退";
    showSaveStatus(`已读取 Orbbec 设置，profile 来源：${source}`);
  } catch (error) {
    latestBridgeSettings = null;
    populateProfileSelect(fields.rgb_profile, [], "", "读取 profile 失败");
    populateProfileSelect(fields.depth_profile, [], "", "读取 profile 失败");
    updateDepthMatchHint();
    updateBridgeStatus(null);
    const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
    showSaveStatus(`读取 Orbbec 设置失败：${detail}`, "error");
  } finally {
    bridgeSettingsLoading = false;
  }
}

function fill(config) {
  const displayFps = config.display_fps ?? intervalMsToFps(config.preview_refresh_interval_ms, 5);
  for (const [key, id] of Object.entries(fields)) {
    if (key === "display_fps") setValue(id, displayFps);
    else setValue(id, config[key]);
  }
  const overlay = config.overlay || {};
  for (const [key, id] of Object.entries(overlayFields)) {
    if (key === "mask_opacity") setValue(id, overlay[key]);
    else setChecked(id, overlay[key]);
  }
  updateDepthMatchHint();
}

function readConfigFromForm() {
  const current = getState().config;
  const displayFps = getNumber(fields.display_fps, 5);
  const previewIntervalMs = Math.max(100, Math.round(1000 / Math.max(1, displayFps)));
  const rgbProfile = getValue(fields.rgb_profile, current.rgb_profile || "orbbec:1280x720@30");
  const rgbParsed = parseProfile(rgbProfile, { resolution: current.camera_resolution || "1280x720", fps: current.camera_read_fps || 30 });
  return normalizeConfig({
    ...current,
    camera_type: "sdk_bridge",
    camera_model: getValue(fields.camera_model, "orbbec336l"),
    display_fps: displayFps,
    preview_refresh_interval_ms: previewIntervalMs,
    snapshot_refresh_interval_ms: previewIntervalMs,
    inference_interval_ms: getNumber(fields.inference_interval_ms, 500),
    status_refresh_interval_ms: getNumber(fields.status_refresh_interval_ms, 2000),
    rgb_profile: rgbProfile,
    depth_profile: getValue(fields.depth_profile, current.depth_profile || "orbbec:1280x720@30"),
    depth_unit: getValue(fields.depth_unit, "mm"),
    rgb_source_preference: getValue(fields.rgb_source_preference, "auto"),
    flip_vertical: getValue(fields.flip_vertical, "false"),
    flip_horizontal: getValue(fields.flip_horizontal, "false"),
    rgb_order: getValue(fields.rgb_order, "bgr"),
    orbbec_serial: getValue(fields.orbbec_serial, ""),
    camera_read_fps: rgbParsed.fps,
    camera_resolution: rgbParsed.resolution,
    camera_jpeg_quality: getNumber(fields.camera_jpeg_quality, 100),
    default_mode: getValue(fields.default_mode, "factory"),
    models_root: getValue(fields.models_root, "/opt/visionops_v3/models"),
    data_root: getValue(fields.data_root, "/opt/visionops_v3/data"),
    log_root: getValue(fields.log_root, "/opt/visionops_v3/logs"),
    disk_warning_percent: getNumber(fields.disk_warning_percent, 85),
    runtime_port: getNumber(fields.runtime_port, 28081),
    collector_port: getNumber(fields.collector_port, 18091),
    preprocess_backend_preference: getValue(fields.preprocess_backend_preference, "auto"),
    task_view_preference: getValue(fields.task_view_preference, "auto"),
    overlay: {
      show_labels: getChecked(overlayFields.show_labels, true),
      show_centers: getChecked(overlayFields.show_centers, true),
      show_detection_bbox: getChecked(overlayFields.show_detection_bbox, true),
      show_obb_rotated: getChecked(overlayFields.show_obb_rotated, true),
      show_obb_bbox: getChecked(overlayFields.show_obb_bbox, false),
      show_segmentation_bbox: getChecked(overlayFields.show_segmentation_bbox, true),
      show_segmentation_mask: getChecked(overlayFields.show_segmentation_mask, true),
      mask_opacity: getNumber(overlayFields.mask_opacity, 0.28),
    },
  });
}

function open() {
  fill(getState().config);
  const modal = element("settings-modal");
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
  setNotice("相机设置已接入 Orbbec 336L SDK Bridge API；保存会写入 v3 camera_bridge env。未检测到 env 变更时不会重启服务。");
  loadOrbbecSettings();
}

function close() {
  const modal = element("settings-modal");
  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
}

function activateSettingsTab(panelId) {
  document.querySelectorAll(".settings-main-tab").forEach((tab) => {
    const active = tab.dataset.settingsTab === panelId;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".settings-panel").forEach((panel) => panel.classList.toggle("active", panel.id === panelId));
}

function markUnsaved() { showSaveStatus("有未保存的修改"); }

function buildOrbbecPayload(config) {
  return {
    camera_model: "orbbec336l",
    rgb_profile: config.rgb_profile,
    depth_profile: config.depth_profile,
    display_fps: config.display_fps,
    camera_jpeg_quality: config.camera_jpeg_quality,
    flip_vertical: config.flip_vertical,
    flip_horizontal: config.flip_horizontal,
    depth_unit: config.depth_unit,
    orbbec_serial: config.orbbec_serial,
    known_profiles: latestBridgeSettings?.profiles || null,
  };
}

async function saveSettings() {
  const config = readConfigFromForm();
  const cameraModel = config.camera_model || "orbbec336l";
  showSaveStatus("正在保存设置...", "loading");
  if (cameraModel === "orbbec336l" || cameraModel === "auto") {
    try {
      const result = await postJson(endpoints.orbbecSettings, buildOrbbecPayload(config));
      latestBridgeSettings = result;
      updateBridgeStatus(result);
      populateProfileSelect(fields.rgb_profile, result.profiles?.color || [], result.settings?.rgb_profile || config.rgb_profile, "未读取到 RGB profile");
      populateProfileSelect(fields.depth_profile, result.profiles?.depth || [], result.settings?.depth_profile || config.depth_profile, "未读取到 Depth profile");
      updateDepthMatchHint();
      const timings = result.apply_timings_ms || {};
      const totalMs = timings.total_apply_ms != null ? `，总耗时 ${timings.total_apply_ms}ms` : "";
      if (result.changed === false || result.skipped_restart) {
        showSaveStatus(`设置未变化，已跳过写入和服务重启${totalMs}`, "ok");
        setNotice("设置未变化：env 内容与当前界面一致，已跳过重启和健康检查。", "ok");
      } else {
        showSaveStatus(`已写入 v3 camera_bridge env 并重启服务${totalMs}`, "ok");
        const restartMs = timings.restart_service_ms != null ? `restart=${timings.restart_service_ms}ms` : "restart=?";
        const healthMs = timings.wait_health_ms != null ? `health=${timings.wait_health_ms}ms` : "health=?";
        const profileMs = timings.profile_validation_ms != null ? `profile=${timings.profile_validation_ms}ms` : "profile=?";
        setNotice(`设置已真实应用到 Orbbec 336L SDK Bridge；如果画面短暂中断，这是服务重启导致的正常现象。耗时：${profileMs}, ${restartMs}, ${healthMs}。`, "ok");
      }
    } catch (error) {
      const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
      showSaveStatus(`保存失败：${detail}`, "error");
      setNotice(`Orbbec 设置未生效：${detail}`, "error");
      return;
    }
  } else {
    showSaveStatus("当前只接入 Orbbec 336L 设置 API，HP60C 暂未真实应用", "warn");
  }

  persistConfig(config);
  updateState({ config });
  window.dispatchEvent(new CustomEvent("visionops:settings-saved", { detail: { config } }));
  // 不自动关闭，保留 apply 结果给现场人员确认。
}

export function initSettings() {
  element("open-settings").addEventListener("click", open);
  element("close-settings").addEventListener("click", close);
  element("settings-modal").addEventListener("click", (event) => { if (event.target.id === "settings-modal") close(); });
  document.querySelectorAll(".settings-main-tab").forEach((tab) => tab.addEventListener("click", () => activateSettingsTab(tab.dataset.settingsTab)));
  document.querySelectorAll("#settings-modal input, #settings-modal select").forEach((input) => input.addEventListener("change", markUnsaved));
  [fields.rgb_profile, fields.depth_profile].forEach((id) => {
    const node = element(id);
    if (node) node.addEventListener("change", updateDepthMatchHint);
  });
  const refreshProfiles = element("settings-refresh-profiles");
  if (refreshProfiles) refreshProfiles.addEventListener("click", loadOrbbecSettings);
  element("settings-reset").addEventListener("click", () => {
    fill(getState().savedConfig || getState().config);
    loadOrbbecSettings();
    showSaveStatus("已恢复到 Collector 默认或当前 Bridge 配置，尚未保存");
  });
  element("settings-save").addEventListener("click", saveSettings);
}
