import { ApiError, endpoints, postJson, requestJson } from "../api.js";
import { getState, normalizeConfig, persistConfig, updateState } from "../state.js";

const fields = {
  runtime_url: "setting-runtime-url",
  gateway_url: "setting-gateway-url",
  business_app_url: "setting-business-app-url",
  device_id: "setting-device-id",
  status_refresh_fps: "setting-status-fps",
  display_fps: "setting-display-fps",
  camera_model: "setting-camera-model",
  rgb_profile: "setting-rgb-profile",
  depth_profile: "setting-depth-profile",
  depth_unit: "setting-depth-unit",
  rgb_source_preference: "setting-rgb-source-preference",
  flip_vertical: "setting-flip-vertical",
  flip_horizontal: "setting-flip-horizontal",
  rgb_order: "setting-rgb-order",
  orbbec_serial: "setting-orbbec-serial",
  hp60c_config_path: "setting-hp60c-config-path",
  hp60c_fx: "setting-hp60c-fx",
  hp60c_fy: "setting-hp60c-fy",
  hp60c_cx: "setting-hp60c-cx",
  hp60c_cy: "setting-hp60c-cy",
  camera_jpeg_quality: "setting-camera-jpeg-quality",
  default_mode: "setting-default-mode",
  models_root: "setting-models-root",
  data_root: "setting-data-root",
  log_root: "setting-log-root",
  disk_warning_percent: "setting-disk-warning",
  runtime_port: "setting-runtime-port",
  collector_port: "setting-collector-port",
  upload_server_ip: "setting-upload-server-ip",
  upload_ssh_user: "setting-upload-ssh-user",
  upload_ssh_password: "setting-upload-ssh-password",
  upload_ssh_port: "setting-upload-ssh-port",
  upload_remote_dir: "setting-upload-remote-dir",
  upload_timeout: "setting-upload-timeout",
  eth0_ip: "setting-eth0-ip",
  eth0_netmask: "setting-eth0-netmask",
  eth0_gateway: "setting-eth0-gateway",
  eth1_ip: "setting-eth1-ip",
  eth1_netmask: "setting-eth1-netmask",
  eth1_gateway: "setting-eth1-gateway",
  inference_fps: "setting-inference-fps",
  algorithm_model: "setting-algorithm-model",
  algorithm_task: "setting-algorithm-task",
  algorithm_input_size: "setting-algorithm-input-size",
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

const taskThresholdFields = {
  classification: {
    score: "setting-classification-score-threshold",
    nms: null,
  },
  detection: {
    score: "setting-detection-score-threshold",
    nms: "setting-detection-nms-threshold",
  },
  obb: {
    score: "setting-obb-score-threshold",
    nms: "setting-obb-nms-threshold",
  },
  segmentation: {
    score: "setting-seg-score-threshold",
    nms: "setting-seg-nms-threshold",
  },
};

let latestBridgeSettings = null;
let bridgeSettingsLoading = false;
let latestAlgorithmSettings = null;
let algorithmSettingsLoading = false;
let latestVisionBoxSettings = null;
let visionBoxSettingsLoading = false;

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

function fpsToIntervalMs(fps, fallbackMs = 500) {
  const value = Number(fps);
  if (!Number.isFinite(value) || value <= 0) return fallbackMs;
  return Math.max(16, Math.round(1000 / value));
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
  const model = settings?.camera_model || getValue(fields.camera_model, "orbbec336l");
  const name = model === "hp60c" ? "HP60C / HP60CN" : "Orbbec Gemini 336L";
  if (serviceNode) {
    const service = settings?.service || {};
    const active = service.active || "unknown";
    const activeMark = settings?.active_camera === model ? "当前图像源" : "待切换";
    serviceNode.textContent = `${name} / ${service.name || "SDK Bridge"}: ${active}；${activeMark}`;
    serviceNode.className = active === "active" ? "profile-match-hint ok" : "profile-match-hint warn";
  }
  const sourceNode = element("setting-bridge-profile-source");
  if (sourceNode) {
    const profiles = settings?.profiles || {};
    const hpNote = model === "hp60c" ? "；HP60C 的实际曝光/profile 由 Angstrong 配置文件控制" : "";
    if (profiles.source === "bridge_api") {
      sourceNode.textContent = `Profile 来源：${name} Bridge 实时接口 (${profiles.profile_url || ""})${hpNote}`;
      sourceNode.className = "settings-inline-status";
    } else {
      sourceNode.textContent = `${profiles.warning || "Profile 来源：当前 env 回退值"}${hpNote}`;
      sourceNode.className = "settings-inline-status warn";
    }
  }
}

function updateCameraSpecificFields(model) {
  const hp = model === "hp60c";
  const serialField = element("setting-orbbec-serial-field");
  const hpConfigField = element("setting-hp60c-config-field");
  const hpIntrinsics = element("setting-hp60c-intrinsics");
  if (serialField) serialField.hidden = hp;
  if (hpConfigField) hpConfigField.hidden = !hp;
  if (hpIntrinsics) hpIntrinsics.hidden = !hp;
  const source = element(fields.rgb_source_preference);
  const order = element(fields.rgb_order);
  const rgbProfile = element(fields.rgb_profile);
  const depthProfile = element(fields.depth_profile);
  if (source) source.disabled = !hp;
  if (order) order.disabled = !hp;
  if (rgbProfile) rgbProfile.disabled = hp;
  if (depthProfile) depthProfile.disabled = hp;
  const hint = element("setting-camera-apply-hint");
  if (hint) {
    hint.textContent = hp
      ? "HP60C 参数写入 hp60c_sdk_bridge.env，Bridge 端口 18181。保存后重启 HP60C Bridge 与当前 Runtime。实际曝光/profile 仍由 Angstrong 配置文件决定。"
      : "Orbbec 参数写入 orbbec336l_bridge.env，Bridge 端口 18182。保存后重启 Orbbec Bridge 与当前 Runtime。";
  }
}


function updateNetworkStatus(payload) {
  const network = payload?.network || {};
  const interfaces = network.interfaces || {};
  for (const iface of ["eth0", "eth1"]) {
    const item = interfaces[iface] || {};
    const stateNode = element(`setting-${iface}-state`);
    if (stateNode) {
      if (item.exists) {
        const ip = item.ip || "未配置 IP";
        const gateway = item.gateway ? `，网关 ${item.gateway}` : "";
        stateNode.textContent = `${item.state || "unknown"}，${ip}/${item.prefix ?? ""}${gateway}`;
        stateNode.className = item.ip ? "profile-match-hint ok" : "profile-match-hint warn";
      } else {
        stateNode.textContent = item.error ? `未检测到：${item.error}` : "未检测到该网口";
        stateNode.className = "profile-match-hint warn";
      }
    }
  }
  const status = element("setting-network-status");
  if (status) {
    const apply = payload?.network_apply || null;
    if (apply?.attempted) {
      status.textContent = apply.ok ? `双网口配置已应用，耗时 ${apply.duration_ms ?? 0} ms。` : `双网口配置应用失败：${JSON.stringify(apply.errors || [])}`;
      status.className = apply.ok ? "settings-inline-status" : "settings-inline-status warn";
    } else {
      status.textContent = "已读取 eth0 / eth1 当前状态；修改后点击保存会立即执行 ip 命令应用。";
      status.className = "settings-inline-status warn";
    }
  }
}

function updateVisionBoxStatus(payload) {
  const badge = element("vision-box-settings-badge");
  if (badge) badge.textContent = payload?.config_path ? "配置已接入" : "边缘端设置 API";
  const storageNode = element("setting-storage-status");
  if (storageNode) {
    const storage = payload?.storage?.project_root || null;
    if (storage && storage.used_percent != null) {
      storageNode.textContent = `磁盘使用率 ${storage.used_percent}% / 告警阈值 ${storage.warning_percent}%（${storage.path}）`;
      storageNode.className = storage.warning ? "settings-inline-status warn" : "settings-inline-status";
    } else {
      storageNode.textContent = "未读取到磁盘状态";
      storageNode.className = "settings-inline-status warn";
    }
  }
}

function fillVisionBoxSettings(payload) {
  const settings = payload?.settings || {};
  const services = payload?.services || {};
  const paths = payload?.paths || {};
  const upload = settings.upload || {};
  setValue(fields.runtime_url, services.runtime_url ?? getState().config.runtime_url);
  setValue(fields.gateway_url, services.gateway_url ?? getState().config.gateway_url);
  setValue(fields.business_app_url, services.business_app_url ?? getState().config.business_app_url);
  setValue(fields.device_id, services.device_id ?? getState().config.device_id);
  setValue(fields.status_refresh_fps, settings.status_refresh_fps ?? intervalMsToFps(getState().config.status_refresh_interval_ms, 0.5));
  setValue(fields.default_mode, settings.default_mode ?? getState().config.default_mode ?? "factory");
  setValue(fields.models_root, paths.models_root ?? getState().config.models_root);
  setValue(fields.data_root, paths.data_root ?? getState().config.data_root);
  setValue(fields.log_root, paths.log_root ?? getState().config.log_root);
  setValue(fields.disk_warning_percent, settings.disk_warning_percent ?? getState().config.disk_warning_percent);
  setValue(fields.runtime_port, services.runtime_port ?? getState().config.runtime_port);
  setValue(fields.collector_port, services.collector_port ?? getState().config.collector_port);
  setValue(fields.upload_server_ip, upload.server_ip ?? "");
  setValue(fields.upload_ssh_user, upload.ssh_user ?? "");
  setValue(fields.upload_ssh_password, upload.ssh_password ?? "");
  setValue(fields.upload_ssh_port, upload.ssh_port ?? 22);
  setValue(fields.upload_remote_dir, upload.remote_dir ?? "/opt/visionops_uploads");
  setValue(fields.upload_timeout, upload.timeout_s ?? 60);
  const configuredNetwork = payload?.settings?.network || {};
  const liveNetwork = payload?.network?.interfaces || {};
  const eth0 = configuredNetwork.eth0 || liveNetwork.eth0 || {};
  const eth1 = configuredNetwork.eth1 || liveNetwork.eth1 || {};
  setValue(fields.eth0_ip, eth0.ip ?? "");
  setValue(fields.eth0_netmask, eth0.netmask ?? "");
  setValue(fields.eth0_gateway, eth0.gateway ?? "");
  setValue(fields.eth1_ip, eth1.ip ?? "");
  setValue(fields.eth1_netmask, eth1.netmask ?? "");
  setValue(fields.eth1_gateway, eth1.gateway ?? "");
  updateVisionBoxStatus(payload);
  updateNetworkStatus(payload);
}

async function loadVisionBoxSettings() {
  if (visionBoxSettingsLoading) return;
  visionBoxSettingsLoading = true;
  try {
    const payload = await requestJson(endpoints.visionBoxSettings);
    latestVisionBoxSettings = payload;
    fillVisionBoxSettings(payload);
  } catch (error) {
    latestVisionBoxSettings = null;
    const storageNode = element("setting-storage-status");
    if (storageNode) {
      const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
      storageNode.textContent = `读取视觉盒子设置失败：${detail}`;
      storageNode.className = "settings-inline-status warn";
    }
  } finally {
    visionBoxSettingsLoading = false;
  }
}

function buildVisionBoxPayload(config) {
  return {
    default_mode: config.default_mode,
    status_refresh_fps: getNumber(fields.status_refresh_fps, intervalMsToFps(config.status_refresh_interval_ms, 0.5)),
    disk_warning_percent: config.disk_warning_percent,
    upload: {
      server_ip: getValue(fields.upload_server_ip, ""),
      ssh_user: getValue(fields.upload_ssh_user, ""),
      ssh_password: getValue(fields.upload_ssh_password, ""),
      ssh_port: getNumber(fields.upload_ssh_port, 22),
      remote_dir: getValue(fields.upload_remote_dir, "/opt/visionops_uploads"),
      timeout_s: getNumber(fields.upload_timeout, 60),
    },
    network: {
      interfaces: {
        eth0: {
          ip: getValue(fields.eth0_ip, ""),
          netmask: getValue(fields.eth0_netmask, ""),
          gateway: getValue(fields.eth0_gateway, ""),
        },
        eth1: {
          ip: getValue(fields.eth1_ip, ""),
          netmask: getValue(fields.eth1_netmask, ""),
          gateway: getValue(fields.eth1_gateway, ""),
        },
      },
    },
  };
}

async function saveVisionBoxSettings(config) {
  const result = await postJson(endpoints.visionBoxSettings, buildVisionBoxPayload(config));
  latestVisionBoxSettings = result;
  fillVisionBoxSettings(result);
  if (result.changed === false) return { skipped: true, message: "视觉盒子设置未变化" };
  const net = result.network_apply || {};
  if (net.attempted) return { skipped: false, message: net.ok ? "视觉盒子设置已保存，双网口配置已应用" : "视觉盒子设置已保存，但双网口配置应用失败" };
  return { skipped: false, message: "视觉盒子设置已保存" };
}

async function loadCameraSettings(requestedModel = "") {
  if (bridgeSettingsLoading) return;
  bridgeSettingsLoading = true;
  const selected = requestedModel || getValue(fields.camera_model, "");
  const suffix = selected ? `?camera_model=${encodeURIComponent(selected)}` : "";
  showSaveStatus("正在读取相机 SDK Bridge 设置...", "loading");
  try {
    const settings = await requestJson(`${endpoints.sdkBridgeSettings}${suffix}`);
    latestBridgeSettings = settings;
    const current = settings.settings || {};
    const model = settings.camera_model || settings.active_camera || "orbbec336l";
    populateProfileSelect(fields.rgb_profile, settings.profiles?.color || [], current.rgb_profile, "未读取到 RGB profile");
    populateProfileSelect(fields.depth_profile, settings.profiles?.depth || [], current.depth_profile, "未读取到 Depth profile");
    setValue(fields.camera_model, model);
    setValue(fields.display_fps, current.display_fps ?? getState().config.display_fps);
    setValue(fields.camera_jpeg_quality, current.camera_jpeg_quality ?? getState().config.camera_jpeg_quality);
    setValue(fields.flip_vertical, String(current.flip_vertical ?? getState().config.flip_vertical));
    setValue(fields.flip_horizontal, String(current.flip_horizontal ?? getState().config.flip_horizontal));
    setValue(fields.depth_unit, current.depth_unit ?? getState().config.depth_unit);
    setValue(fields.orbbec_serial, current.orbbec_serial ?? "");
    setValue(fields.hp60c_config_path, current.hp60c_config_path ?? "");
    setValue(fields.hp60c_fx, current.hp60c_fx ?? 0);
    setValue(fields.hp60c_fy, current.hp60c_fy ?? 0);
    setValue(fields.hp60c_cx, current.hp60c_cx ?? 0);
    setValue(fields.hp60c_cy, current.hp60c_cy ?? 0);
    setValue(fields.rgb_source_preference, current.rgb_source_preference ?? "auto");
    setValue(fields.rgb_order, current.rgb_order === "auto" ? "bgr" : (current.rgb_order ?? "bgr"));
    updateCameraSpecificFields(model);
    updateDepthMatchHint();
    updateBridgeStatus(settings);
    const source = settings.profiles?.source === "bridge_api" ? "Bridge 实时接口" : "env 回退";
    const activeText = settings.active_camera === model ? "当前已选中" : "保存后切换";
    showSaveStatus(`已读取 ${model === "hp60c" ? "HP60C" : "Orbbec 336L"} 设置，${activeText}，profile 来源：${source}`);
  } catch (error) {
    latestBridgeSettings = null;
    populateProfileSelect(fields.rgb_profile, [], "", "读取 profile 失败");
    populateProfileSelect(fields.depth_profile, [], "", "读取 profile 失败");
    updateDepthMatchHint();
    updateBridgeStatus(null);
    const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
    showSaveStatus(`读取相机设置失败：${detail}`, "error");
  } finally {
    bridgeSettingsLoading = false;
  }
}

function populateAlgorithmModelSelect(settings) {
  const node = element(fields.algorithm_model);
  if (!node) return;
  const selectedModel = settings?.selected_model || null;
  const currentValue = selectedModel?.model_id || node.value;
  node.innerHTML = "";
  const models = Array.isArray(settings?.models) ? settings.models.filter((item) => item.valid) : [];
  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "没有可用模型";
    node.appendChild(option);
    return;
  }
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = `${model.model_name || model.package_dir} / ${model.task_type || "unknown"} / ${model.input_size?.join?.("×") || "--"}`;
    node.appendChild(option);
  }
  if (currentValue && Array.from(node.options).some((option) => option.value === currentValue)) {
    node.value = currentValue;
  } else {
    node.value = node.options[0]?.value || "";
  }
}

function setTaskCardEnabled(card, enabled) {
  card.classList.toggle("disabled", !enabled);
  card.querySelectorAll("input, select").forEach((input) => {
    input.disabled = !enabled;
  });
}

function updateAlgorithmTaskState(taskType) {
  const task = String(taskType || "").toLowerCase();
  document.querySelectorAll(".algorithm-task-card").forEach((card) => {
    setTaskCardEnabled(card, card.dataset.algorithmTask === task);
  });
  const badge = element("algorithm-model-task-badge");
  if (badge) badge.textContent = task ? `${task} 参数` : "未选择模型";
}

function fillTaskThresholds(settings) {
  const task = String(settings?.settings?.task_type || "").toLowerCase();
  const score = settings?.settings?.score_threshold ?? 0.5;
  const nms = settings?.settings?.nms_threshold ?? 0.45;
  for (const [name, ids] of Object.entries(taskThresholdFields)) {
    setValue(ids.score, name === task ? score : "");
    if (ids.nms) setValue(ids.nms, name === task ? nms : "");
  }
}

function renderAlgorithmSettings(settings) {
  latestAlgorithmSettings = settings;
  populateAlgorithmModelSelect(settings);
  const selected = settings?.selected_model || {};
  const alg = settings?.settings || {};
  const task = String(alg.task_type || selected.task_type || "").toLowerCase();
  setValue(fields.algorithm_task, task || "--");
  setValue(fields.algorithm_input_size, Array.isArray(alg.input_size) ? alg.input_size.join(" × ") : (Array.isArray(selected.input_size) ? selected.input_size.join(" × ") : "--"));
  fillTaskThresholds(settings);
  updateAlgorithmTaskState(task);
  const status = element("setting-algorithm-status");
  if (status) {
    const activeText = alg.active ? "当前 Runtime 正在使用该模型" : "当前选择模型未必是 Runtime 正在使用的模型";
    status.textContent = `${selected.model_name || selected.package_dir || "未选择模型"}: ${activeText}；阈值来源 ${alg.yaml_path || "model.yaml"}`;
    status.className = "settings-inline-status";
  }
}

async function loadAlgorithmSettings(modelId = "") {
  if (algorithmSettingsLoading) return;
  algorithmSettingsLoading = true;
  const suffix = modelId ? `?model_id=${encodeURIComponent(modelId)}` : "";
  try {
    const settings = await requestJson(`${endpoints.algorithmSettings}${suffix}`);
    renderAlgorithmSettings(settings);
  } catch (error) {
    latestAlgorithmSettings = null;
    updateAlgorithmTaskState("");
    const status = element("setting-algorithm-status");
    if (status) {
      const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
      status.textContent = `读取算法设置失败：${detail}`;
      status.className = "settings-inline-status warn";
    }
  } finally {
    algorithmSettingsLoading = false;
  }
}


async function loadAppInferenceSettings() {
  try {
    let payload = await requestJson(endpoints.appInferenceSettings);
    let fps = Number(payload?.production_inference_fps ?? payload?.detection_fps);
    const preferred = Number(getState().config.production_inference_fps);
    if (payload?.settings_source === "default" && Number.isFinite(preferred) && preferred > 0) {
      payload = await postJson(endpoints.appInferenceSettings, { detection_fps: preferred });
      fps = Number(payload?.production_inference_fps ?? payload?.detection_fps);
    }
    if (Number.isFinite(fps) && fps > 0) {
      const config = normalizeConfig({
        ...getState().config,
        production_inference_fps: fps,
      });
      updateState({ config });
      persistConfig(config);
      setValue(fields.inference_fps, fps);
    }
    return payload;
  } catch (_error) {
    // Some production apps are request-driven and do not expose a background FPS.
    return null;
  }
}

async function saveAppInferenceSettings(requestedFps) {
  const detectionFps = Math.min(30, Math.max(0.1, Number(requestedFps)));
  try {
    const result = await postJson(endpoints.appInferenceSettings, {
      detection_fps: detectionFps,
    });
    const applied = Number(result?.production_inference_fps ?? result?.detection_fps);
    return {
      skipped: false,
      appliedFps: Number.isFinite(applied) ? applied : detectionFps,
      message: Number.isFinite(applied)
        ? `生产推理与机器人推送已统一设为 ${applied.toFixed(applied >= 10 ? 1 : 2)} FPS`
        : "生产推理 FPS 已应用",
    };
  } catch (error) {
    if (error instanceof ApiError && [0, 404, 405, 502, 503].includes(error.status)) {
      return { skipped: true, appliedFps: detectionFps, message: "后台业务应用未提供推理 FPS 接口，仅应用模型验证刷新设置" };
    }
    throw error;
  }
}

function fill(config) {
  const displayFps = config.display_fps ?? intervalMsToFps(config.preview_refresh_interval_ms, 5);
  const inferenceFps = Number(config.production_inference_fps || intervalMsToFps(config.inference_interval_ms, 15));
  for (const [key, id] of Object.entries(fields)) {
    if (key === "display_fps") setValue(id, displayFps);
    else if (key === "status_refresh_fps") setValue(id, intervalMsToFps(config.status_refresh_interval_ms, 0.5));
    else if (key === "inference_fps") setValue(id, inferenceFps);
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
  const previewIntervalMs = Math.max(16, Math.round(1000 / Math.max(1, displayFps)));
  const inferenceFps = getNumber(fields.inference_fps, current.production_inference_fps || 15);
  const inferenceIntervalMs = fpsToIntervalMs(inferenceFps, current.inference_interval_ms || 500);
  const rgbProfile = getValue(fields.rgb_profile, current.rgb_profile || "orbbec:1280x720@30");
  const rgbParsed = parseProfile(rgbProfile, { resolution: current.camera_resolution || "1280x720", fps: current.camera_read_fps || 30 });
  return normalizeConfig({
    ...current,
    camera_type: "sdk_bridge",
    camera_model: getValue(fields.camera_model, "orbbec336l"),
    display_fps: displayFps,
    preview_refresh_interval_ms: previewIntervalMs,
    snapshot_refresh_interval_ms: previewIntervalMs,
    production_inference_fps: inferenceFps,
    inference_interval_ms: inferenceIntervalMs,
    status_refresh_interval_ms: fpsToIntervalMs(getNumber(fields.status_refresh_fps, intervalMsToFps(current.status_refresh_interval_ms, 0.5)), current.status_refresh_interval_ms || 2000),
    rgb_profile: rgbProfile,
    depth_profile: getValue(fields.depth_profile, current.depth_profile || "orbbec:1280x720@30"),
    depth_unit: getValue(fields.depth_unit, "mm"),
    rgb_source_preference: getValue(fields.rgb_source_preference, "auto"),
    flip_vertical: getValue(fields.flip_vertical, "false"),
    flip_horizontal: getValue(fields.flip_horizontal, "false"),
    rgb_order: getValue(fields.rgb_order, "bgr"),
    orbbec_serial: getValue(fields.orbbec_serial, ""),
    hp60c_config_path: getValue(fields.hp60c_config_path, ""),
    hp60c_fx: getNumber(fields.hp60c_fx, 0),
    hp60c_fy: getNumber(fields.hp60c_fy, 0),
    hp60c_cx: getNumber(fields.hp60c_cx, 0),
    hp60c_cy: getNumber(fields.hp60c_cy, 0),
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
    preprocess_backend_preference: "rga",
    task_view_preference: "auto",
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

function buildCameraPayload(config) {
  return {
    camera_model: config.camera_model,
    rgb_profile: config.rgb_profile,
    depth_profile: config.depth_profile,
    display_fps: config.display_fps,
    camera_jpeg_quality: config.camera_jpeg_quality,
    flip_vertical: config.flip_vertical,
    flip_horizontal: config.flip_horizontal,
    depth_unit: config.depth_unit,
    orbbec_serial: config.orbbec_serial,
    hp60c_config_path: config.hp60c_config_path,
    hp60c_fx: config.hp60c_fx,
    hp60c_fy: config.hp60c_fy,
    hp60c_cx: config.hp60c_cx,
    hp60c_cy: config.hp60c_cy,
    rgb_source_preference: config.rgb_source_preference,
    rgb_order: config.rgb_order,
    known_profiles: latestBridgeSettings?.profiles || null,
  };
}

function currentAlgorithmTask() {
  return String(latestAlgorithmSettings?.settings?.task_type || "").toLowerCase();
}

function buildAlgorithmPayload() {
  const task = currentAlgorithmTask();
  const ids = taskThresholdFields[task] || {};
  return {
    model_id: getValue(fields.algorithm_model, latestAlgorithmSettings?.selected_model?.model_id || ""),
    score_threshold: ids.score ? getNumber(ids.score, latestAlgorithmSettings?.settings?.score_threshold ?? 0.5) : null,
    nms_threshold: ids.nms ? getNumber(ids.nms, latestAlgorithmSettings?.settings?.nms_threshold ?? 0.45) : null,
    reload_runtime: true,
  };
}

async function saveAlgorithmSettings() {
  if (!latestAlgorithmSettings?.selected_model) return { skipped: true, message: "未选择模型" };
  const result = await postJson(endpoints.algorithmSettings, buildAlgorithmPayload());
  renderAlgorithmSettings(result);
  if (result.changed === false) return { skipped: true, message: "算法阈值未变化" };
  const reloaded = result.runtime_reload?.attempted ? (result.runtime_reload?.ok ? "，当前模型已重新加载" : "，但 Runtime 重新加载失败") : "";
  return { skipped: false, message: `算法阈值已写入 model.yaml${reloaded}` };
}

function open() {
  fill(getState().config);
  const modal = element("settings-modal");
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
  setNotice("Orbbec 336L 与 HP60C Bridge 可同时运行；相机型号保存后会切换 Runtime、采集、模型验证和生产画面。", "");
  loadCameraSettings();
  loadVisionBoxSettings();
  loadAlgorithmSettings();
  loadAppInferenceSettings();
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
  if (panelId === "algorithm-settings-panel") {
    loadAlgorithmSettings(getValue(fields.algorithm_model, ""));
    loadAppInferenceSettings();
  }
  if (panelId === "board-settings-panel") loadVisionBoxSettings();
  if (panelId === "camera-settings-panel") loadCameraSettings();
}

function markUnsaved() { showSaveStatus("有未保存的修改"); }

async function saveSettings() {
  let config = readConfigFromForm();
  const cameraModel = config.camera_model || "orbbec336l";
  showSaveStatus("正在保存设置...", "loading");
  const messages = [];

  try {
    const result = await postJson(endpoints.sdkBridgeSettings, buildCameraPayload(config));
    latestBridgeSettings = result;
    updateBridgeStatus(result);
    populateProfileSelect(fields.rgb_profile, result.profiles?.color || [], result.settings?.rgb_profile || config.rgb_profile, "未读取到 RGB profile");
    populateProfileSelect(fields.depth_profile, result.profiles?.depth || [], result.settings?.depth_profile || config.depth_profile, "未读取到 Depth profile");
    updateDepthMatchHint();
    updateCameraSpecificFields(cameraModel);
    if (result.changed === false) messages.push("相机设置未变化");
    else if (result.camera_switched) messages.push(`相机已切换为 ${cameraModel === "hp60c" ? "HP60C" : "Orbbec 336L"}，Runtime 图像源已重启`);
    else messages.push("相机 Bridge 参数已应用");
    if (result.camera_switched) {
      window.dispatchEvent(new CustomEvent("visionops:camera-switched", { detail: { camera_model: cameraModel } }));
    }
  } catch (error) {
    const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
    showSaveStatus(`保存失败：${detail}`, "error");
    setNotice(`相机设置未生效：${detail}`, "error");
    return;
  }

  try {
    const board = await saveVisionBoxSettings(config);
    messages.push(board.message);
  } catch (error) {
    const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
    showSaveStatus(`视觉盒子设置保存失败：${detail}`, "error");
    setNotice(`视觉盒子设置未生效：${detail}`, "error");
    return;
  }

  try {
    const alg = await saveAlgorithmSettings();
    messages.push(alg.message);
    const inference = await saveAppInferenceSettings(config.production_inference_fps);
    messages.push(inference.message);
    config = normalizeConfig({
      ...config,
      production_inference_fps: inference.appliedFps,
    });
  } catch (error) {
    const detail = error instanceof ApiError && error.body?.error?.message ? error.body.error.message : error.message;
    showSaveStatus(`算法设置保存失败：${detail}`, "error");
    setNotice(`算法设置未生效：${detail}`, "error");
    return;
  }

  persistConfig(config);
  updateState({ config });
  window.dispatchEvent(new CustomEvent("visionops:settings-saved", { detail: { config } }));
  showSaveStatus(messages.join("；"), "ok");
  setNotice(messages.join("；"), "ok");
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
  const cameraModelSelect = element(fields.camera_model);
  if (cameraModelSelect) cameraModelSelect.addEventListener("change", () => loadCameraSettings(cameraModelSelect.value));
  const algorithmModel = element(fields.algorithm_model);
  if (algorithmModel) algorithmModel.addEventListener("change", () => loadAlgorithmSettings(algorithmModel.value));
  const refreshProfiles = element("settings-refresh-profiles");
  if (refreshProfiles) refreshProfiles.addEventListener("click", () => {
    loadCameraSettings(getValue(fields.camera_model, ""));
    loadAlgorithmSettings(getValue(fields.algorithm_model, ""));
  });
  element("settings-reset").addEventListener("click", () => {
    fill(getState().savedConfig || getState().config);
    loadCameraSettings();
    loadVisionBoxSettings();
    loadAlgorithmSettings();
    loadAppInferenceSettings();
    showSaveStatus("已恢复到 Collector 默认、当前 Bridge 与当前模型配置，尚未保存");
  });
  element("settings-save").addEventListener("click", saveSettings);
}
