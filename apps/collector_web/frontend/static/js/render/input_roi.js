function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function positiveInt(value, fallback = 0) {
  const number = Math.round(Number(value));
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function parseSize(raw, fallbackWidth = 0, fallbackHeight = 0) {
  if (Array.isArray(raw) && raw.length >= 2) {
    return {
      width: positiveInt(raw[0], fallbackWidth),
      height: positiveInt(raw[1], fallbackHeight),
    };
  }
  if (raw && typeof raw === "object") {
    return {
      width: positiveInt(raw.width, fallbackWidth),
      height: positiveInt(raw.height, fallbackHeight),
    };
  }
  const single = positiveInt(raw, 0);
  if (single > 0) return { width: single, height: single };
  return { width: fallbackWidth, height: fallbackHeight };
}

function parsePixelRoi(raw, fullWidth, fullHeight, enabled) {
  const values = Array.isArray(raw) ? raw.map(Number) : [];
  if (values.length === 4 && values.every(Number.isFinite) && values[2] > values[0] && values[3] > values[1]) {
    return values;
  }
  return [0, 0, fullWidth, fullHeight];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function resolveInputRoiVisualState(result) {
  const inputRoi = result?.input_roi || result?.debug?.input_roi || {};
  const full = parseSize(
    inputRoi.full_resolution || result?.image,
    positiveInt(result?.image?.width, 0),
    positiveInt(result?.image?.height, 0),
  );
  const modelInput = parseSize(
    result?.model?.input_size || result?.loaded_model?.input_size,
    640,
    640,
  );
  const enabled = inputRoi.enabled === true;
  const pixel = parsePixelRoi(inputRoi.pixel_xyxy, full.width, full.height, enabled);
  const x0 = clamp(pixel[0], 0, full.width || pixel[2]);
  const y0 = clamp(pixel[1], 0, full.height || pixel[3]);
  const x1 = clamp(pixel[2], x0, full.width || pixel[2]);
  const y1 = clamp(pixel[3], y0, full.height || pixel[3]);
  const crop = parseSize(inputRoi.crop_resolution, Math.round(x1 - x0), Math.round(y1 - y0));
  const debug = result?.debug || {};
  const letterbox = debug.letterbox || {};
  const timing = result?.timing || {};
  const backend = String(debug.preprocess_backend_active || debug.preprocess_backend_requested || "--");
  const resizeMode = String(result?.model?.runtime_preprocess || debug.preprocess_mode || "letterbox");
  const rgaUsed = debug.rga_used === true;
  const fused = debug.rga_fused_crop_resize === true;
  return {
    enabled,
    coordinateSpace: String(inputRoi.coordinate_space || "runtime_snapshot"),
    full,
    pixel: [x0, y0, x1, y1],
    crop,
    modelInput,
    scaledFromNormalized: inputRoi.scaled_from_normalized === true,
    backend,
    resizeMode,
    rgaUsed,
    fused,
    rgaMode: String(debug.rga_mode || "--"),
    warning: debug.preprocess_warning || null,
    letterbox: {
      scale: finiteNumber(letterbox.scale),
      scaleX: finiteNumber(letterbox.scale_x),
      scaleY: finiteNumber(letterbox.scale_y),
      padX: finiteNumber(letterbox.pad_x),
      padY: finiteNumber(letterbox.pad_y),
    },
    timing: {
      resolveMs: finiteNumber(timing.input_roi_resolve_ms),
      cropResizeMs: finiteNumber(timing.crop_resize_ms ?? timing.preprocess_ms),
      preprocessMs: finiteNumber(timing.preprocess_ms),
      inferenceMs: finiteNumber(timing.inference_ms),
      totalMs: finiteNumber(timing.total_ms),
    },
  };
}

function formatMs(value, digits = 3) {
  return Number.isFinite(value) ? `${value.toFixed(value >= 10 ? 2 : digits)} ms` : "--";
}

function createMetric(label) {
  const item = document.createElement("div");
  const name = document.createElement("span");
  const strong = document.createElement("b");
  const small = document.createElement("em");
  name.textContent = label;
  item.append(name, strong, small);
  return { item, strong, small };
}

const METRIC_LABELS = [
  "模型输入 ROI",
  "模型输入尺寸",
  "预处理后端",
  "ROI 解析",
  "裁剪 + 缩放",
  "NPU 推理",
];

function metricNodes(container) {
  if (Array.isArray(container.__visionopsInputRoiMetrics)) {
    return container.__visionopsInputRoiMetrics;
  }
  const nodes = METRIC_LABELS.map(createMetric);
  container.replaceChildren(...nodes.map((node) => node.item));
  container.__visionopsInputRoiMetrics = nodes;
  return nodes;
}

function updateMetric(node, value, detail = "") {
  node.strong.textContent = value;
  node.small.textContent = detail;
  node.small.hidden = !detail;
}

export function renderInputRoiDiagnostics(container, result) {
  if (!container) return null;
  if (!result) {
    container.replaceChildren();
    delete container.__visionopsInputRoiMetrics;
    container.classList.add("inactive");
    container.classList.remove("warning");
    return null;
  }
  const state = resolveInputRoiVisualState(result);
  container.classList.toggle("inactive", !state.enabled);
  container.classList.toggle("warning", Boolean(state.warning));

  const [x0, y0, x1, y1] = state.pixel.map((value) => Math.round(value));
  const roiValue = state.enabled ? `${state.crop.width}×${state.crop.height}` : "全画面";
  const roiDetail = state.enabled ? `[${x0}, ${y0}, ${x1}, ${y1}]` : `${state.full.width}×${state.full.height}`;
  const backendValue = state.rgaUsed
    ? (state.fused ? "RGA 融合裁剪" : "RGA")
    : state.backend.toUpperCase();
  const backendDetail = state.rgaUsed ? state.rgaMode : (state.warning ? "已回退" : "");
  const nodes = metricNodes(container);

  updateMetric(nodes[0], roiValue, roiDetail);
  updateMetric(nodes[1], `${state.modelInput.width}×${state.modelInput.height}`, `${state.resizeMode}${state.scaledFromNormalized ? " · 归一化缩放" : ""}`);
  updateMetric(nodes[2], backendValue, backendDetail);
  updateMetric(nodes[3], formatMs(state.timing.resolveMs), state.coordinateSpace);
  updateMetric(nodes[4], formatMs(state.timing.cropResizeMs), state.fused ? "单次 RGA 操作" : "预处理路径");
  updateMetric(nodes[5], formatMs(state.timing.inferenceMs), state.warning ? String(state.warning) : "");
  return state;
}

export function drawModelInputPreview(canvas, image, result) {
  if (!canvas || !image?.naturalWidth || !image?.naturalHeight || !result) return null;
  const state = resolveInputRoiVisualState(result);
  const modelWidth = positiveInt(state.modelInput.width, 640);
  const modelHeight = positiveInt(state.modelInput.height, 640);
  canvas.width = modelWidth;
  canvas.height = modelHeight;
  canvas.style.aspectRatio = `${modelWidth} / ${modelHeight}`;

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, modelWidth, modelHeight);

  const [x0, y0, x1, y1] = state.pixel;
  const cropWidth = Math.max(1, x1 - x0);
  const cropHeight = Math.max(1, y1 - y0);
  const sourceScaleX = image.naturalWidth / Math.max(1, state.full.width || image.naturalWidth);
  const sourceScaleY = image.naturalHeight / Math.max(1, state.full.height || image.naturalHeight);
  const sourceX = x0 * sourceScaleX;
  const sourceY = y0 * sourceScaleY;
  const sourceWidth = cropWidth * sourceScaleX;
  const sourceHeight = cropHeight * sourceScaleY;

  let drawWidth;
  let drawHeight;
  let drawX;
  let drawY;
  const scaleX = state.letterbox.scaleX;
  const scaleY = state.letterbox.scaleY;
  const padX = state.letterbox.padX;
  const padY = state.letterbox.padY;
  if (state.resizeMode === "resize") {
    drawWidth = modelWidth;
    drawHeight = modelHeight;
    drawX = 0;
    drawY = 0;
  } else if ([scaleX, scaleY, padX, padY].every(Number.isFinite) && scaleX > 0 && scaleY > 0) {
    drawWidth = cropWidth * scaleX;
    drawHeight = cropHeight * scaleY;
    drawX = padX;
    drawY = padY;
  } else {
    const scale = Math.min(modelWidth / cropWidth, modelHeight / cropHeight);
    drawWidth = cropWidth * scale;
    drawHeight = cropHeight * scale;
    drawX = (modelWidth - drawWidth) / 2;
    drawY = (modelHeight - drawHeight) / 2;
  }

  ctx.drawImage(
    image,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    drawX,
    drawY,
    drawWidth,
    drawHeight,
  );
  ctx.strokeStyle = "rgba(34, 211, 238, .95)";
  ctx.lineWidth = Math.max(2, Math.round(modelWidth / 320));
  ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
  ctx.restore();
  return state;
}

export function inputRoiPreviewDescription(state) {
  if (!state) return "等待 Runtime 结果";
  const [x0, y0, x1, y1] = state.pixel.map((value) => Math.round(value));
  const range = state.enabled ? `[${x0}, ${y0}, ${x1}, ${y1}]` : "完整画面";
  return `${range} → ${state.modelInput.width}×${state.modelInput.height} ${state.resizeMode}`;
}
