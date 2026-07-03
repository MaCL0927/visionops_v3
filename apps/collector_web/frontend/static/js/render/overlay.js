import { getState } from "../state.js";

function point(value) { return Array.isArray(value) && value.length >= 2 ? [Number(value[0]), Number(value[1])] : null; }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function polygonRings(mask) {
  const polygon = mask?.polygon;
  if (!Array.isArray(polygon)) return [];
  if (polygon.length >= 3 && Array.isArray(polygon[0]) && typeof polygon[0][0] === "number") {
    return [polygon.map(point).filter(Boolean)];
  }
  return polygon
    .map((ring) => Array.isArray(ring) ? ring.map(point).filter(Boolean) : [])
    .filter((ring) => ring.length >= 3);
}

function drawRing(ctx, ring, sx, sy) {
  ctx.beginPath();
  ring.forEach(([px, py], index) => index ? ctx.lineTo(px * sx, py * sy) : ctx.moveTo(px * sx, py * sy));
  ctx.closePath();
}

function drawLabel(ctx, x, y, label, canvasWidth) {
  const labelHeight = Math.max(24, Math.round(canvasWidth / 42));
  const textWidth = ctx.measureText(label).width + 16;
  const labelY = Math.max(0, y - labelHeight);
  ctx.fillStyle = "rgba(9,121,105,.92)";
  ctx.fillRect(x, labelY, textWidth, labelHeight);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x + 8, Math.max(labelHeight - 7, y - 8));
}

function drawBBox(ctx, box, sx, sy, strokeStyle = "#49e3b1") {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const x = Number(box[0]) * sx;
  const y = Number(box[1]) * sy;
  const width = (Number(box[2]) - Number(box[0])) * sx;
  const height = (Number(box[3]) - Number(box[1])) * sy;
  if (![x, y, width, height].every(Number.isFinite)) return null;
  ctx.strokeStyle = strokeStyle;
  ctx.strokeRect(x, y, width, height);
  return { x, y, width, height };
}

function drawCenter(ctx, detection, sx, sy) {
  const center = point(detection.center_xy) || point([detection.obb?.cx, detection.obb?.cy]);
  if (!center) return;
  const [cx, cy] = center;
  ctx.beginPath();
  ctx.fillStyle = "#ff4d4f";
  ctx.arc(cx * sx, cy * sy, 4, 0, Math.PI * 2);
  ctx.fill();
}

export function clearOverlay(canvas) { canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height); }

export function drawInferenceOverlay(canvas, image, result) {
  if (!image.naturalWidth || !image.naturalHeight) return;
  const rect = image.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width)); canvas.height = Math.max(1, Math.round(rect.height));
  canvas.style.left = `${image.offsetLeft}px`; canvas.style.top = `${image.offsetTop}px`;
  canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
  const sourceWidth = Number(result?.image?.width) || image.naturalWidth;
  const sourceHeight = Number(result?.image?.height) || image.naturalHeight;
  const sx = canvas.width / sourceWidth; const sy = canvas.height / sourceHeight;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 480));
  ctx.font = `${Math.max(16, Math.round(canvas.width / 60))}px system-ui`;
  const overlay = getState().config.overlay || {};
  const maskOpacity = clamp(Number(overlay.mask_opacity ?? 0.28), 0, 1);

  for (const detection of result?.detections || []) {
    const hasObb = Array.isArray(detection?.obb?.points) && detection.obb.points.length >= 3;
    const hasMask = Boolean(detection?.mask);
    const isSeg = result?.task_type === "segmentation" || hasMask;
    const isObb = result?.task_type === "obb" || hasObb;

    let labelAnchor = null;

    if (isSeg && overlay.show_segmentation_mask !== false) {
      const rings = polygonRings(detection.mask);
      for (const ring of rings) {
        drawRing(ctx, ring, sx, sy);
        ctx.fillStyle = `rgba(73, 227, 177, ${maskOpacity})`;
        ctx.fill();
        ctx.strokeStyle = "#16a085";
        ctx.stroke();
      }
    }

    const shouldDrawBBox = Array.isArray(detection.bbox_xyxy) && (
      (isObb && overlay.show_obb_bbox === true) ||
      (isSeg && overlay.show_segmentation_bbox !== false) ||
      (!isObb && !isSeg && overlay.show_detection_bbox !== false)
    );
    if (shouldDrawBBox) {
      const boxDrawn = drawBBox(ctx, detection.bbox_xyxy, sx, sy, isObb ? "#8cc9ff" : "#49e3b1");
      if (boxDrawn) labelAnchor = boxDrawn;
    }

    if (isObb && overlay.show_obb_rotated !== false) {
      const normalized = detection.obb.points.map(point).filter(Boolean);
      if (normalized.length >= 3) {
        drawRing(ctx, normalized, sx, sy);
        ctx.strokeStyle = "#ffb454";
        ctx.stroke();
        const xs = normalized.map(([px]) => px * sx);
        const ys = normalized.map(([, py]) => py * sy);
        labelAnchor = labelAnchor || { x: Math.min(...xs), y: Math.min(...ys) };
      }
    }

    if (overlay.show_centers !== false) drawCenter(ctx, detection, sx, sy);

    if (overlay.show_labels !== false) {
      const label = `${detection.class_name ?? detection.class_id ?? "object"} ${Number(detection.score || 0).toFixed(2)}`;
      let x = 0, y = 0;
      if (labelAnchor) { x = labelAnchor.x; y = labelAnchor.y; }
      else if (Array.isArray(detection.bbox_xyxy) && detection.bbox_xyxy.length === 4) { x = Number(detection.bbox_xyxy[0]) * sx; y = Number(detection.bbox_xyxy[1]) * sy; }
      drawLabel(ctx, x, y, label, canvas.width);
    }
  }
}
