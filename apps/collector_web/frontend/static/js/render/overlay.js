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


function roiCoordinates(roi, sourceWidth, sourceHeight) {
  if (!roi?.enabled) return null;
  const normalized = Array.isArray(roi.normalized_xyxy) ? roi.normalized_xyxy.map(Number) : null;
  if (normalized?.length === 4 && normalized.every(Number.isFinite)) {
    return [
      normalized[0] * sourceWidth,
      normalized[1] * sourceHeight,
      normalized[2] * sourceWidth,
      normalized[3] * sourceHeight,
    ];
  }
  const pixels = Array.isArray(roi.pixel_xyxy) ? roi.pixel_xyxy.map(Number) : null;
  if (pixels?.length === 4 && pixels.every(Number.isFinite)) return pixels;
  const values = [Number(roi.x1), Number(roi.y1), Number(roi.x2), Number(roi.y2)];
  if (values.every(Number.isFinite)) {
    return values.every((value) => value >= 0 && value <= 1)
      ? [values[0] * sourceWidth, values[1] * sourceHeight, values[2] * sourceWidth, values[3] * sourceHeight]
      : values;
  }
  return null;
}

function drawOutputRoi(ctx, roi, sourceWidth, sourceHeight, sx, sy, canvasWidth) {
  const coordinates = roiCoordinates(roi, sourceWidth, sourceHeight);
  if (!coordinates) return;
  const [x1, y1, x2, y2] = coordinates;
  if (!(x2 > x1 && y2 > y1)) return;
  ctx.save();
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = Math.max(2, Math.round(canvasWidth / 420));
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  ctx.setLineDash([]);
  ctx.font = `bold ${Math.max(14, Math.round(canvasWidth / 68))}px system-ui`;
  const label = "ROI 输出区域";
  const labelX = x1 * sx;
  const labelY = Math.max(0, y1 * sy - 25);
  const labelWidth = ctx.measureText(label).width + 16;
  ctx.fillStyle = "rgba(250, 204, 21, .94)";
  ctx.fillRect(labelX, labelY, labelWidth, 24);
  ctx.fillStyle = "#111827";
  ctx.fillText(label, labelX + 8, labelY + 17);
  ctx.restore();
}


function drawPlacementArrow(ctx, slot, sx, sy, strokeStyle) {
  const center = point(slot?.center_xy);
  if (!center) return;
  const angle = Number(slot.orientation_deg || 0) * Math.PI / 180;
  const bbox = Array.isArray(slot.bbox_xyxy) ? slot.bbox_xyxy.map(Number) : null;
  const baseLength = bbox?.length === 4
    ? Math.max(18, Math.min(Math.abs(bbox[2] - bbox[0]) * sx, Math.abs(bbox[3] - bbox[1]) * sy) * 0.28)
    : 30;
  const cx = center[0] * sx;
  const cy = center[1] * sy;
  const dx = Math.cos(angle) * baseLength;
  const dy = Math.sin(angle) * baseLength;
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.fillStyle = strokeStyle;
  ctx.lineWidth = Math.max(2, ctx.lineWidth);
  ctx.beginPath();
  ctx.moveTo(cx - dx, cy - dy);
  ctx.lineTo(cx + dx, cy + dy);
  ctx.stroke();
  const tipX = cx + dx;
  const tipY = cy + dy;
  const wing = Math.max(7, baseLength * 0.22);
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - Math.cos(angle - Math.PI / 5) * wing, tipY - Math.sin(angle - Math.PI / 5) * wing);
  ctx.lineTo(tipX - Math.cos(angle + Math.PI / 5) * wing, tipY - Math.sin(angle + Math.PI / 5) * wing);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawPlacementOverlay(ctx, placement, sx, sy, canvasWidth, canvasHeight) {
  if (!placement) return;
  const layer = Math.max(1, Number(placement.layer) || 1);
  const slots = Array.isArray(placement.slots) ? placement.slots : [];
  const nextSlotId = String(placement.next_slot_id || "");
  const fontSize = Math.max(14, Math.round(canvasWidth / 72));

  for (const slot of slots) {
    if (slot?.visible_mask === false || slot?.occupied === true) continue;
    const ring = Array.isArray(slot?.polygon) ? slot.polygon.map(point).filter(Boolean) : [];
    if (ring.length < 3) continue;
    const isTarget = String(slot.slot_id || "") === nextSlotId;
    const stroke = isTarget ? "#22c55e" : "#facc15";
    const fill = isTarget ? "rgba(34,197,94,.30)" : "rgba(250,204,21,.24)";
    drawRing(ctx, ring, sx, sy);
    ctx.save();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = Math.max(3, Math.round(canvasWidth / 380));
    ctx.setLineDash(isTarget ? [] : [12, 7]);
    ctx.stroke();
    ctx.setLineDash([]);
    drawPlacementArrow(ctx, slot, sx, sy, stroke);

    const anchorX = Math.min(...ring.map(([x]) => x)) * sx;
    const anchorY = Math.min(...ring.map(([, y]) => y)) * sy;
    ctx.font = `bold ${fontSize}px system-ui`;
    const orientation = slot.orientation_label || (Math.abs((Number((slot.template_orientation_deg ?? slot.orientation_deg) || 0) % 180)) < 45 ? "横向" : "竖向");
    const label = `L${layer}-${slot.slot_id || "P?"} ${orientation}${isTarget ? " · 下一位置" : ""}`;
    const labelWidth = ctx.measureText(label).width + 16;
    const labelHeight = fontSize + 10;
    ctx.fillStyle = isTarget ? "rgba(22,163,74,.94)" : "rgba(202,138,4,.94)";
    ctx.fillRect(anchorX, Math.max(0, anchorY - labelHeight), labelWidth, labelHeight);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, anchorX + 8, Math.max(fontSize + 2, anchorY - 7));
    ctx.restore();
  }

  ctx.save();
  ctx.font = `bold ${Math.max(16, Math.round(canvasWidth / 62))}px system-ui`;
  let summary = "等待检测托盘";
  const occupied = Number(placement.occupied_count || 0);
  const total = Number(placement.slot_count || slots.length || 0);
  const state = String(placement.state || "");
  if (state === "STACK_COMPLETE" || placement.stack_complete === true) {
    summary = `堆垛完成 · ${layer}层`;
  } else if (state.includes("SETTLING")) {
    const capture = placement.transition?.baseline_capture || {};
    summary = `第${layer}层已放满 · 等待画面稳定 ${Number(capture.settled_frames || 0)}/${Number(capture.required_settle_frames || 0)}`;
  } else if (state.includes("CAPTURING_BASELINE")) {
    const capture = placement.transition?.baseline_capture || {};
    summary = `第${layer}层已放满 · 采集深度基准 ${Number(capture.captured_frames || 0)}/${Number(capture.required_frames || 0)}`;
  } else if (state.includes("WAIT_DEPTH")) {
    summary = `第${layer}层等待深度图`;
  } else if (state !== "WAIT_TRAY") {
    summary = placement.layer_complete ? `第${layer}层已放满 ${occupied}/${total}` : `第${layer}层摆放 ${occupied}/${total}`;
  }
  const width = ctx.measureText(summary).width + 24;
  const height = Math.max(34, Math.round(canvasHeight / 22));
  ctx.fillStyle = placement.stack_complete || placement.layer_complete ? "rgba(22,163,74,.94)" : "rgba(17,24,39,.86)";
  ctx.fillRect(12, 12, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillText(summary, 24, 12 + Math.round(height * 0.68));
  ctx.restore();
}


function drawBoxGraspPoint(ctx, rawPoint, label, sx, sy, fillStyle, canvasWidth) {
  const parsed = point(rawPoint);
  if (!parsed) return;
  const x = parsed[0] * sx;
  const y = parsed[1] * sy;
  const radius = Math.max(5, Math.round(canvasWidth / 160));
  ctx.save();
  ctx.fillStyle = fillStyle;
  ctx.strokeStyle = "rgba(17,24,39,.85)";
  ctx.lineWidth = Math.max(1, Math.round(canvasWidth / 800));
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.font = `bold ${Math.max(12, Math.round(canvasWidth / 78))}px system-ui`;
  const text = `${label}(${Math.round(parsed[0])},${Math.round(parsed[1])})`;
  const width = ctx.measureText(text).width + 10;
  const height = Math.max(20, Math.round(canvasWidth / 48));
  let tx = x + radius + 4;
  if (tx + width > ctx.canvas.width) tx = Math.max(0, x - radius - width - 4);
  let ty = y - height - 3;
  if (ty < 0) ty = Math.min(ctx.canvas.height - height, y + radius + 3);
  ctx.fillStyle = "rgba(17,24,39,.78)";
  ctx.fillRect(tx, ty, width, height);
  ctx.fillStyle = fillStyle;
  ctx.fillText(text, tx + 5, ty + Math.round(height * 0.72));
  ctx.restore();
}

function drawBoxGraspOverlay(ctx, grasp, sx, sy, canvasWidth) {
  const items = Array.isArray(grasp?.items) ? grasp.items : [];
  for (const item of items) {
    const contour = Array.isArray(item?.contour_px) ? item.contour_px.map(point).filter(Boolean) : [];
    if (contour.length >= 3) {
      drawRing(ctx, contour, sx, sy);
      ctx.save();
      ctx.strokeStyle = "rgba(34,211,238,.95)";
      ctx.lineWidth = Math.max(1, Math.round(canvasWidth / 600));
      ctx.stroke();
      ctx.restore();
    }
    const corners = item?.corners_px || {};
    const quad = [corners.top_left, corners.top_right, corners.bottom_right, corners.bottom_left]
      .map(point).filter(Boolean);
    if (quad.length === 4) {
      drawRing(ctx, quad, sx, sy);
      ctx.save();
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = Math.max(3, Math.round(canvasWidth / 320));
      ctx.stroke();
      ctx.restore();
    }
    drawBoxGraspPoint(ctx, corners.top_left, "TL", sx, sy, "#3b82f6", canvasWidth);
    drawBoxGraspPoint(ctx, corners.top_right, "TR", sx, sy, "#f97316", canvasWidth);
    drawBoxGraspPoint(ctx, corners.bottom_right, "BR", sx, sy, "#ef4444", canvasWidth);
    drawBoxGraspPoint(ctx, corners.bottom_left, "BL", sx, sy, "#06b6d4", canvasWidth);
    drawBoxGraspPoint(ctx, item?.center_px, "C", sx, sy, "#facc15", canvasWidth);
    drawBoxGraspPoint(ctx, item?.grasp_points_px?.left_mid, "L", sx, sy, "#fde047", canvasWidth);
    drawBoxGraspPoint(ctx, item?.grasp_points_px?.right_mid, "R", sx, sy, "#fde047", canvasWidth);
  }
}

export function clearOverlay(canvas) { canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height); }

export function drawInferenceOverlay(canvas, image, result) {
  if (!image.naturalWidth || !image.naturalHeight) return;
  const rect = image.getBoundingClientRect();
  const parentRect = image.parentElement?.getBoundingClientRect() || rect;
  canvas.width = Math.max(1, Math.round(rect.width)); canvas.height = Math.max(1, Math.round(rect.height));
  canvas.style.left = `${Math.max(0, rect.left - parentRect.left)}px`;
  canvas.style.top = `${Math.max(0, rect.top - parentRect.top)}px`;
  canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
  const sourceWidth = Number(result?.image?.width) || image.naturalWidth;
  const sourceHeight = Number(result?.image?.height) || image.naturalHeight;
  const sx = canvas.width / sourceWidth; const sy = canvas.height / sourceHeight;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 480));
  ctx.font = `${Math.max(16, Math.round(canvas.width / 60))}px system-ui`;
  const overlay = getState().config.overlay || {};
  const maskOpacity = clamp(Number(overlay.mask_opacity ?? 0.28), 0, 1);

  drawOutputRoi(ctx, result?.roi, sourceWidth, sourceHeight, sx, sy, canvas.width);
  drawPlacementOverlay(ctx, result?.placement, sx, sy, canvas.width, canvas.height);
  drawBoxGraspOverlay(ctx, result?.box_grasp, sx, sy, canvas.width);

  const classifications = Array.isArray(result?.classifications) ? result.classifications : [];
  if (classifications.length && (!Array.isArray(result?.detections) || !result.detections.length)) {
    const top = classifications[0] || {};
    const name = top.class_name ?? top.label ?? top.class_id ?? "class";
    const score = Number(top.score || 0).toFixed(2);
    drawLabel(ctx, 8, Math.max(28, Math.round(canvas.height * 0.08)), `${name} ${score}`, canvas.width);
  }

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
