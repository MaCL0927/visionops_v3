function point(value) { return Array.isArray(value) && value.length >= 2 ? [Number(value[0]), Number(value[1])] : null; }

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
  ctx.lineWidth = 2; ctx.font = "12px system-ui";
  for (const detection of result?.detections || []) {
    const box = detection.bbox_xyxy;
    if (Array.isArray(box) && box.length === 4) {
      const x = Number(box[0]) * sx, y = Number(box[1]) * sy;
      const width = (Number(box[2]) - Number(box[0])) * sx, height = (Number(box[3]) - Number(box[1])) * sy;
      ctx.strokeStyle = "#49e3b1"; ctx.strokeRect(x, y, width, height);
      const label = `${detection.class_name ?? detection.class_id ?? "object"} ${Number(detection.score || 0).toFixed(2)}`;
      const textWidth = ctx.measureText(label).width + 10; ctx.fillStyle = "rgba(9,121,105,.9)"; ctx.fillRect(x, Math.max(0, y - 20), textWidth, 20); ctx.fillStyle = "#fff"; ctx.fillText(label, x + 5, Math.max(14, y - 6));
    }
    const points = detection?.obb?.points;
    if (Array.isArray(points) && points.length >= 3) {
      const normalized = points.map(point).filter(Boolean);
      if (normalized.length >= 3) { ctx.beginPath(); ctx.strokeStyle = "#ffb454"; normalized.forEach(([px, py], index) => index ? ctx.lineTo(px * sx, py * sy) : ctx.moveTo(px * sx, py * sy)); ctx.closePath(); ctx.stroke(); }
    }
  }
}
