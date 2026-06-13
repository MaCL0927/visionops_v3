import { endpoints, postJson, requestBlob, requestJson } from "../api.js";
import { updateState } from "../state.js";
import { clearOverlay, drawInferenceOverlay } from "../render/overlay.js";

const image = document.getElementById("validate-image"), canvas = document.getElementById("validate-overlay"), empty = document.getElementById("validate-empty"), output = document.getElementById("validate-result");
let snapshotUrl = null, currentResult = null;

function showResult(result) {
  currentResult = result; updateState({ latestResult: result }); output.textContent = JSON.stringify(result, null, 2);
  const timing = result.timing || {}; for (const key of ["preprocess", "inference", "postprocess", "total"]) document.getElementById(`timing-${key}`).textContent = timing[`${key}_ms`] == null ? "--" : `${timing[`${key}_ms`]} ms`;
}

async function refreshValidationImage() {
  const blob = await requestBlob(`${endpoints.snapshot}?t=${Date.now()}`); if (snapshotUrl) URL.revokeObjectURL(snapshotUrl); snapshotUrl = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = snapshotUrl; }); empty.classList.add("hidden"); drawInferenceOverlay(canvas, image, currentResult);
}

export async function inferOnce() {
  try { const result = await postJson(endpoints.inferOnce); showResult(result); await refreshValidationImage(); }
  catch (error) { output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2); empty.classList.remove("hidden"); empty.textContent = "Runtime infer_once unavailable"; clearOverlay(canvas); }
}

export async function refreshLatestResult() {
  try { const result = await requestJson(endpoints.latestResult); showResult(result); await refreshValidationImage(); }
  catch (error) { output.textContent = JSON.stringify(error.body || { error: error.message }, null, 2); }
}

export function initValidate() { document.getElementById("validate-infer").addEventListener("click", inferOnce); document.getElementById("validate-refresh").addEventListener("click", refreshLatestResult); window.addEventListener("resize", () => currentResult && drawInferenceOverlay(canvas, image, currentResult)); }
