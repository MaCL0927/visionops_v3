function loadSnapshot(kind) {
  const image = document.getElementById(`calibration-${kind}-image`);
  const empty = document.getElementById(`calibration-${kind}-empty`);
  image.removeAttribute("src");
  empty.classList.remove("hidden");
  empty.textContent = kind === "position" ? "位置校验占位图：后续再接实时预览" : "光照校验占位图：后续再接实时预览";
  return Promise.resolve(true);
}

function activateStep(kind) {
  document.querySelectorAll("[data-calibration-step]").forEach((button) => button.classList.toggle("active", button.dataset.calibrationStep === kind));
  document.getElementById("calibration-position").classList.toggle("active", kind === "position");
  document.getElementById("calibration-light").classList.toggle("active", kind === "light");
  document.getElementById("calibration-tip").textContent = kind === "position" ? "先调整摄像头，直到画面主体与半透明标准框重合。" : "观察实时画面的亮度、阴影与反光，当前指标为占位状态。";
  loadSnapshot(kind);
}

export async function refreshCalibration(kind = null) {
  const active = kind || (document.getElementById("calibration-light").classList.contains("active") ? "light" : "position");
  return loadSnapshot(active);
}

export function initCalibration() {
  document.querySelectorAll("[data-calibration-step]").forEach((button) => button.addEventListener("click", () => activateStep(button.dataset.calibrationStep)));
  document.getElementById("calibration-position-refresh").addEventListener("click", () => loadSnapshot("position"));
  document.getElementById("start-light-check").addEventListener("click", async () => {
    const ok = await loadSnapshot("light");
    document.getElementById("light-status").textContent = ok ? "光照校验完成" : "画面不可达";
    document.getElementById("light-brightness").textContent = ok ? "待算法接入" : "--";
    document.getElementById("light-result").textContent = ok ? "人工确认" : "unreachable";
  });
}
