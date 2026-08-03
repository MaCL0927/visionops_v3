#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
cd "${EDGE_ROOT}"

VENV="${VISIONOPS_VENV:-${EDGE_ROOT}/venv}"
PYTHON_BIN="${VENV}/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] VisionOps v3 venv 不存在: ${PYTHON_BIN}" >&2
  echo "        请先运行: sudo bash ${EDGE_ROOT}/scripts/setup_edge_env.sh" >&2
  exit 1
fi

MODEL_DIR="${1:-${MODEL_DIR:-/opt/visionops_v3/models/test_rknn_model}}"
RUNTIME_BIN="${VISIONOPS_RUNTIME_BIN:-./build-rknn/edge/runtime_cpp/visionops_runtime_mock}"
DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"
PORT="${VISIONOPS_RUNTIME_PORT:-28081}"

CAMERA_SELECTION_FILE="${VISIONOPS_CAMERA_SELECTION_FILE:-/opt/visionops_v3/configs/runtime/generated/active_camera.json}"
if [[ ! -f "${CAMERA_SELECTION_FILE}" ]]; then
  echo "[WARN] camera selection file does not exist: ${CAMERA_SELECTION_FILE}" >&2
  echo "       camera_selection.py will use its built-in orbbec336l defaults." >&2
fi

read -r ACTIVE_CAMERA_MODEL ACTIVE_BRIDGE_URL ACTIVE_SNAPSHOT_PATH ACTIVE_HEALTH_PATH <<< "$({
  VISIONOPS_CAMERA_SELECTION_FILE="${CAMERA_SELECTION_FILE}" \
  "${PYTHON_BIN}" - <<'PY'
from edge.camera_bridge.camera_selection import active_camera_spec

spec = active_camera_spec()
print(
    spec["camera_model"],
    spec["base_url"],
    spec.get("snapshot_path", "/stream/snapshot.jpg"),
    spec.get("health_path", "/health"),
)
PY
})"

CAMERA_BRIDGE_URL="${VISIONOPS_CAMERA_BRIDGE_URL_OVERRIDE:-}"
if [[ -z "${CAMERA_BRIDGE_URL}" ]]; then
  CAMERA_BRIDGE_URL="${VISIONOPS_HP60C_URL:-${ACTIVE_BRIDGE_URL}}"
fi

FRAME_SOURCE="${VISIONOPS_FRAME_SOURCE:-auto}"
if [[ -z "${FRAME_SOURCE}" || "${FRAME_SOURCE}" == "auto" ]]; then
  if [[ "${ACTIVE_CAMERA_MODEL}" == "orbbec336l" ]]; then
    FRAME_SOURCE="shared_memory"
  else
    FRAME_SOURCE="hp60c_bridge"
  fi
fi

SHARED_MEMORY_NAME="${VISIONOPS_SHARED_MEMORY_NAME:-/visionops_orbbec336l_rgb}"
SHARED_MEMORY_FALLBACK_HTTP="${VISIONOPS_SHARED_MEMORY_FALLBACK_HTTP:-true}"
CAMERA_FPS="${VISIONOPS_CAMERA_FPS:-30}"
PREPROCESS_BACKEND="${VISIONOPS_PREPROCESS_BACKEND:-rga}"
RGA_MODE="${VISIONOPS_RGA_MODE:-resize_rgb}"
STALE_FRAME_TIMEOUT_MS="${VISIONOPS_STALE_FRAME_TIMEOUT_MS:-3000}"
RECONNECT_FAILURE_THRESHOLD="${VISIONOPS_RECONNECT_FAILURE_THRESHOLD:-3}"
RECONNECT_INITIAL_MS="${VISIONOPS_RECONNECT_INITIAL_MS:-200}"
RECONNECT_MAX_MS="${VISIONOPS_RECONNECT_MAX_MS:-2000}"

if [[ ! -x "${RUNTIME_BIN}" ]]; then
  echo "[ERROR] Runtime binary 不存在或不可执行: ${RUNTIME_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.rknn" || ! -f "${MODEL_DIR}/model.yaml" ]]; then
  echo "[ERROR] 模型包必须包含 model.rknn 和 model.yaml: ${MODEL_DIR}" >&2
  exit 1
fi

case "${FRAME_SOURCE}" in
  shared_memory|hp60c_bridge|hp60c|v4l2|test_image|mock) ;;
  *)
    echo "[ERROR] 不支持的 VISIONOPS_FRAME_SOURCE: ${FRAME_SOURCE}" >&2
    exit 1
    ;;
esac

printf '%s\n' \
  "[INFO] camera selection file: ${CAMERA_SELECTION_FILE}" \
  "[INFO] active camera model: ${ACTIVE_CAMERA_MODEL}" \
  "[INFO] active camera bridge: ${CAMERA_BRIDGE_URL}" \
  "[INFO] frame source: ${FRAME_SOURCE}" \
  "[INFO] shared RGB: ${SHARED_MEMORY_NAME} (HTTP fallback=${SHARED_MEMORY_FALLBACK_HTTP})" \
  "[INFO] model dir: ${MODEL_DIR}" \
  "[INFO] runtime endpoint: 0.0.0.0:${PORT}"

exec "${RUNTIME_BIN}" \
  --backend rknn \
  --preprocess-backend "${PREPROCESS_BACKEND}" \
  --rga-mode "${RGA_MODE}" \
  --frame-source "${FRAME_SOURCE}" \
  --hp60c-url "${CAMERA_BRIDGE_URL}" \
  --hp60c-snapshot-path "${ACTIVE_SNAPSHOT_PATH}" \
  --hp60c-health-path "${ACTIVE_HEALTH_PATH}" \
  --shared-memory-name "${SHARED_MEMORY_NAME}" \
  --shared-memory-fallback-http "${SHARED_MEMORY_FALLBACK_HTTP}" \
  --camera-fps "${CAMERA_FPS}" \
  --enable-camera-thread true \
  --stale-frame-timeout-ms "${STALE_FRAME_TIMEOUT_MS}" \
  --camera-reconnect-failure-threshold "${RECONNECT_FAILURE_THRESHOLD}" \
  --camera-reconnect-initial-ms "${RECONNECT_INITIAL_MS}" \
  --camera-reconnect-max-ms "${RECONNECT_MAX_MS}" \
  --model-dir "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device-id "${DEVICE_ID}"
