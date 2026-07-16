#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
cd "${EDGE_ROOT}"

MODEL_DIR="${1:-${MODEL_DIR:-/opt/visionops_v3/models/test_rknn_model}}"
RUNTIME_BIN="${VISIONOPS_RUNTIME_BIN:-./build-rknn/edge/runtime_cpp/visionops_runtime_mock}"
DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"
PORT="${VISIONOPS_RUNTIME_PORT:-28081}"

CAMERA_SELECTION_FILE="${VISIONOPS_CAMERA_SELECTION_FILE:-/opt/visionops_v3/config/active_camera.json}"

ACTIVE_BRIDGE_URL="$(
  VISIONOPS_CAMERA_SELECTION_FILE="${CAMERA_SELECTION_FILE}" \
  python3 - <<'PY'
from edge.camera_bridge.camera_selection import active_camera_spec

spec = active_camera_spec()
print(spec["base_url"])
PY
)"

CAMERA_BRIDGE_URL="${VISIONOPS_CAMERA_BRIDGE_URL_OVERRIDE:-}"
if [[ -z "${CAMERA_BRIDGE_URL}" ]]; then
  CAMERA_BRIDGE_URL="${VISIONOPS_HP60C_URL:-${ACTIVE_BRIDGE_URL}}"
fi

echo "[INFO] camera selection file: ${CAMERA_SELECTION_FILE}"
echo "[INFO] active camera bridge: ${CAMERA_BRIDGE_URL}"

exec "${RUNTIME_BIN}" \
  --backend rknn \
  --preprocess-backend rga \
  --rga-mode resize_rgb \
  --frame-source hp60c_bridge \
  --hp60c-url "${CAMERA_BRIDGE_URL}" \
  --hp60c-snapshot-path /stream/snapshot.jpg \
  --hp60c-health-path /health \
  --model-dir "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device-id "${DEVICE_ID}"