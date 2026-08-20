#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${VISIONOPS_PYTHON_BIN:-python3}"
CONFIG_PATH="${VISIONOPS_FOAM_RING_CONFIG:-${REPO_ROOT}/production/foam_ring_grasp/config/line.yaml}"
RUNTIME_URL="${VISIONOPS_FOAM_RING_RUNTIME_URL:-http://127.0.0.1:28081}"
LISTEN_HOST="${VISIONOPS_FOAM_RING_LISTEN_HOST:-}"
LISTEN_PORT="${VISIONOPS_FOAM_RING_LISTEN_PORT:-}"
GEOMETRY_MODE="${VISIONOPS_FOAM_RING_GEOMETRY_MODE:-first_valid}"

for path in \
  /dev/shm/visionops_orbbec336l_rgb \
  /dev/shm/visionops_orbbec336l_depth; do
  if [[ ! -r "${path}" ]]; then
    echo "[ERROR] current user cannot read ${path}" >&2
    exit 2
  fi
done

ARGS=(
  --config "${CONFIG_PATH}"
  --runtime-url "${RUNTIME_URL}"
  --geometry-mode "${GEOMETRY_MODE}"
)
if [[ -n "${LISTEN_HOST}" ]]; then
  ARGS+=(--host "${LISTEN_HOST}")
fi
if [[ -n "${LISTEN_PORT}" ]]; then
  ARGS+=(--port "${LISTEN_PORT}")
fi

cd "${REPO_ROOT}"

echo "[foam_ring] resolved config: ${CONFIG_PATH}"
echo "[foam_ring] production preflight..."
"${PYTHON_BIN}" \
  production/foam_ring_grasp/scripts/verify_production.py \
  --config "${CONFIG_PATH}"

echo "[foam_ring] starting online service"
exec "${PYTHON_BIN}" -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.service \
  "${ARGS[@]}" \
  "$@"