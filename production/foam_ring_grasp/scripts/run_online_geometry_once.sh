#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${VISIONOPS_PYTHON_BIN:-python3}"
CONFIG_PATH="${VISIONOPS_FOAM_RING_CONFIG:-${REPO_ROOT}/production/foam_ring_grasp/config/line.yaml}"
RUNTIME_URL="${VISIONOPS_FOAM_RING_RUNTIME_URL:-http://127.0.0.1:28081}"
OUTPUT_ROOT="${VISIONOPS_FOAM_RING_ONLINE_OUTPUT:-${REPO_ROOT}/data/foam_ring_online_geometry}"
GEOMETRY_MODE="${VISIONOPS_FOAM_RING_GEOMETRY_MODE:-}"

for path in \
  /dev/shm/visionops_orbbec336l_rgb \
  /dev/shm/visionops_orbbec336l_depth; do
  if [[ ! -r "${path}" ]]; then
    echo "[ERROR] current user cannot read ${path}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
EXTRA_ARGS=()
if [[ -n "${GEOMETRY_MODE}" ]]; then
  EXTRA_ARGS+=(--geometry-mode "${GEOMETRY_MODE}")
fi

exec "${PYTHON_BIN}" -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_validate \
  --config "${CONFIG_PATH}" \
  --runtime-url "${RUNTIME_URL}" \
  --output "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
