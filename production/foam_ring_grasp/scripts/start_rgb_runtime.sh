#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
MODEL_DIR="${1:-${VISIONOPS_FOAM_RING_MODEL_DIR:-${EDGE_ROOT}/models/rk3576-001_ring_seg_20260729_100731}}"

export VISIONOPS_EDGE_ROOT="${EDGE_ROOT}"
export VISIONOPS_FRAME_SOURCE="${VISIONOPS_FRAME_SOURCE:-shared_memory}"
export VISIONOPS_SHARED_MEMORY_NAME="${VISIONOPS_SHARED_MEMORY_NAME:-/visionops_orbbec336l_rgb}"
export VISIONOPS_SHARED_MEMORY_FALLBACK_HTTP="${VISIONOPS_SHARED_MEMORY_FALLBACK_HTTP:-true}"
export VISIONOPS_RUNTIME_PORT="${VISIONOPS_RUNTIME_PORT:-28081}"
export VISIONOPS_DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"
export VISIONOPS_CAMERA_FPS="${VISIONOPS_CAMERA_FPS:-30}"
export VISIONOPS_PREPROCESS_BACKEND="${VISIONOPS_PREPROCESS_BACKEND:-rga}"

SHM_FILE="/dev/shm/${VISIONOPS_SHARED_MEMORY_NAME#/}"
if [[ -e "${SHM_FILE}" && ! -r "${SHM_FILE}" ]]; then
  echo "[ERROR] Current user cannot read RGB shared memory: ${SHM_FILE}" >&2
  ls -l "${SHM_FILE}" >&2 || true
  echo "        Temporary test: sudo chmod 0664 ${SHM_FILE}" >&2
  echo "        Persistent fix: set VISIONOPS_ORBBEC336L_SHARED_MEMORY_MODE=0664" >&2
  echo "        and rebuild/restart visionops-orbbec336l-bridge.service." >&2
  exit 3
fi

exec "${EDGE_ROOT}/scripts/start_runtime.sh" "${MODEL_DIR}"
