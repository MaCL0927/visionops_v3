#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 server Web/API launcher.
# Usage:
#   bash scripts/start_server_api.sh
# Environment overrides:
#   VISIONOPS_SERVER_HOST, VISIONOPS_SERVER_PORT, VISIONOPS_SERVER_DATA_ROOT,
#   VISIONOPS_SERVER_INCOMING_ROOT, VISIONOPS_SERVER_PUBLISH_ROOT,
#   VISIONOPS_CONDA_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${VISIONOPS_CONDA_ENV:-visionops}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
fi

HOST="${VISIONOPS_SERVER_HOST:-0.0.0.0}"
PORT="${VISIONOPS_SERVER_PORT:-18100}"
DATA_ROOT="${VISIONOPS_SERVER_DATA_ROOT:-${PROJECT_ROOT}/server_data}"
INCOMING_ROOT="${VISIONOPS_SERVER_INCOMING_ROOT:-${DATA_ROOT}/incoming}"
PUBLISH_ROOT="${VISIONOPS_SERVER_PUBLISH_ROOT:-${DATA_ROOT}/published_models}"

python3 -m apps.server_api.backend.main \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-root "${DATA_ROOT}" \
  --incoming-root "${INCOMING_ROOT}" \
  --publish-root "${PUBLISH_ROOT}"
