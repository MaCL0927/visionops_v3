#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-}"
ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
VENV="${VISIONOPS_VENV:-/opt/visionops/venv}"
CONFIG="${VISIONOPS_CARTON_LINE_CONFIG:-${ROOT}/production/carton_line/config/line.yaml}"
cd "${ROOT}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
fi
exec python3 -m production.carton_line.launcher --config "${CONFIG}" collector "${TASK}"
