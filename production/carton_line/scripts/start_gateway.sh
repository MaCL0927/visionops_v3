#!/usr/bin/env bash
set -euo pipefail
ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
VENV="${VISIONOPS_VENV:-${ROOT}/venv}"
CONFIG="${VISIONOPS_CARTON_LINE_CONFIG:-${ROOT}/production/carton_line/config/line.yaml}"
cd "${ROOT}"
PYTHON_BIN="${VENV}/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] VisionOps v3 venv 不存在: ${PYTHON_BIN}" >&2
  echo "        请先运行: sudo bash ${ROOT}/scripts/setup_edge_env.sh" >&2
  exit 1
fi
exec "${PYTHON_BIN}" -m production.carton_line.launcher --config "${CONFIG}" gateway
