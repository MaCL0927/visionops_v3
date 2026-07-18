#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 generic Collector Web launcher.
# Usage:
#   bash scripts/start_collector.sh
# Environment overrides:
#   VISIONOPS_EDGE_ROOT, VISIONOPS_COLLECTOR_VENV, VISIONOPS_COLLECTOR_PORT,
#   VISIONOPS_RUNTIME_URL, VISIONOPS_GATEWAY_URL, VISIONOPS_BUSINESS_APP_URL,
#   VISIONOPS_DEVICE_ID

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
VENV_DIR="${VISIONOPS_COLLECTOR_VENV:-${VISIONOPS_VENV:-${EDGE_ROOT}/venv}}"
cd "${EDGE_ROOT}"

PYTHON_BIN="${VENV_DIR}/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] VisionOps v3 venv 不存在: ${PYTHON_BIN}" >&2
  echo "        请先运行: sudo bash ${EDGE_ROOT}/scripts/setup_edge_env.sh" >&2
  exit 1
fi

HOST="${VISIONOPS_COLLECTOR_HOST:-0.0.0.0}"
PORT="${VISIONOPS_COLLECTOR_PORT:-18091}"
RUNTIME_URL="${VISIONOPS_RUNTIME_URL:-http://127.0.0.1:28081}"
GATEWAY_URL="${VISIONOPS_GATEWAY_URL:-http://127.0.0.1:19090}"
BUSINESS_APP_URL="${VISIONOPS_BUSINESS_APP_URL:-http://127.0.0.1:19110}"
DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"

exec "${PYTHON_BIN}" -m apps.collector_web.backend.main \
  --host "${HOST}" \
  --port "${PORT}" \
  --runtime-url "${RUNTIME_URL}" \
  --gateway-url "${GATEWAY_URL}" \
  --business-app-url "${BUSINESS_APP_URL}" \
  --device-id "${DEVICE_ID}"
