#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 Collector Web launcher.
# Usage:
#   bash scripts/start_collector_web.sh
# Environment overrides:
#   VISIONOPS_EDGE_ROOT, VISIONOPS_COLLECTOR_VENV, VISIONOPS_COLLECTOR_PORT,
#   VISIONOPS_RUNTIME_URL, VISIONOPS_GATEWAY_URL, VISIONOPS_BUSINESS_APP_URL,
#   VISIONOPS_DEVICE_ID

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
VENV_DIR="${VISIONOPS_COLLECTOR_VENV:-/opt/visionops/venv}"
cd "${EDGE_ROOT}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

HOST="${VISIONOPS_COLLECTOR_HOST:-0.0.0.0}"
PORT="${VISIONOPS_COLLECTOR_PORT:-18091}"
RUNTIME_URL="${VISIONOPS_RUNTIME_URL:-http://127.0.0.1:28081}"
GATEWAY_URL="${VISIONOPS_GATEWAY_URL:-http://127.0.0.1:19090}"
BUSINESS_APP_URL="${VISIONOPS_BUSINESS_APP_URL:-http://127.0.0.1:19110}"
DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"

python3 -m apps.collector_web.backend.main \
  --host "${HOST}" \
  --port "${PORT}" \
  --runtime-url "${RUNTIME_URL}" \
  --gateway-url "${GATEWAY_URL}" \
  --business-app-url "${BUSINESS_APP_URL}" \
  --device-id "${DEVICE_ID}"
