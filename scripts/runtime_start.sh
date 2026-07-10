#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 RKNN Runtime launcher for LB3576 + HP60C bridge.
# Usage:
#   bash scripts/start_edge_runtime_hp60c.sh [/opt/visionops_v3/models/<model_id>]
# Environment overrides:
#   VISIONOPS_EDGE_ROOT, MODEL_DIR, VISIONOPS_RUNTIME_PORT, VISIONOPS_DEVICE_ID,
#   VISIONOPS_HP60C_URL, VISIONOPS_RUNTIME_BIN

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
cd "${EDGE_ROOT}"

MODEL_DIR="${1:-${MODEL_DIR:-/opt/visionops_v3/models/test_rknn_model}}"
RUNTIME_BIN="${VISIONOPS_RUNTIME_BIN:-./build-rknn/edge/runtime_cpp/visionops_runtime_mock}"
DEVICE_ID="${VISIONOPS_DEVICE_ID:-lb3576-001}"
PORT="${VISIONOPS_RUNTIME_PORT:-28081}"
HP60C_URL="${VISIONOPS_HP60C_URL:-http://127.0.0.1:18182}"

exec "${RUNTIME_BIN}" \
  --backend rknn \
  --preprocess-backend rga \
  --rga-mode resize_rgb \
  --frame-source hp60c_bridge \
  --hp60c-url "${HP60C_URL}" \
  --hp60c-snapshot-path /stream/snapshot.jpg \
  --hp60c-health-path /health \
  --model-dir "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device-id "${DEVICE_ID}"
