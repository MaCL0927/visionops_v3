#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VISIONOPS_VENV:-${ROOT}/venv}"
CONFIG="${VISIONOPS_DETERGENT_GRASP_CONFIG:-/etc/visionops_v3/detergent_grasp.yaml}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
exec "${VENV}/bin/python3" -m production.detergent_grasp.launcher --config "${CONFIG}" collector
