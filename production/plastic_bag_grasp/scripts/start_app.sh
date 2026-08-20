#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VISIONOPS_VENV:-${ROOT}/venv}"
CONFIG="${VISIONOPS_PLASTIC_BAG_GRASP_CONFIG:-/etc/visionops_v3/plastic_bag_grasp.yaml}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
exec "${VENV}/bin/python3" -m production.plastic_bag_grasp.launcher --config "${CONFIG}" app
