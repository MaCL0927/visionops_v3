#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${VISIONOPS_CARTON_PALLETIZING_CONFIG:-${ROOT}/production/carton_palletizing/config/line.yaml}"
cd "${ROOT}"
exec python3 -m production.carton_palletizing.launcher --config "${CONFIG}" collector
