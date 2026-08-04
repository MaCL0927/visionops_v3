#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${VISIONOPS_M37_CONFIG:-${ROOT}/production/foam_ring_grasp/config/line.yaml}"
OUTPUT="${VISIONOPS_M37_OUTPUT:-${ROOT}/data/foam_ring_side_template_m37}"

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
Usage:
  $0 /path/to/M36_debug_bundle [extra args]

Example:
  $0 /opt/visionops_v3/data/foam_ring_online_geometry/1785833810687
EOF
  exit 2
fi

BUNDLE="$1"
shift
cd "${ROOT}"
exec python3 -m \
  production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate \
  --config "${CONFIG}" \
  --bundle "${BUNDLE}" \
  --output "${OUTPUT}" \
  "$@"
