#!/usr/bin/env bash
set -euo pipefail

ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
UNIT_DIR=/etc/systemd/system
CONFIG_DIR=/etc/visionops_v3
LINE_CONFIG="${CONFIG_DIR}/carton_line.yaml"
ENV_FILE="${CONFIG_DIR}/carton_line.env"
SOURCE_DIR="${ROOT}/production/carton_line"

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run with sudo" >&2
  exit 1
fi

install -d -m 0755 "${CONFIG_DIR}"
if [[ ! -f "${LINE_CONFIG}" ]]; then
  install -m 0644 "${SOURCE_DIR}/config/line.yaml" "${LINE_CONFIG}"
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<EOF
VISIONOPS_V3_ROOT=${ROOT}
VISIONOPS_VENV=/opt/visionops/venv
VISIONOPS_CARTON_LINE_CONFIG=${LINE_CONFIG}
VISIONOPS_CAMERA_BRIDGE_URL=http://127.0.0.1:18182
VISIONOPS_PARTITION_MODEL_DIR=${ROOT}/models/carton_partition_check/current
VISIONOPS_TUBE_MODEL_DIR=${ROOT}/models/carton_tube_check/current
EOF
fi

for unit in \
  visionops-v3-runtime-partition.service \
  visionops-v3-runtime-tube.service \
  visionops-v3-robot-gateway.service \
  visionops-v3-collector-partition.service \
  visionops-v3-collector-tube.service; do
  install -m 0644 "${SOURCE_DIR}/deploy/systemd/${unit}" "${UNIT_DIR}/${unit}"
done

systemctl daemon-reload
systemctl enable \
  visionops-v3-runtime-partition.service \
  visionops-v3-runtime-tube.service \
  visionops-v3-robot-gateway.service \
  visionops-v3-collector-partition.service \
  visionops-v3-collector-tube.service

echo "Line config: ${LINE_CONFIG}"
echo "Environment: ${ENV_FILE}"
echo "Start: sudo systemctl start visionops-v3-runtime-partition visionops-v3-runtime-tube visionops-v3-robot-gateway visionops-v3-collector-partition visionops-v3-collector-tube"
