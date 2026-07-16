#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_DIR="${ROOT}/production/carton_palletizing"
CONFIG_DIR="/etc/visionops_v3"
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_FILE="${CONFIG_DIR}/carton_palletizing.yaml"
ENV_FILE="${CONFIG_DIR}/carton_palletizing.env"
ENABLE=1

if [[ "${1:-}" == "--no-enable" ]]; then
  ENABLE=0
elif [[ $# -gt 0 ]]; then
  echo "Usage: sudo bash production/carton_palletizing/deploy/install_services.sh [--no-enable]" >&2
  exit 2
fi

install -d -m 0755 "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  install -m 0644 "${SOURCE_DIR}/config/line.yaml" "${CONFIG_FILE}"
  echo "Created ${CONFIG_FILE}; calibrate class names and four slot polygons before production use."
else
  install -m 0644 "${SOURCE_DIR}/config/line.yaml" "${CONFIG_FILE}.example"
  echo "Kept existing ${CONFIG_FILE}; refreshed ${CONFIG_FILE}.example."
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  install -m 0644 "${SOURCE_DIR}/deploy/production.env.example" "${ENV_FILE}"
fi

for unit in "${SOURCE_DIR}"/deploy/systemd/*.service; do
  install -m 0644 "${unit}" "${SYSTEMD_DIR}/$(basename "${unit}")"
done
systemctl daemon-reload

UNITS=(
  visionops-v3-carton-palletizing-runtime.service
  visionops-v3-carton-palletizing-app.service
  visionops-v3-carton-palletizing-collector.service
)
if [[ ${ENABLE} -eq 1 ]]; then
  systemctl enable --now "${UNITS[@]}"
  systemctl --no-pager --full status "${UNITS[@]}" || true
else
  echo "Installed only. Start with: systemctl enable --now ${UNITS[*]}"
fi
