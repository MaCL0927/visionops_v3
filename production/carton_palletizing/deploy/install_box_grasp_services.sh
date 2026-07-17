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
  echo "Usage: sudo bash production/carton_palletizing/deploy/install_box_grasp_services.sh [--no-enable]" >&2
  exit 2
fi

install -d -m 0755 "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  install -m 0644 "${SOURCE_DIR}/config/line.yaml" "${CONFIG_FILE}"
  echo "Created ${CONFIG_FILE}. Set box_grasp.video.public_url to the box LAN IP before robot integration."
else
  install -m 0644 "${SOURCE_DIR}/config/line.yaml" "${CONFIG_FILE}.example"
  echo "Kept existing ${CONFIG_FILE}; refreshed ${CONFIG_FILE}.example. Merge the box_grasp section when upgrading."
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  install -m 0644 "${SOURCE_DIR}/deploy/production.env.example" "${ENV_FILE}"
fi

UNITS=(
  visionops-v3-carton-box-grasp-runtime.service
  visionops-v3-carton-box-grasp-app.service
  visionops-v3-carton-box-grasp-collector.service
)
for unit in "${UNITS[@]}"; do
  install -m 0644 "${SOURCE_DIR}/deploy/systemd/${unit}" "${SYSTEMD_DIR}/${unit}"
done
systemctl daemon-reload
if [[ ${ENABLE} -eq 1 ]]; then
  systemctl enable --now "${UNITS[@]}"
  systemctl --no-pager --full status "${UNITS[@]}" || true
else
  echo "Installed only. Start with: systemctl enable --now ${UNITS[*]}"
fi
