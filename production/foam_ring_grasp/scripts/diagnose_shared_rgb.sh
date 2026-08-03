#!/usr/bin/env bash
set -euo pipefail

RUNTIME_URL="${VISIONOPS_RUNTIME_URL:-http://127.0.0.1:28081}"
BRIDGE_URL="${VISIONOPS_BRIDGE_URL:-http://127.0.0.1:18182}"
SHM_NAME="${VISIONOPS_SHARED_MEMORY_NAME:-/visionops_orbbec336l_rgb}"
SHM_FILE="/dev/shm/${SHM_NAME#/}"

echo "[INFO] user: $(id)"
echo "[INFO] bridge: ${BRIDGE_URL}"
echo "[INFO] runtime: ${RUNTIME_URL}"
echo "[INFO] shared memory: ${SHM_FILE}"

if [[ ! -e "${SHM_FILE}" ]]; then
  echo "[FAIL] shared-memory file does not exist: ${SHM_FILE}" >&2
  exit 2
fi

stat -c '[INFO] mode=%a owner=%U group=%G size=%s path=%n' "${SHM_FILE}"
if [[ -r "${SHM_FILE}" ]]; then
  echo "[PASS] current user can read shared memory"
else
  echo "[FAIL] current user cannot read shared memory" >&2
  echo "       Temporary test: sudo chmod 0664 ${SHM_FILE}" >&2
  exit 3
fi

python3 - "${BRIDGE_URL}" "${RUNTIME_URL}" <<'PY'
import json
import sys
from urllib.request import urlopen

bridge_url, runtime_url = sys.argv[1:]
with urlopen(bridge_url.rstrip('/') + '/health', timeout=5) as r:
    bridge = json.load(r)
with urlopen(runtime_url.rstrip('/') + '/api/runtime/status', timeout=5) as r:
    runtime = json.load(r)
source = runtime.get('frame_source') or {}
print('[INFO] bridge shared_rgb_ready =', bridge.get('shared_rgb_ready'))
print('[INFO] bridge shared_rgb_publish_count =', bridge.get('shared_rgb_publish_count'))
print('[INFO] bridge shared_memory_mode =', bridge.get('shared_memory_mode'))
print('[INFO] runtime configured_transport =', source.get('configured_transport'))
print('[INFO] runtime transport =', source.get('transport'))
print('[INFO] runtime shared_memory_sequence =', source.get('shared_memory_sequence'))
print('[INFO] runtime shared_memory_last_error =', source.get('shared_memory_last_error'))
PY
