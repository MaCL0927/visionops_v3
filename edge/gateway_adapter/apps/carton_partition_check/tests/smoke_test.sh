#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PORT=19120
LOG_FILE="$(mktemp /tmp/visionops-carton-partition.XXXXXX.log)"
PID=""

cleanup() {
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    kill -TERM "${PID}" 2>/dev/null || true
    wait "${PID}" 2>/dev/null || true
  fi
  rm -f "${LOG_FILE}"
}
trap cleanup EXIT INT TERM

python -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", 19120)); s.close()'
cd "${ROOT_DIR}"
python -m edge.gateway_adapter.apps.carton_partition_check.service \
  --host 127.0.0.1 --port "${PORT}" --upstream-kind file --mock-case defect \
  --poll-interval-ms 5000 >"${LOG_FILE}" 2>&1 &
PID=$!

for _ in $(seq 1 50); do
  curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null && break
  sleep 0.1
done
curl --silent --fail "http://127.0.0.1:${PORT}/health" | python -m json.tool >/dev/null
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:${PORT}/api/app/evaluate_once" | python -m json.tool >/dev/null
curl --silent --fail "http://127.0.0.1:${PORT}/api/app/latest_decision" | python -m json.tool >/dev/null
curl --silent --fail "http://127.0.0.1:${PORT}/api/app/registers" | python -m json.tool >/dev/null
curl --silent --fail "http://127.0.0.1:${PORT}/api/app/register_map" | python -m json.tool >/dev/null
kill -TERM "${PID}"
wait "${PID}"
PID=""
echo "carton_partition_check Mock 冒烟测试通过"
