#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${VISIONOPS_BUILD_DIR:-${ROOT_DIR}/build}"
BINARY="${BUILD_DIR}/edge/runtime_cpp/visionops_runtime_mock"
HOST="127.0.0.1"
PORT="${VISIONOPS_SMOKE_PORT:-}"
LOG_FILE="$(mktemp /tmp/visionops-runtime-mock.XXXXXX.log)"
SNAPSHOT_FILE="$(mktemp /tmp/visionops-runtime-snapshot.XXXXXX.jpg)"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${LOG_FILE}" "${SNAPSHOT_FILE}"
}
trap cleanup EXIT INT TERM

if [[ -z "${PORT}" ]]; then
  PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j4 --target visionops_runtime_mock

"${BINARY}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device-id example-edge-smoke \
  --component rknn_runtime \
  --mock-task-type detection \
  >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

BASE_URL="http://${HOST}:${PORT}"
for _ in $(seq 1 50); do
  if curl --silent --fail "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl --silent --fail "${BASE_URL}/health" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/status" | python -c \
  'import json, sys; data=json.load(sys.stdin); model=data["loaded_model"]; assert model["backend"] == "mock" and model["model_load_error"] is None'
MISSING_CODE="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "${BASE_URL}/api/runtime/latest_result")"
[[ "${MISSING_CODE}" == "404" ]]
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "${BASE_URL}/api/runtime/start_preview" | python -m json.tool >/dev/null
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "${BASE_URL}/api/runtime/infer_once" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/latest_result" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/snapshot.jpg" -o "${SNAPSHOT_FILE}"
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "${BASE_URL}/api/runtime/stop_preview" | python -c \
  'import json, sys; data=json.load(sys.stdin); assert data["mode"] == "idle" and data["running"] is False'

python -c 'import pathlib, sys; data=pathlib.Path(sys.argv[1]).read_bytes(); assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"' "${SNAPSHOT_FILE}"

kill -TERM "${SERVER_PID}"
wait "${SERVER_PID}"
SERVER_PID=""
echo "Runtime Mock 冒烟测试通过"
