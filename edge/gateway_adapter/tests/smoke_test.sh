#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${VISIONOPS_BUILD_DIR:-${ROOT_DIR}/build}"
RUNTIME_BINARY="${BUILD_DIR}/edge/runtime_cpp/visionops_runtime_mock"
RUNTIME_PORT="${VISIONOPS_RUNTIME_PORT:-18080}"
COLLECTOR_PORT="${VISIONOPS_COLLECTOR_PORT:-8090}"
GATEWAY_PORT="${VISIONOPS_GATEWAY_PORT:-19090}"
MODBUS_PORT="${VISIONOPS_MODBUS_PORT:-1502}"
RUNTIME_LOG="$(mktemp /tmp/visionops-runtime.XXXXXX.log)"
COLLECTOR_LOG="$(mktemp /tmp/visionops-collector.XXXXXX.log)"
GATEWAY_LOG="$(mktemp /tmp/visionops-gateway.XXXXXX.log)"
RUNTIME_PID=""
COLLECTOR_PID=""
GATEWAY_PID=""

cleanup() {
  for pid_name in GATEWAY_PID COLLECTOR_PID RUNTIME_PID; do
    pid="${!pid_name}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f "${RUNTIME_LOG}" "${COLLECTOR_LOG}" "${GATEWAY_LOG}"
}
trap cleanup EXIT INT TERM

python -c 'import socket, sys
for port in map(int, sys.argv[1:]):
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"端口 {port} 不可用: {exc}")
    finally:
        sock.close()' "${RUNTIME_PORT}" "${COLLECTOR_PORT}" "${GATEWAY_PORT}" "${MODBUS_PORT}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j4 --target visionops_runtime_mock

"${RUNTIME_BINARY}" \
  --host 127.0.0.1 \
  --port "${RUNTIME_PORT}" \
  --device-id example-edge-gateway-smoke \
  --mock-task-type detection \
  >"${RUNTIME_LOG}" 2>&1 &
RUNTIME_PID=$!

cd "${ROOT_DIR}"
python -m apps.collector_web.backend.main \
  --host 127.0.0.1 \
  --port "${COLLECTOR_PORT}" \
  --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
  --device-id example-edge-gateway-smoke \
  >"${COLLECTOR_LOG}" 2>&1 &
COLLECTOR_PID=$!

python -m edge.gateway_adapter.gateway_mock_service \
  --host 127.0.0.1 \
  --port "${GATEWAY_PORT}" \
  --upstream-url "http://127.0.0.1:${COLLECTOR_PORT}" \
  --upstream-kind collector \
  --modbus-host 127.0.0.1 \
  --modbus-port "${MODBUS_PORT}" \
  --poll-interval-ms 500 \
  --device-id example-edge-gateway-smoke \
  --app-id generic_mock \
  >"${GATEWAY_LOG}" 2>&1 &
GATEWAY_PID=$!

wait_for_health() {
  local url="$1"
  for _ in $(seq 1 50); do
    if curl --silent --fail "${url}" >/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "服务未就绪: ${url}" >&2
  return 1
}

wait_for_health "http://127.0.0.1:${RUNTIME_PORT}/health"
wait_for_health "http://127.0.0.1:${COLLECTOR_PORT}/health"
wait_for_health "http://127.0.0.1:${GATEWAY_PORT}/health"

curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "http://127.0.0.1:${RUNTIME_PORT}/api/runtime/infer_once" | python -m json.tool >/dev/null
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "http://127.0.0.1:${GATEWAY_PORT}/api/gateway/poll_once" | python -m json.tool >/dev/null
curl --silent --fail \
  "http://127.0.0.1:${GATEWAY_PORT}/api/gateway/latest_message" | python -m json.tool >/dev/null
curl --silent --fail \
  "http://127.0.0.1:${GATEWAY_PORT}/api/gateway/registers" | python -m json.tool >/dev/null
python -m edge.modbus_adapter.modbus_test_client \
  --host 127.0.0.1 \
  --port "${MODBUS_PORT}" \
  --read-start 0 \
  --read-count 20 \
  --print-registers >/dev/null

kill -TERM "${GATEWAY_PID}"
wait "${GATEWAY_PID}"
GATEWAY_PID=""
kill -TERM "${COLLECTOR_PID}"
wait "${COLLECTOR_PID}"
COLLECTOR_PID=""
kill -TERM "${RUNTIME_PID}"
wait "${RUNTIME_PID}"
RUNTIME_PID=""
echo "Gateway / Modbus Mock 冒烟测试通过"
