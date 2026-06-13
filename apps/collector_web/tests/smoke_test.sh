#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${VISIONOPS_BUILD_DIR:-${ROOT_DIR}/build}"
RUNTIME_BINARY="${BUILD_DIR}/edge/runtime_cpp/visionops_runtime_mock"
RUNTIME_PORT="18080"
COLLECTOR_PORT="8090"
RUNTIME_LOG="$(mktemp /tmp/visionops-runtime.XXXXXX.log)"
COLLECTOR_LOG="$(mktemp /tmp/visionops-collector.XXXXXX.log)"
SNAPSHOT_FILE="$(mktemp /tmp/visionops-collector-snapshot.XXXXXX.jpg)"
RUNTIME_PID=""
COLLECTOR_PID=""

cleanup() {
  if [[ -n "${COLLECTOR_PID}" ]] && kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
    kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    wait "${COLLECTOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RUNTIME_PID}" ]] && kill -0 "${RUNTIME_PID}" 2>/dev/null; then
    kill -TERM "${RUNTIME_PID}" 2>/dev/null || true
    wait "${RUNTIME_PID}" 2>/dev/null || true
  fi
  rm -f "${RUNTIME_LOG}" "${COLLECTOR_LOG}" "${SNAPSHOT_FILE}"
}
trap cleanup EXIT INT TERM

python -c 'import socket, sys
for port in (18080, 8090):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"端口 {port} 不可用: {exc}")
    finally:
        sock.close()'

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j4 --target visionops_runtime_mock

"${RUNTIME_BINARY}" \
  --host 127.0.0.1 \
  --port "${RUNTIME_PORT}" \
  --device-id example-edge-smoke \
  --component rknn_runtime \
  --mock-task-type detection \
  >"${RUNTIME_LOG}" 2>&1 &
RUNTIME_PID=$!

cd "${ROOT_DIR}"
python -m apps.collector_web.backend.main \
  --host 127.0.0.1 \
  --port "${COLLECTOR_PORT}" \
  --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
  --device-id example-edge-smoke \
  --component collector_web \
  >"${COLLECTOR_LOG}" 2>&1 &
COLLECTOR_PID=$!

BASE_URL="http://127.0.0.1:${COLLECTOR_PORT}"
for _ in $(seq 1 50); do
  if curl --silent --fail "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done

curl --silent --fail "${BASE_URL}/health" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/collector/status" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/status" | python -m json.tool >/dev/null
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "${BASE_URL}/api/runtime/start_preview" | python -m json.tool >/dev/null
curl --silent --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "${BASE_URL}/api/runtime/infer_once" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/latest_result" | python -m json.tool >/dev/null
curl --silent --fail "${BASE_URL}/api/runtime/snapshot.jpg" -o "${SNAPSHOT_FILE}"
python -c 'import pathlib, sys; data=pathlib.Path(sys.argv[1]).read_bytes(); assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"' "${SNAPSHOT_FILE}"

kill -TERM "${COLLECTOR_PID}"
wait "${COLLECTOR_PID}"
COLLECTOR_PID=""
kill -TERM "${RUNTIME_PID}"
wait "${RUNTIME_PID}"
RUNTIME_PID=""
echo "Collector Web 代理冒烟测试通过"
