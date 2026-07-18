#!/usr/bin/env bash
set -euo pipefail

ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
VENV="${VISIONOPS_VENV:-${ROOT}/venv}"
PYTHON_BIN="${VISIONOPS_PYTHON_BIN:-${VENV}/bin/python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  # Watchdogs only use the Python standard library. Falling back to the board's
  # system Python keeps camera recovery alive before/while the v3 venv is repaired.
  PYTHON_BIN="$(command -v python3)"
fi

# External safety net for Orbbec SDK hangs.
#
# Recovery order:
#   1. The Bridge rebuilds its own SDK Pipeline.
#   2. This watchdog restarts the Bridge when the camera remains unavailable or
#      the internal recovery state stops making progress.
#   3. If ten Bridge service restarts still cannot restore fresh RGB + depth
#      frames, the watchdog reboots the visual box once for the current fault
#      incident. The persistent incident marker prevents a reboot loop while
#      the USB cable remains physically disconnected.

BRIDGE_SERVICE="${VISIONOPS_CAMERA_BRIDGE_SERVICE:-visionops-orbbec336l-bridge.service}"
RUNTIME_SERVICE="${VISIONOPS_PICK_RUNTIME_SERVICE:-visionops-v3-runtime-pick.service}"
BRIDGE_URL="${VISIONOPS_CAMERA_BRIDGE_URL:-http://127.0.0.1:18182}"
RUNTIME_URL="${VISIONOPS_PICK_RUNTIME_URL:-http://127.0.0.1:28083}"
STALE_MS="${VISIONOPS_CAMERA_WATCHDOG_STALE_MS:-5000}"
NO_PROGRESS_S="${VISIONOPS_CAMERA_WATCHDOG_NO_PROGRESS_S:-45}"
STALE_STATE_NO_PROGRESS_S="${VISIONOPS_CAMERA_WATCHDOG_STALE_STATE_NO_PROGRESS_S:-15}"
COOLDOWN_S="${VISIONOPS_CAMERA_WATCHDOG_COOLDOWN_S:-60}"
RECOVERY_WAIT_S="${VISIONOPS_CAMERA_WATCHDOG_RECOVERY_WAIT_S:-20}"
RUNTIME_RECOVERY_WAIT_S="${VISIONOPS_CAMERA_WATCHDOG_RUNTIME_WAIT_S:-3}"

# 7x24 escalation policy. A physically unavailable camera is allowed a grace
# period for the Bridge's own reconnect loop. After that, the watchdog performs
# controlled Bridge service restarts. Ten unsuccessful service restarts trigger
# one whole-box reboot for the current camera fault incident.
RESTART_WHILE_UNHEALTHY="${VISIONOPS_CAMERA_WATCHDOG_RESTART_WHILE_UNHEALTHY:-true}"
UNHEALTHY_RESTART_AFTER_S="${VISIONOPS_CAMERA_WATCHDOG_UNHEALTHY_RESTART_AFTER_S:-30}"
MAX_SERVICE_RESTARTS="${VISIONOPS_CAMERA_WATCHDOG_MAX_SERVICE_RESTARTS:-10}"
REBOOT_ENABLED="${VISIONOPS_CAMERA_WATCHDOG_REBOOT_ENABLED:-true}"
REBOOT_DELAY_S="${VISIONOPS_CAMERA_WATCHDOG_REBOOT_DELAY_S:-5}"
REBOOT_ONCE_PER_INCIDENT="${VISIONOPS_CAMERA_WATCHDOG_REBOOT_ONCE_PER_INCIDENT:-true}"

LOCK_FILE="${VISIONOPS_CAMERA_WATCHDOG_LOCK_FILE:-/run/visionops-orbbec336l-watchdog.lock}"
STATE_FILE="${VISIONOPS_CAMERA_WATCHDOG_STATE_FILE:-/run/visionops-orbbec336l-watchdog.state}"
STAMP_FILE="${VISIONOPS_CAMERA_WATCHDOG_STAMP_FILE:-/run/visionops-orbbec336l-watchdog.last_restart}"
DISABLE_FILE="${VISIONOPS_CAMERA_WATCHDOG_DISABLE_FILE:-/run/visionops-orbbec336l-watchdog.disabled}"
PERSIST_DIR="${VISIONOPS_CAMERA_WATCHDOG_PERSIST_DIR:-/var/lib/visionops_v3/watchdog}"
FAIL_COUNT_FILE="${VISIONOPS_CAMERA_WATCHDOG_FAIL_COUNT_FILE:-${PERSIST_DIR}/orbbec336l_failed_service_restarts}"
INCIDENT_FILE="${VISIONOPS_CAMERA_WATCHDOG_INCIDENT_FILE:-${PERSIST_DIR}/orbbec336l_incident_started_at}"
REBOOT_MARK_FILE="${VISIONOPS_CAMERA_WATCHDOG_REBOOT_MARK_FILE:-${PERSIST_DIR}/orbbec336l_reboot_issued}"

log() {
  logger -t visionops-orbbec-watchdog -- "$*"
  echo "[visionops-orbbec-watchdog] $*"
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

as_uint() {
  local value="${1:-}"
  local fallback="${2:-0}"
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${fallback}"
  fi
}

atomic_write() {
  local path="$1"
  local value="$2"
  local tmp="${path}.tmp.$$"
  printf '%s\n' "${value}" >"${tmp}"
  mv -f "${tmp}" "${path}"
}

read_uint_file() {
  local path="$1"
  local fallback="${2:-0}"
  local value="${fallback}"
  if [[ -r "${path}" ]]; then
    value="$(cat "${path}" 2>/dev/null || printf '%s' "${fallback}")"
  fi
  as_uint "${value}" "${fallback}"
}

reset_fault_incident() {
  atomic_write "${FAIL_COUNT_FILE}" 0
  rm -f "${INCIDENT_FILE}" "${REBOOT_MARK_FILE}"
}

ensure_fault_incident() {
  local now="$1"
  if [[ ! -r "${INCIDENT_FILE}" ]]; then
    atomic_write "${INCIDENT_FILE}" "${now}"
  fi
}

increment_failed_restart_count() {
  local count
  count="$(read_uint_file "${FAIL_COUNT_FILE}" 0)"
  count=$((count + 1))
  atomic_write "${FAIL_COUNT_FILE}" "${count}"
  printf '%s' "${count}"
}

maybe_reboot_box() {
  local failure_count="$1"
  local reason="$2"

  if (( MAX_SERVICE_RESTARTS <= 0 || failure_count < MAX_SERVICE_RESTARTS )); then
    return 0
  fi
  if ! is_true "${REBOOT_ENABLED}"; then
    log "Bridge 连续恢复失败 ${failure_count} 次，已达到阈值，但整机重启功能已禁用"
    return 0
  fi
  if is_true "${REBOOT_ONCE_PER_INCIDENT}" && [[ -e "${REBOOT_MARK_FILE}" ]]; then
    log "当前相机故障事件已执行过整机重启，不再重复 reboot，避免相机未插入时形成启动循环"
    return 0
  fi

  local now_s boot_id
  now_s="$(date +%s)"
  boot_id="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
  atomic_write "${REBOOT_MARK_FILE}" "timestamp=${now_s} boot_id=${boot_id} failures=${failure_count} reason=${reason}"
  sync "${REBOOT_MARK_FILE}" 2>/dev/null || true

  log "CRITICAL: ${BRIDGE_SERVICE} 连续重启 ${failure_count} 次后相机仍异常；${REBOOT_DELAY_S}s 后重启视觉盒子。reason=${reason}"
  if (( REBOOT_DELAY_S > 0 )); then
    sleep "${REBOOT_DELAY_S}"
  fi

  # The watchdog service runs as root, so systemctl reboot is equivalent to
  # sudo reboot and is easier to audit in journal/systemd.
  if systemctl reboot --no-block; then
    exit 0
  fi

  # If systemd rejected the reboot request, allow a later watchdog run to retry.
  rm -f "${REBOOT_MARK_FILE}"
  log "ERROR: systemctl reboot 执行失败，已清除 reboot 标记，后续 watchdog 将重试"
  return 1
}

STALE_MS="$(as_uint "${STALE_MS}" 5000)"
NO_PROGRESS_S="$(as_uint "${NO_PROGRESS_S}" 45)"
STALE_STATE_NO_PROGRESS_S="$(as_uint "${STALE_STATE_NO_PROGRESS_S}" 15)"
COOLDOWN_S="$(as_uint "${COOLDOWN_S}" 60)"
RECOVERY_WAIT_S="$(as_uint "${RECOVERY_WAIT_S}" 20)"
RUNTIME_RECOVERY_WAIT_S="$(as_uint "${RUNTIME_RECOVERY_WAIT_S}" 3)"
UNHEALTHY_RESTART_AFTER_S="$(as_uint "${UNHEALTHY_RESTART_AFTER_S}" 30)"
MAX_SERVICE_RESTARTS="$(as_uint "${MAX_SERVICE_RESTARTS}" 10)"
REBOOT_DELAY_S="$(as_uint "${REBOOT_DELAY_S}" 5)"

# For planned maintenance, either stop/disable the timer or create this file.
[[ -e "${DISABLE_FILE}" ]] && exit 0

mkdir -p "${PERSIST_DIR}"
chmod 0755 "${PERSIST_DIR}" 2>/dev/null || true
[[ -e "${FAIL_COUNT_FILE}" ]] || atomic_write "${FAIL_COUNT_FILE}" 0

exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

now_s="$(date +%s)"
health_json="$(curl -fsS --max-time 3 "${BRIDGE_URL}/health" 2>/dev/null || true)"

health_state="$({
  HEALTH_JSON="${health_json}" STALE_MS="${STALE_MS}" "${PYTHON_BIN}" - <<'PY'
import json
import os

try:
    data = json.loads(os.environ.get("HEALTH_JSON") or "{}")
except Exception:
    data = {}

reachable = bool(data)
connected = data.get("camera_connected") is True
state = str(data.get("camera_state") or "unknown")
thread_alive = data.get("camera_thread_alive") is True
try:
    color_age = int(data.get("last_color_age_ms", -1))
except Exception:
    color_age = -1
try:
    depth_age = int(data.get("last_depth_age_ms", -1))
except Exception:
    depth_age = -1
try:
    attempts = int(data.get("reconnect_attempt_count", 0))
except Exception:
    attempts = 0
try:
    frame_count = int(data.get("frame_count", 0))
except Exception:
    frame_count = 0
limit = int(os.environ["STALE_MS"])
fresh = connected and 0 <= color_age <= limit and 0 <= depth_age <= limit
print(
    int(reachable), int(fresh), int(thread_alive), state,
    color_age, depth_age, attempts, frame_count,
)
PY
} 2>/dev/null || echo '0 0 0 unknown -1 -1 0 0')"
read -r reachable healthy thread_alive camera_state color_age depth_age attempts frame_count <<<"${health_state}"

# Healthy Bridge: clear the entire fault incident, including the failed service
# restart counter and the once-per-incident reboot marker. A later camera fault
# starts with a clean count.
if [[ "${healthy}" == "1" ]]; then
  printf '%s %s %s\n' "${attempts}" "${frame_count}" "${now_s}" >"${STATE_FILE}"
  reset_fault_incident

  if systemctl is-active --quiet "${RUNTIME_SERVICE}"; then
    runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"
    if ! RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import json
import os
try:
    data = json.loads(os.environ.get("RUNTIME_JSON") or "{}")
except Exception:
    raise SystemExit(1)
source = data.get("frame_source") or {}
preview = data.get("running") is True and data.get("mode") == "preview"
if not preview:
    raise SystemExit(0)
try:
    age = int(source.get("latest_frame_age_ms", -1))
except Exception:
    age = -1
ok = (
    source.get("opened") is True
    and source.get("thread_alive") is True
    and source.get("stale") is False
    and 0 <= age <= int(os.environ["STALE_MS"])
)
raise SystemExit(0 if ok else 1)
PY
    then
      log "Bridge 已恢复，但 Pick Runtime 帧仍异常；重置 preview"
      curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/stop_preview" >/dev/null 2>&1 || true
      sleep 0.5
      curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/start_preview" >/dev/null 2>&1 || true
      sleep "${RUNTIME_RECOVERY_WAIT_S}"
    fi
  fi
  exit 0
fi

ensure_fault_incident "${now_s}"
incident_since="$(read_uint_file "${INCIDENT_FILE}" "${now_s}")"

# Read previous internal recovery progress. A physically unplugged camera may
# continue incrementing reconnect_attempt_count. We still allow the Bridge's own
# reconnect loop a grace period, but after that the service-level recovery path
# is exercised and counted toward the ten-restart reboot threshold.
prev_attempts=-1
prev_frames=-1
progress_since="${now_s}"
if [[ -r "${STATE_FILE}" ]]; then
  read -r prev_attempts prev_frames progress_since <"${STATE_FILE}" || true
fi
if [[ "${attempts}" != "${prev_attempts}" || "${frame_count}" != "${prev_frames}" ]]; then
  progress_since="${now_s}"
fi
printf '%s %s %s\n' "${attempts}" "${frame_count}" "${progress_since}" >"${STATE_FILE}"

restart_reason=""
if [[ "${reachable}" != "1" ]]; then
  restart_reason="Bridge /health unreachable"
elif [[ "${thread_alive}" != "1" ]]; then
  restart_reason="camera recovery thread is not alive"
elif [[ "${camera_state}" == "stale" ]] && (( now_s - progress_since >= STALE_STATE_NO_PROGRESS_S )); then
  restart_reason="pipeline stop/rebuild remained stale for $((now_s - progress_since))s"
elif (( now_s - progress_since >= NO_PROGRESS_S )); then
  restart_reason="internal recovery made no progress for $((now_s - progress_since))s"
elif is_true "${RESTART_WHILE_UNHEALTHY}" \
     && [[ ! -e "${REBOOT_MARK_FILE}" ]] \
     && (( now_s - incident_since >= UNHEALTHY_RESTART_AFTER_S )); then
  restart_reason="camera remained unavailable for $((now_s - incident_since))s"
fi

if [[ -z "${restart_reason}" ]]; then
  exit 0
fi

last_restart=0
if [[ -r "${STAMP_FILE}" ]]; then
  last_restart="$(read_uint_file "${STAMP_FILE}" 0)"
fi
if (( now_s - last_restart < COOLDOWN_S )); then
  exit 0
fi
atomic_write "${STAMP_FILE}" "${now_s}"

current_failures="$(read_uint_file "${FAIL_COUNT_FILE}" 0)"
log "${restart_reason}; state=${camera_state}, color_age_ms=${color_age}, depth_age_ms=${depth_age}, attempts=${attempts}; restarting ${BRIDGE_SERVICE} (previous_failed_restarts=${current_failures}/${MAX_SERVICE_RESTARTS})"

restart_command_ok=1
if ! systemctl restart "${BRIDGE_SERVICE}"; then
  restart_command_ok=0
fi

# Wait until both RGB and depth are fresh. If the physical camera is still
# unplugged this loop times out and counts as one failed service-level recovery.
recovered=0
if [[ "${restart_command_ok}" == "1" ]]; then
  for _ in $(seq 1 $((RECOVERY_WAIT_S * 2))); do
    sleep 0.5
    health_json="$(curl -fsS --max-time 1 "${BRIDGE_URL}/health" 2>/dev/null || true)"
    if HEALTH_JSON="${health_json}" STALE_MS="${STALE_MS}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import json
import os
data = json.loads(os.environ.get("HEALTH_JSON") or "{}")
try:
    color_age = int(data.get("last_color_age_ms", -1))
    depth_age = int(data.get("last_depth_age_ms", -1))
except Exception:
    raise SystemExit(1)
limit = int(os.environ["STALE_MS"])
ok = data.get("camera_connected") is True and 0 <= color_age <= limit and 0 <= depth_age <= limit
raise SystemExit(0 if ok else 1)
PY
    then
      recovered=1
      break
    fi
  done
fi

if [[ "${recovered}" != "1" ]]; then
  failed_restarts="$(increment_failed_restart_count)"
  log "Bridge 服务重启后相机仍不可用；failed_service_restarts=${failed_restarts}/${MAX_SERVICE_RESTARTS}"
  maybe_reboot_box "${failed_restarts}" "${restart_reason}"
  exit 0
fi

log "Bridge camera recovered after service restart; clearing fault escalation counter"
reset_fault_incident

log "Bridge camera recovered; resetting Pick Runtime preview"
if systemctl is-active --quiet "${RUNTIME_SERVICE}"; then
  curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/stop_preview" >/dev/null 2>&1 || true
  sleep 0.5
  curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/start_preview" >/dev/null 2>&1 || true
  sleep "${RUNTIME_RECOVERY_WAIT_S}"

  runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"
  if ! RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import json
import os
try:
    data = json.loads(os.environ.get("RUNTIME_JSON") or "{}")
except Exception:
    raise SystemExit(1)
source = data.get("frame_source") or {}
try:
    age = int(source.get("latest_frame_age_ms", -1))
except Exception:
    age = -1
ok = (
    data.get("running") is True
    and data.get("mode") == "preview"
    and source.get("opened") is True
    and source.get("thread_alive") is True
    and source.get("stale") is False
    and 0 <= age <= int(os.environ["STALE_MS"])
)
raise SystemExit(0 if ok else 1)
PY
  then
    log "Pick Runtime did not recover after preview reset; restarting ${RUNTIME_SERVICE}"
    systemctl restart "${RUNTIME_SERVICE}"
  fi
fi
