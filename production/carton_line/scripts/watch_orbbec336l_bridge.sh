#!/usr/bin/env bash
set -euo pipefail

# External safety net for Orbbec SDK hangs.
# The Bridge itself normally detects stale RGB/depth frames and rebuilds the
# complete Pipeline. This watchdog only restarts the process when HTTP is down,
# the recovery thread has died, or the internal recovery state stops making
# progress for an extended period.

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
LOCK_FILE="${VISIONOPS_CAMERA_WATCHDOG_LOCK_FILE:-/run/visionops-orbbec336l-watchdog.lock}"
STATE_FILE="${VISIONOPS_CAMERA_WATCHDOG_STATE_FILE:-/run/visionops-orbbec336l-watchdog.state}"
STAMP_FILE="${VISIONOPS_CAMERA_WATCHDOG_STAMP_FILE:-/run/visionops-orbbec336l-watchdog.last_restart}"
DISABLE_FILE="${VISIONOPS_CAMERA_WATCHDOG_DISABLE_FILE:-/run/visionops-orbbec336l-watchdog.disabled}"

log() {
  logger -t visionops-orbbec-watchdog -- "$*"
  echo "[visionops-orbbec-watchdog] $*"
}

# For planned maintenance, either stop/disable the timer or create this file.
[[ -e "${DISABLE_FILE}" ]] && exit 0

exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

now_s="$(date +%s)"
health_json="$(curl -fsS --max-time 3 "${BRIDGE_URL}/health" 2>/dev/null || true)"

health_state="$({
  HEALTH_JSON="${health_json}" STALE_MS="${STALE_MS}" python3 - <<'PY'
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

# Healthy Bridge: reset progress tracker. If Runtime still has stale cache, nudge
# it here so recovery after USB reinsertion is end-to-end.
if [[ "${healthy}" == "1" ]]; then
  printf '%s %s %s\n' "${attempts}" "${frame_count}" "${now_s}" >"${STATE_FILE}"

  if systemctl is-active --quiet "${RUNTIME_SERVICE}"; then
    runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"
    if ! RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" python3 - <<'PY' >/dev/null 2>&1
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

# Read previous progress. A physically unplugged camera is expected to keep
# incrementing reconnect_attempt_count according to exponential backoff. Do not
# restart the process while that internal recovery loop is making progress.
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
fi

# When the cable is simply absent and the internal attempt counter continues to
# progress, leave the process running so it can detect reinsertion without a
# restart storm.
if [[ -z "${restart_reason}" ]]; then
  exit 0
fi

last_restart=0
if [[ -r "${STAMP_FILE}" ]]; then
  last_restart="$(cat "${STAMP_FILE}" 2>/dev/null || echo 0)"
fi
if (( now_s - last_restart < COOLDOWN_S )); then
  exit 0
fi
printf '%s\n' "${now_s}" >"${STAMP_FILE}"

log "${restart_reason}; state=${camera_state}, color_age_ms=${color_age}, depth_age_ms=${depth_age}, attempts=${attempts}; restarting ${BRIDGE_SERVICE}"
systemctl restart "${BRIDGE_SERVICE}"

# Wait until both RGB and depth are fresh. If the physical camera is still
# unplugged this loop times out; the Bridge remains alive and continues its own
# exponential reconnect attempts.
recovered=0
for _ in $(seq 1 $((RECOVERY_WAIT_S * 2))); do
  sleep 0.5
  health_json="$(curl -fsS --max-time 1 "${BRIDGE_URL}/health" 2>/dev/null || true)"
  if HEALTH_JSON="${health_json}" STALE_MS="${STALE_MS}" python3 - <<'PY' >/dev/null 2>&1
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

if [[ "${recovered}" != "1" ]]; then
  log "Bridge process restarted, but camera is still unavailable; internal reconnect remains active"
  exit 0
fi

log "Bridge camera recovered; resetting Pick Runtime preview"
if systemctl is-active --quiet "${RUNTIME_SERVICE}"; then
  curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/stop_preview" >/dev/null 2>&1 || true
  sleep 0.5
  curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/start_preview" >/dev/null 2>&1 || true
  sleep "${RUNTIME_RECOVERY_WAIT_S}"

  runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"
  if ! RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" python3 - <<'PY' >/dev/null 2>&1
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
