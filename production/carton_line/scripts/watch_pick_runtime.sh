#!/usr/bin/env bash
set -euo pipefail

# Tube-pick Runtime 帧新鲜度 watchdog。
# 正常状态只读取两个本地 HTTP 状态接口，不重启任何服务。
# 仅当 Pick Runtime 正在 preview 且帧过期/线程异常时执行分级恢复：
#   1. Bridge 正常：stop_preview -> start_preview
#   2. 仍未恢复：重启 Pick Runtime
#   3. Bridge 自身帧过期/不可访问：先重启 Bridge，再重启 Pick Runtime

RUNTIME_SERVICE="${VISIONOPS_PICK_RUNTIME_SERVICE:-visionops-v3-runtime-pick.service}"
RUNTIME_URL="${VISIONOPS_PICK_RUNTIME_URL:-http://127.0.0.1:28083}"
SELECTION_FILE="${VISIONOPS_CAMERA_SELECTION_FILE:-/opt/visionops_v3/config/active_camera.json}"
active_camera="$(sed -nE 's/.*"active_camera"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$SELECTION_FILE" 2>/dev/null | head -n1)"
if [[ "$active_camera" == "hp60c" ]]; then
  SELECTED_BRIDGE_SERVICE="visionops-hp60c-sdk-bridge.service"
  SELECTED_BRIDGE_URL="http://127.0.0.1:18181"
else
  SELECTED_BRIDGE_SERVICE="visionops-orbbec336l-bridge.service"
  SELECTED_BRIDGE_URL="http://127.0.0.1:18182"
fi
BRIDGE_SERVICE="${VISIONOPS_CAMERA_BRIDGE_SERVICE_OVERRIDE:-$SELECTED_BRIDGE_SERVICE}"
BRIDGE_URL="${VISIONOPS_CAMERA_BRIDGE_URL_OVERRIDE:-$SELECTED_BRIDGE_URL}"
STALE_MS="${VISIONOPS_PICK_WATCHDOG_STALE_MS:-5000}"
COOLDOWN_S="${VISIONOPS_PICK_WATCHDOG_COOLDOWN_S:-30}"
RECOVERY_WAIT_S="${VISIONOPS_PICK_WATCHDOG_RECOVERY_WAIT_S:-3}"

LOCK_FILE="${VISIONOPS_PICK_WATCHDOG_LOCK_FILE:-/run/visionops-v3-pick-watchdog.lock}"
STAMP_FILE="${VISIONOPS_PICK_WATCHDOG_STAMP_FILE:-/run/visionops-v3-pick-watchdog.last_action}"

log() {
  logger -t visionops-pick-watchdog -- "$*"
  echo "[visionops-pick-watchdog] $*"
}

# Runtime 被人工停止时，watchdog 不应擅自启动它。
if ! systemctl is-active --quiet "${RUNTIME_SERVICE}"; then
  exit 0
fi

exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"

runtime_state="$({
  RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" python3 - <<'PY'
import json
import os

try:
    data = json.loads(os.environ.get("RUNTIME_JSON") or "{}")
except Exception:
    data = {}

source = data.get("frame_source") or {}
preview = bool(data.get("running")) and data.get("mode") == "preview"
stale = bool(source.get("stale", True))
opened = bool(source.get("opened", False))
thread_alive = bool(source.get("thread_alive", False))
age = int(source.get("latest_frame_age_ms") or 0)
limit = int(os.environ["STALE_MS"])
healthy = preview and opened and thread_alive and not stale and age <= limit
print(int(bool(data)), int(preview), int(healthy), age, int(opened), int(thread_alive), int(stale))
PY
} 2>/dev/null || echo '0 0 0 0 0 0 1')"

read -r runtime_reachable preview_active runtime_healthy runtime_age runtime_opened thread_alive runtime_stale <<<"${runtime_state}"

# 没有开启实时预览时不处理；infer_once 仍可按需主动取帧。
if [[ "${runtime_reachable}" == "1" && "${preview_active}" != "1" ]]; then
  exit 0
fi

if [[ "${runtime_healthy}" == "1" ]]; then
  exit 0
fi

now_s="$(date +%s)"
last_action_s=0
if [[ -f "${STAMP_FILE}" ]]; then
  last_action_s="$(cat "${STAMP_FILE}" 2>/dev/null || echo 0)"
fi
if (( now_s - last_action_s < COOLDOWN_S )); then
  exit 0
fi
printf '%s\n' "${now_s}" >"${STAMP_FILE}"

bridge_json="$(curl -fsS --max-time 2 "${BRIDGE_URL}/health" 2>/dev/null || true)"
bridge_state="$({
  BRIDGE_JSON="${bridge_json}" STALE_MS="${STALE_MS}" python3 - <<'PY'
import json
import os

try:
    data = json.loads(os.environ.get("BRIDGE_JSON") or "{}")
except Exception:
    data = {}

started = bool(data.get("camera_started", False))
connected = data.get("camera_connected")
color_age = int(data.get("last_color_age_ms", -1))
depth_age = int(data.get("last_depth_age_ms", color_age))
limit = int(os.environ["STALE_MS"])
if connected is None:
    healthy = bool(data) and started and 0 <= color_age <= limit and 0 <= depth_age <= limit
else:
    healthy = bool(data) and connected is True and 0 <= color_age <= limit and 0 <= depth_age <= limit
print(int(bool(data)), int(healthy), max(color_age, depth_age))
PY
} 2>/dev/null || echo '0 0 -1')"
read -r bridge_reachable bridge_healthy bridge_age <<<"${bridge_state}"

if [[ "${bridge_healthy}" != "1" ]]; then
  # Camera/SDK lifecycle is owned by the dedicated Bridge watchdog. Avoid two
  # independent timers restarting the same process and creating a restart storm.
  log "Bridge 异常：reachable=${bridge_reachable}, color/depth_age_ms=${bridge_age}；等待 Orbbec Bridge 自恢复/watchdog 处理"
  exit 0
fi

log "Pick Runtime 帧异常：reachable=${runtime_reachable}, age_ms=${runtime_age}, opened=${runtime_opened}, thread_alive=${thread_alive}, stale=${runtime_stale}；先重置 preview"

curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/stop_preview" >/dev/null 2>&1 || true
sleep 0.5
curl -fsS --max-time 3 -X POST "${RUNTIME_URL}/api/runtime/start_preview" >/dev/null 2>&1 || true
sleep "${RECOVERY_WAIT_S}"

runtime_json="$(curl -fsS --max-time 2 "${RUNTIME_URL}/api/runtime/status" 2>/dev/null || true)"
if RUNTIME_JSON="${runtime_json}" STALE_MS="${STALE_MS}" python3 - <<'PY' >/dev/null 2>&1
import json
import os

data = json.loads(os.environ.get("RUNTIME_JSON") or "{}")
source = data.get("frame_source") or {}
age = int(source.get("latest_frame_age_ms") or 0)
healthy = (
    data.get("running") is True
    and data.get("mode") == "preview"
    and source.get("opened") is True
    and source.get("thread_alive") is True
    and source.get("stale") is False
    and age <= int(os.environ["STALE_MS"])
)
raise SystemExit(0 if healthy else 1)
PY
then
  log "Pick Runtime preview 已自动恢复"
  exit 0
fi

log "preview 重置后仍未恢复，重启 ${RUNTIME_SERVICE}"
systemctl restart "${RUNTIME_SERVICE}"
