#!/usr/bin/env bash
set -euo pipefail

BRIDGE_SERVICE="${VISIONOPS_HP60C_BRIDGE_SERVICE:-visionops-hp60c-sdk-bridge.service}"
BRIDGE_URL="${VISIONOPS_HP60C_BRIDGE_URL:-http://127.0.0.1:18181}"
SELECTION_FILE="${VISIONOPS_CAMERA_SELECTION_FILE:-/opt/visionops_v3/config/active_camera.json}"
STALE_MS="${VISIONOPS_HP60C_WATCHDOG_STALE_MS:-5000}"
UNHEALTHY_RESTART_AFTER_S="${VISIONOPS_HP60C_WATCHDOG_UNHEALTHY_RESTART_AFTER_S:-30}"
COOLDOWN_S="${VISIONOPS_HP60C_WATCHDOG_COOLDOWN_S:-60}"
RECOVERY_WAIT_S="${VISIONOPS_HP60C_WATCHDOG_RECOVERY_WAIT_S:-20}"
MAX_SERVICE_RESTARTS="${VISIONOPS_HP60C_WATCHDOG_MAX_SERVICE_RESTARTS:-10}"
REBOOT_ENABLED="${VISIONOPS_HP60C_WATCHDOG_REBOOT_ENABLED:-true}"
REBOOT_DELAY_S="${VISIONOPS_HP60C_WATCHDOG_REBOOT_DELAY_S:-5}"
PERSIST_DIR="${VISIONOPS_HP60C_WATCHDOG_PERSIST_DIR:-/var/lib/visionops_v3/watchdog}"
LOCK_FILE="/run/visionops-hp60c-watchdog.lock"
INCIDENT_FILE="${PERSIST_DIR}/hp60c_incident_started_at"
FAIL_FILE="${PERSIST_DIR}/hp60c_failed_service_restarts"
REBOOT_FILE="${PERSIST_DIR}/hp60c_reboot_issued"
STAMP_FILE="/run/visionops-hp60c-watchdog.last_restart"
DISABLE_FILE="/run/visionops-hp60c-watchdog.disabled"

log() { logger -t visionops-hp60c-watchdog -- "$*"; echo "[visionops-hp60c-watchdog] $*"; }
is_true() { case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }
uint() { [[ "${1:-}" =~ ^[0-9]+$ ]] && printf '%s' "$1" || printf '%s' "${2:-0}"; }
write_atomic() { local p="$1" v="$2" t="${p}.tmp.$$"; printf '%s\n' "$v" >"$t"; mv -f "$t" "$p"; }
read_count() { local v=0; [[ -r "$1" ]] && v="$(cat "$1" 2>/dev/null || echo 0)"; uint "$v" 0; }

[[ -e "$DISABLE_FILE" ]] && exit 0
# The tube-pick installer may install this timer before the optional HP60C Bridge.
# A missing/disabled bridge is not a camera fault and must never advance reboot counters.
if ! systemctl cat "$BRIDGE_SERVICE" >/dev/null 2>&1; then
  exit 0
fi
if ! systemctl is-enabled --quiet "$BRIDGE_SERVICE" 2>/dev/null    && ! systemctl is-active --quiet "$BRIDGE_SERVICE" 2>/dev/null; then
  exit 0
fi
mkdir -p "$PERSIST_DIR"
exec 9>"$LOCK_FILE"; flock -n 9 || exit 0
now="$(date +%s)"
health="$(curl -fsS --max-time 3 "$BRIDGE_URL/health" 2>/dev/null || true)"
result="$(HEALTH="$health" STALE_MS="$STALE_MS" python3 - <<'PY' 2>/dev/null || echo '0 0 unknown -1 -1'
import json, os
try: d=json.loads(os.environ.get('HEALTH') or '{}')
except Exception: d={}
reachable=bool(d)
try: ca=int(d.get('last_color_age_ms',-1)); da=int(d.get('last_depth_age_ms',-1))
except Exception: ca=da=-1
fresh=d.get('camera_connected') is True and 0<=ca<=int(os.environ['STALE_MS']) and 0<=da<=int(os.environ['STALE_MS'])
print(int(reachable),int(fresh),str(d.get('camera_state') or 'unknown'),ca,da)
PY
)"
read -r reachable healthy state color_age depth_age <<<"$result"
if [[ "$healthy" == "1" ]]; then
  rm -f "$INCIDENT_FILE" "$REBOOT_FILE"
  write_atomic "$FAIL_FILE" 0
  exit 0
fi
[[ -r "$INCIDENT_FILE" ]] || write_atomic "$INCIDENT_FILE" "$now"
incident="$(read_count "$INCIDENT_FILE")"
if (( now - incident < UNHEALTHY_RESTART_AFTER_S )); then exit 0; fi
last=0; [[ -r "$STAMP_FILE" ]] && last="$(read_count "$STAMP_FILE")"
if (( now - last < COOLDOWN_S )); then exit 0; fi
write_atomic "$STAMP_FILE" "$now"
log "HP60C unhealthy: reachable=$reachable state=$state color_age_ms=$color_age depth_age_ms=$depth_age; restarting $BRIDGE_SERVICE"
systemctl restart "$BRIDGE_SERVICE" || true
recovered=0
for _ in $(seq 1 $((RECOVERY_WAIT_S * 2))); do
  sleep 0.5
  health="$(curl -fsS --max-time 1 "$BRIDGE_URL/health" 2>/dev/null || true)"
  if HEALTH="$health" STALE_MS="$STALE_MS" python3 - <<'PY' >/dev/null 2>&1
import json, os
d=json.loads(os.environ.get('HEALTH') or '{}')
ca=int(d.get('last_color_age_ms',-1)); da=int(d.get('last_depth_age_ms',-1))
raise SystemExit(0 if d.get('camera_connected') is True and 0<=ca<=int(os.environ['STALE_MS']) and 0<=da<=int(os.environ['STALE_MS']) else 1)
PY
  then recovered=1; break; fi
done
if [[ "$recovered" == "1" ]]; then
  log "HP60C recovered"
  rm -f "$INCIDENT_FILE" "$REBOOT_FILE"; write_atomic "$FAIL_FILE" 0
  active="$(python3 - "$SELECTION_FILE" <<'PY' 2>/dev/null || true
import json,sys
try: print(json.load(open(sys.argv[1],encoding='utf-8')).get('active_camera',''))
except Exception: pass
PY
)"
  if [[ "$active" == "hp60c" ]]; then
    for svc in visionops-v3-runtime-partition.service visionops-v3-runtime-tube.service visionops-v3-runtime-pick.service visionops-v3-carton-palletizing-runtime.service; do
      systemctl is-active --quiet "$svc" && systemctl try-restart "$svc" || true
    done
  fi
  exit 0
fi
count="$(read_count "$FAIL_FILE")"; count=$((count+1)); write_atomic "$FAIL_FILE" "$count"
log "HP60C service restart failed: $count/$MAX_SERVICE_RESTARTS"
if (( MAX_SERVICE_RESTARTS > 0 && count >= MAX_SERVICE_RESTARTS )) && is_true "$REBOOT_ENABLED" && [[ ! -e "$REBOOT_FILE" ]]; then
  write_atomic "$REBOOT_FILE" "timestamp=$now failures=$count"
  log "CRITICAL: HP60C failed after $count service restarts; rebooting box in ${REBOOT_DELAY_S}s"
  sleep "$REBOOT_DELAY_S"
  systemctl reboot --no-block || rm -f "$REBOOT_FILE"
fi
