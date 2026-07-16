#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-hp60c-sdk-bridge.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SRC_DIR/../../.." && pwd)"
DST_DIR="${VISIONOPS_HP60C_BRIDGE_DIR:-/opt/visionops_v3/edge/camera_bridge/hp60c_bridge}"
BIN_DIR="${VISIONOPS_BIN_DIR:-/opt/visionops_v3/bin}"
ENV_FILE="$DST_DIR/hp60c_sdk_bridge.env"
EXAMPLE_FILE="$DST_DIR/hp60c_sdk_bridge.env.example"
SDK_ROOT_DEFAULT="/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros/linux"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERROR] $*" >&2; }
append_default() {
  local key="$1" value="$2"
  if ! grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
    printf '%s=%s\n' "$key" "$value" | sudo tee -a "$ENV_FILE" >/dev/null
  fi
}

log "install VisionOps v3 HP60C Angstrong SDK bridge"
# Avoid two processes opening the same HP60C device.
sudo systemctl disable --now visionops-hp60c-ros1-bridge.service >/dev/null 2>&1 || true
log "SRC_DIR=$SRC_DIR"
log "DST_DIR=$DST_DIR"
sudo mkdir -p "$DST_DIR" "$BIN_DIR"
if [[ "$SRC_DIR" != "$DST_DIR" ]]; then
  sudo cp -f "$SRC_DIR/visionops_hp60c_sdk_bridge.cpp" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/CMakeLists.txt" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/hp60c_sdk_bridge.env.example" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/README.md" "$DST_DIR/" 2>/dev/null || true
  sudo cp -f "$SRC_DIR/install_hp60c_sdk_bridge_service.sh" "$DST_DIR/"
else
  log "source and destination are the same; skip self-copy"
fi
if [[ ! -f "$ENV_FILE" ]]; then
  sudo cp -f "$EXAMPLE_FILE" "$ENV_FILE"
fi
append_default VISIONOPS_HP60C_HTTP_HOST 0.0.0.0
append_default VISIONOPS_HP60C_HTTP_PORT 18181
append_default VISIONOPS_HP60C_COLOR_WIDTH 640
append_default VISIONOPS_HP60C_COLOR_HEIGHT 480
append_default VISIONOPS_HP60C_DEPTH_WIDTH 640
append_default VISIONOPS_HP60C_DEPTH_HEIGHT 480
append_default VISIONOPS_HP60C_FPS 30
append_default VISIONOPS_HP60C_JPEG_QUALITY 85
append_default VISIONOPS_HP60C_MJPEG_FPS 10
append_default VISIONOPS_HP60C_FLIP_VERTICAL true
append_default VISIONOPS_HP60C_FLIP_HORIZONTAL false
append_default VISIONOPS_HP60C_RGB_SOURCE auto
append_default VISIONOPS_HP60C_RGB_ORDER bgr
append_default VISIONOPS_HP60C_DEPTH_ALIGNED_TO_COLOR true
append_default VISIONOPS_HP60C_STALE_TIMEOUT_MS 3000
append_default VISIONOPS_HP60C_FIRST_FRAME_TIMEOUT_MS 8000
append_default VISIONOPS_HP60C_RECONNECT_INITIAL_MS 1000
append_default VISIONOPS_HP60C_RECONNECT_MAX_MS 30000
append_default VISIONOPS_HP60C_RECONNECT_FAILURE_ALARM_SEC 15
append_default VISIONOPS_HP60C_FX 0
append_default VISIONOPS_HP60C_FY 0
append_default VISIONOPS_HP60C_CX 0
append_default VISIONOPS_HP60C_CY 0

# shellcheck disable=SC1090
source "$ENV_FILE"
SDK_ROOT="${VISIONOPS_HP60C_SDK_ROOT:-$SDK_ROOT_DEFAULT}"
SDK_LIB_DIR="${VISIONOPS_HP60C_SDK_LIB_DIR:-$SDK_ROOT/libs/lib/aarch64-linux-gnu-gcc-5}"
CONFIG_FILE="${VISIONOPS_HP60C_CONFIG:-$SDK_ROOT/configurationfiles/hp60c_v2_01_20241104_configEncrypt.json}"
[[ -d "$SDK_ROOT" ]] || { err "SDK root not found: $SDK_ROOT"; exit 2; }
[[ -d "$SDK_LIB_DIR" ]] || { err "SDK lib dir not found: $SDK_LIB_DIR"; exit 3; }
[[ -f "$CONFIG_FILE" ]] || { err "HP60C config file not found: $CONFIG_FILE"; exit 4; }

sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev
if [[ -f "$SDK_ROOT/scripts/create_udev_rules.sh" ]]; then
  (cd "$SDK_ROOT/scripts" && sudo bash ./create_udev_rules.sh || true)
  sudo udevadm control --reload-rules || true
  sudo udevadm trigger || true
else
  warn "udev script not found: $SDK_ROOT/scripts/create_udev_rules.sh"
fi

BUILD_DIR="$DST_DIR/build"
sudo rm -rf "$BUILD_DIR"
sudo mkdir -p "$BUILD_DIR"
sudo chown -R "$(id -u):$(id -g)" "$BUILD_DIR"
cmake -S "$DST_DIR" -B "$BUILD_DIR" \
  -DANGSTRONG_SDK_ROOT="$SDK_ROOT" \
  -DANGSTRONG_LIB_DIR="$SDK_LIB_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"
sudo cp -f "$BUILD_DIR/visionops_hp60c_sdk_bridge" "$BIN_DIR/visionops_hp60c_sdk_bridge"
sudo chmod +x "$BIN_DIR/visionops_hp60c_sdk_bridge"

sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF
[Unit]
Description=VisionOps v3 HP60C Angstrong SDK HTTP Bridge
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory=$DST_DIR
EnvironmentFile=$ENV_FILE
Environment=LD_LIBRARY_PATH=$SDK_LIB_DIR
ExecStart=$BIN_DIR/visionops_hp60c_sdk_bridge
Restart=always
RestartSec=2
TimeoutStopSec=8
KillMode=mixed
User=root

[Install]
WantedBy=multi-user.target
EOF

# Install the external watchdog when this is a full VisionOps v3 checkout.
WATCHDOG_SERVICE="$PROJECT_ROOT/production/carton_line/deploy/systemd/visionops-hp60c-sdk-bridge-watchdog.service"
WATCHDOG_TIMER="$PROJECT_ROOT/production/carton_line/deploy/systemd/visionops-hp60c-sdk-bridge-watchdog.timer"
if [[ -f "$WATCHDOG_SERVICE" && -f "$WATCHDOG_TIMER" ]]; then
  sudo cp -f "$WATCHDOG_SERVICE" /etc/systemd/system/
  sudo cp -f "$WATCHDOG_TIMER" /etc/systemd/system/
fi
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
if [[ -f /etc/systemd/system/visionops-hp60c-sdk-bridge-watchdog.timer ]]; then
  sudo systemctl enable visionops-hp60c-sdk-bridge-watchdog.timer
fi
log "installed $SERVICE_NAME"
log "HP60C API: http://127.0.0.1:18181"
log "Orbbec API remains: http://127.0.0.1:18182"
log "NEXT: sudo systemctl restart $SERVICE_NAME"
log "CHECK: curl -s http://127.0.0.1:18181/health | python3 -m json.tool"
