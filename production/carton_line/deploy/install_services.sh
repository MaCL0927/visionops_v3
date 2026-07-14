#!/usr/bin/env bash
set -euo pipefail

ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
UNIT_DIR="${VISIONOPS_SYSTEMD_UNIT_DIR:-/etc/systemd/system}"
CONFIG_DIR="${VISIONOPS_CONFIG_DIR:-/etc/visionops_v3}"
SYSTEMCTL_BIN="${VISIONOPS_SYSTEMCTL_BIN:-systemctl}"
LINE_CONFIG="${CONFIG_DIR}/carton_line.yaml"
ENV_FILE="${CONFIG_DIR}/carton_line.env"
SOURCE_DIR="${ROOT}/production/carton_line"
PROFILE=""

PARTITION_TUBE_UNITS=(
  visionops-v3-runtime-partition.service
  visionops-v3-runtime-tube.service
  visionops-v3-robot-gateway.service
  visionops-v3-collector-partition.service
  visionops-v3-collector-tube.service
)
PARTITION_TUBE_ENABLE_UNITS=("${PARTITION_TUBE_UNITS[@]}")

TUBE_PICK_UNITS=(
  visionops-v3-runtime-pick.service
  visionops-v3-ws-pick.service
  visionops-v3-collector-pick.service
  visionops-v3-runtime-pick-watchdog.service
  visionops-v3-runtime-pick-watchdog.timer
)
TUBE_PICK_ENABLE_UNITS=(
  visionops-v3-runtime-pick.service
  visionops-v3-ws-pick.service
  visionops-v3-collector-pick.service
  visionops-v3-runtime-pick-watchdog.timer
)

LEGACY_TUBE_PICK_UNITS=(
  visionops-v3-tcp-pick.service
)

usage() {
  cat <<'USAGE'
用法：
  sudo bash production/carton_line/deploy/install_services.sh --profile <profile>

支持的 profile：
  partition-tube  安装“纸隔板 + 纸筒产品”板卡所需服务：
                  partition Runtime、tube Runtime、Modbus Gateway、两个 Collector

  tube-pick       安装“纸筒产品 / 大隔板 / 倒伏纸筒检测”板卡所需服务：
                  pick Runtime、WebSocket Server、pick Collector、帧流 watchdog timer

示例：
  sudo bash production/carton_line/deploy/install_services.sh --profile partition-tube
  sudo bash production/carton_line/deploy/install_services.sh --profile tube-pick

说明：
  - 安装脚本只复制并启用当前 profile 的 systemd unit。
  - 如果板卡以前安装过另一个 profile，脚本会停止、禁用并移除另一组 unit。
  - 安装完成后不会自动启动服务，请按输出提示手动 start，便于先检查配置和模型。
USAGE
}

log_info() { echo "[INFO] $*"; }
log_ok() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        log_error "--profile 缺少参数"
        usage
        exit 2
      fi
      PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "不支持的参数: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${PROFILE}" ]]; then
  log_error "必须指定 --profile partition-tube 或 --profile tube-pick"
  usage
  exit 2
fi

case "${PROFILE}" in
  partition-tube)
    SELECTED_UNITS=("${PARTITION_TUBE_UNITS[@]}")
    SELECTED_ENABLE_UNITS=("${PARTITION_TUBE_ENABLE_UNITS[@]}")
    UNSELECTED_UNITS=("${TUBE_PICK_UNITS[@]}")
    PROFILE_DESCRIPTION="纸隔板 + 纸筒产品（Modbus）"
    ;;
  tube-pick)
    SELECTED_UNITS=("${TUBE_PICK_UNITS[@]}")
    SELECTED_ENABLE_UNITS=("${TUBE_PICK_ENABLE_UNITS[@]}")
    UNSELECTED_UNITS=("${PARTITION_TUBE_UNITS[@]}")
    PROFILE_DESCRIPTION="纸筒产品 / 大隔板 / 倒伏纸筒检测（WebSocket + MJPEG）"
    ;;
  *)
    log_error "未知 profile: ${PROFILE}"
    usage
    exit 2
    ;;
esac

if [[ ${EUID} -ne 0 ]]; then
  log_error "请使用 sudo 运行"
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  log_error "未找到生产线目录: ${SOURCE_DIR}"
  log_error "请确认 VISIONOPS_V3_ROOT 或仓库安装路径"
  exit 1
fi

for required in \
  "${SOURCE_DIR}/config/line.yaml" \
  "${SOURCE_DIR}/deploy/merge_line_config.py"; do
  if [[ ! -f "${required}" ]]; then
    log_error "缺少必要文件: ${required}"
    exit 1
  fi
done

for unit in "${SELECTED_UNITS[@]}"; do
  source_unit="${SOURCE_DIR}/deploy/systemd/${unit}"
  if [[ ! -f "${source_unit}" ]]; then
    log_error "当前 profile 缺少 systemd unit: ${source_unit}"
    exit 1
  fi
done

log_info "安装 profile: ${PROFILE}"
log_info "任务说明: ${PROFILE_DESCRIPTION}"
log_info "仓库目录: ${ROOT}"
log_info "配置目录: ${CONFIG_DIR}"

install -d -m 0755 "${CONFIG_DIR}" "${UNIT_DIR}"

PYTHON_BIN="${VISIONOPS_VENV:-/opt/visionops/venv}/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python3
fi

"${PYTHON_BIN}" "${SOURCE_DIR}/deploy/merge_line_config.py" \
  --template "${SOURCE_DIR}/config/line.yaml" \
  --target "${LINE_CONFIG}" \
  --drop-path pick.tcp

install -m 0644 \
  "${SOURCE_DIR}/config/line.yaml" \
  "${CONFIG_DIR}/carton_line.yaml.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<EOF_ENV
VISIONOPS_V3_ROOT=${ROOT}
VISIONOPS_VENV=/opt/visionops/venv
VISIONOPS_CARTON_LINE_CONFIG=${LINE_CONFIG}
VISIONOPS_CAMERA_BRIDGE_URL=http://127.0.0.1:18182
VISIONOPS_PARTITION_MODEL_DIR=${ROOT}/models/carton_partition_check/current
VISIONOPS_TUBE_MODEL_DIR=${ROOT}/models/carton_tube_check/current
VISIONOPS_PICK_MODEL_DIR=${ROOT}/models/tube_pick_vision/current
VISIONOPS_PICK_RUNTIME_URL=http://127.0.0.1:28083
VISIONOPS_CAMERA_BRIDGE_SERVICE=visionops-orbbec336l-bridge.service
VISIONOPS_PICK_RUNTIME_SERVICE=visionops-v3-runtime-pick.service
VISIONOPS_PICK_WS_SERVICE=visionops-v3-ws-pick.service
VISIONOPS_PICK_WATCHDOG_STALE_MS=5000
VISIONOPS_PICK_WATCHDOG_COOLDOWN_S=30
VISIONOPS_PICK_WATCHDOG_RECOVERY_WAIT_S=3
EOF_ENV
elif ! grep -q '^VISIONOPS_PICK_MODEL_DIR=' "${ENV_FILE}"; then
  printf '\nVISIONOPS_PICK_MODEL_DIR=%s/models/tube_pick_vision/current\n' "${ROOT}" >> "${ENV_FILE}"
fi

append_env_default() {
  local key="$1"
  local value="$2"
  if ! grep -q "^${key}=" "${ENV_FILE}"; then
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}
append_env_default VISIONOPS_PICK_RUNTIME_URL http://127.0.0.1:28083
append_env_default VISIONOPS_CAMERA_BRIDGE_SERVICE visionops-orbbec336l-bridge.service
append_env_default VISIONOPS_PICK_RUNTIME_SERVICE visionops-v3-runtime-pick.service
append_env_default VISIONOPS_PICK_WS_SERVICE visionops-v3-ws-pick.service
append_env_default VISIONOPS_PICK_WATCHDOG_STALE_MS 5000
append_env_default VISIONOPS_PICK_WATCHDOG_COOLDOWN_S 30
append_env_default VISIONOPS_PICK_WATCHDOG_RECOVERY_WAIT_S 3

log_info "清理旧 tube_pick TCP Client 服务..."
for unit in "${LEGACY_TUBE_PICK_UNITS[@]}"; do
  "${SYSTEMCTL_BIN}" disable --now "${unit}" >/dev/null 2>&1 || true
  if [[ -e "${UNIT_DIR}/${unit}" || -L "${UNIT_DIR}/${unit}" ]]; then
    rm -f "${UNIT_DIR}/${unit}"
    log_info "已移除旧服务: ${unit}"
  fi
done

log_info "清理非当前 profile 的服务..."
for unit in "${UNSELECTED_UNITS[@]}"; do
  installed_unit="${UNIT_DIR}/${unit}"
  if [[ -e "${installed_unit}" || -L "${installed_unit}" ]]; then
    "${SYSTEMCTL_BIN}" disable --now "${unit}" >/dev/null 2>&1 || true
    rm -f "${installed_unit}"
    log_info "已移除: ${unit}"
  else
    # 兼容旧安装产生的 enable 链接或 systemd 已加载但文件已被手工删除的情况。
    "${SYSTEMCTL_BIN}" disable --now "${unit}" >/dev/null 2>&1 || true
  fi
done

log_info "安装当前 profile 的服务..."
for unit in "${SELECTED_UNITS[@]}"; do
  install -m 0644 \
    "${SOURCE_DIR}/deploy/systemd/${unit}" \
    "${UNIT_DIR}/${unit}"
  log_info "已安装: ${unit}"
done

"${SYSTEMCTL_BIN}" daemon-reload
"${SYSTEMCTL_BIN}" enable "${SELECTED_ENABLE_UNITS[@]}"

log_ok "profile 安装完成: ${PROFILE}"
echo "Line config: ${LINE_CONFIG}"
echo "Environment: ${ENV_FILE}"
echo
printf '已安装的 unit:\n'
printf '  - %s\n' "${SELECTED_UNITS[@]}"
echo
printf '已启用的服务/timer:\n'
printf '  - %s\n' "${SELECTED_ENABLE_UNITS[@]}"
echo
printf '启动命令:\n  sudo systemctl start'
printf ' %s' "${SELECTED_ENABLE_UNITS[@]}"
printf '\n\n状态检查:\n  systemctl status'
printf ' %s' "${SELECTED_ENABLE_UNITS[@]}"
printf '\n'
