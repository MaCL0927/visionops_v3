#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps v3 RK3576 edge environment bootstrap
#
# This script prepares the operating-system tools and the dedicated Python
# runtime used by an already-cloned VisionOps v3 repository. It does not copy
# code and it does not install a production task/profile.
#
# Default layout:
#   repository: /opt/visionops_v3
#   venv:       /opt/visionops_v3/venv
#
# Usage:
#   sudo bash scripts/setup_edge_env.sh
#   sudo bash scripts/setup_edge_env.sh --with-dev
#   sudo bash scripts/setup_edge_env.sh --recreate
#   sudo bash scripts/setup_edge_env.sh --verify-only
#
# Environment overrides:
#   VISIONOPS_V3_ROOT=/opt/visionops_v3
#   VISIONOPS_VENV=/opt/visionops_v3/venv
#   VISIONOPS_SERVICE_USER=neardi
#   VISIONOPS_PIP_INDEX_URL=https://pypi.org/simple
# ============================================================

ROOT="${VISIONOPS_V3_ROOT:-/opt/visionops_v3}"
VENV="${VISIONOPS_VENV:-${ROOT}/venv}"
RUNTIME_REQUIREMENTS="${ROOT}/requirements/edge-runtime.txt"
DEV_REQUIREMENTS="${ROOT}/requirements/edge-dev.txt"
WITH_DEV=0
RECREATE=0
SKIP_APT=0
VERIFY_ONLY=0

usage() {
  cat <<'USAGE'
用法：
  sudo bash scripts/setup_edge_env.sh [选项]

选项：
  --with-dev    额外安装 pytest 等边缘端开发/测试依赖
  --recreate    删除并重新创建 /opt/visionops_v3/venv
  --skip-apt    跳过 apt-get，仅创建/更新 Python venv
  --verify-only 不安装任何内容，只验证现有 v3 环境
  -h, --help    显示帮助
USAGE
}

log_info() { echo "[INFO] $*"; }
log_ok() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-dev) WITH_DEV=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "不支持的参数: $1"; usage; exit 2 ;;
  esac
done

if [[ ${EUID} -ne 0 ]]; then
  log_error "请使用 sudo 运行本脚本"
  exit 1
fi

if [[ ! -d "${ROOT}" ]]; then
  log_error "未找到 v3 仓库: ${ROOT}"
  log_error "请先将仓库 git clone 到 /opt/visionops_v3，或设置 VISIONOPS_V3_ROOT"
  exit 1
fi
if [[ ! -f "${RUNTIME_REQUIREMENTS}" ]]; then
  log_error "缺少依赖文件: ${RUNTIME_REQUIREMENTS}"
  exit 1
fi

if [[ -n "${VISIONOPS_SERVICE_USER:-}" ]]; then
  SERVICE_USER="${VISIONOPS_SERVICE_USER}"
elif [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  SERVICE_USER="${SUDO_USER}"
elif id neardi >/dev/null 2>&1; then
  SERVICE_USER="neardi"
else
  SERVICE_USER="root"
fi

install_system_packages() {
  if [[ ${SKIP_APT} -eq 1 ]]; then
    log_warn "已跳过 apt-get"
    return
  fi

  log_info "安装 RK3576 边缘端常用系统工具与编译依赖..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config make gcc g++ \
    git rsync curl wget ca-certificates gnupg \
    unzip zip tar xz-utils \
    vim nano tmux htop lsof jq tree \
    net-tools iproute2 iputils-ping tcpdump ethtool \
    openssh-client openssh-server sshpass \
    usbutils udev v4l-utils ffmpeg \
    python3 python3-pip python3-venv python3-dev python3-setuptools \
    python3-testresources \
    python3-numpy python3-opencv python3-yaml python3-psutil \
    python3-requests \
    libopencv-dev libyaml-cpp-dev libgomp1 \
    libjpeg-dev zlib1g-dev libssl-dev libffi-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1

  # These improve upload/deployment convenience but have working fallbacks.
  local optional_packages=(git-lfs python3-paramiko)
  local package
  for package in "${optional_packages[@]}"; do
    if apt-cache show "${package}" >/dev/null 2>&1; then
      if ! apt-get install -y --no-install-recommends "${package}"; then
        log_warn "可选工具安装失败，继续执行: ${package}"
      fi
    else
      log_warn "当前 apt 源没有可选工具，继续执行: ${package}"
    fi
  done
  command -v git-lfs >/dev/null 2>&1 && git lfs install --system >/dev/null 2>&1 || true
  log_ok "系统依赖安装完成"
}

check_python_version() {
  python3 - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit("VisionOps v3 edge requires Python >= 3.8")
print(f"[OK] system python: {sys.version.split()[0]}")
PY
}

create_or_update_venv() {
  if [[ ${RECREATE} -eq 1 && -e "${VENV}" ]]; then
    log_info "删除旧 venv: ${VENV}"
    rm -rf "${VENV}"
  fi

  if [[ ! -x "${VENV}/bin/python3" ]]; then
    log_info "创建 v3 venv: ${VENV}"
    # Reuse Ubuntu's tested aarch64 numpy/cv2 binaries. Installing pip OpenCV
    # over the apt build can produce ABI conflicts on RK3576.
    python3 -m venv --system-site-packages "${VENV}"
  else
    log_info "更新现有 v3 venv: ${VENV}"
  fi

  local include_system
  include_system="$(sed -n 's/^include-system-site-packages[[:space:]]*=[[:space:]]*//p' "${VENV}/pyvenv.cfg" 2>/dev/null | tr '[:upper:]' '[:lower:]')"
  if [[ "${include_system}" != "true" ]]; then
    log_error "现有 venv 未启用 system-site-packages，无法稳定复用 RK3576 的 apt OpenCV"
    log_error "请重新执行: sudo bash scripts/setup_edge_env.sh --recreate"
    exit 1
  fi

  local pip_args=()
  if [[ -n "${VISIONOPS_PIP_INDEX_URL:-}" ]]; then
    pip_args+=(--index-url "${VISIONOPS_PIP_INDEX_URL}")
  fi

  log_info "更新 pip 基础工具（保持 Python 3.8 兼容）..."
  "${VENV}/bin/python3" -m pip install "${pip_args[@]}" --upgrade \
    'pip==24.3.1' 'setuptools==70.3.0' 'wheel==0.44.0'

  log_info "安装 VisionOps v3 边缘端 Python 依赖..."
  "${VENV}/bin/python3" -m pip install "${pip_args[@]}" --no-cache-dir \
    -r "${RUNTIME_REQUIREMENTS}"

  if [[ ${WITH_DEV} -eq 1 ]]; then
    "${VENV}/bin/python3" -m pip install "${pip_args[@]}" --no-cache-dir \
      -r "${DEV_REQUIREMENTS}"
  fi

  chmod -R a+rX "${VENV}"
  if id "${SERVICE_USER}" >/dev/null 2>&1; then
    chown -R "${SERVICE_USER}:${SERVICE_USER}" "${VENV}" || true
  fi
  log_ok "v3 venv 已准备完成"
}

verify_environment() {
  if [[ ! -x "${VENV}/bin/python3" ]]; then
    log_error "未找到 v3 Python: ${VENV}/bin/python3"
    log_error "请先运行本脚本创建环境，或检查 VISIONOPS_VENV"
    exit 1
  fi

  log_info "验证边缘端 Python 依赖与 v3 核心模块..."
  VISIONOPS_V3_ROOT="${ROOT}" "${VENV}/bin/python3" - <<'PY'
import compileall
import importlib
import os
import sys
from pathlib import Path

root = Path(os.environ["VISIONOPS_V3_ROOT"])
sys.path.insert(0, str(root))

required_modules = (
    "cv2",
    "numpy",
    "yaml",
    "psutil",
    "requests",
    "packaging",
    "typing_extensions",
)
loaded = {name: importlib.import_module(name) for name in required_modules}

# Import the real public symbols used by the current repository. The old check
# referenced load_line_config, which has never existed in M25.3; the public
# function in both production profiles is named load_config.
from apps.collector_web.backend.config_loader import CollectorConfig  # noqa: F401
from production.carton_line.gateway.config import load_config as load_carton_line_config
from production.carton_palletizing.config import load_config as load_palletizing_config

if not callable(load_carton_line_config):
    raise RuntimeError("carton_line load_config is not callable")
if not callable(load_palletizing_config):
    raise RuntimeError("carton_palletizing load_config is not callable")

# Catch syntax/import regressions in the edge-facing Python tree without
# starting cameras, runtimes, HTTP servers or robot gateways.
for relative in ("apps/collector_web", "production", "edge/camera_bridge"):
    path = root / relative
    if path.exists() and not compileall.compile_dir(str(path), quiet=1):
        raise RuntimeError(f"compileall failed: {path}")

try:
    paramiko = importlib.import_module("paramiko")
except ImportError:
    paramiko = None

print(f"python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
print(f"numpy={loaded['numpy'].__version__}")
print(f"opencv={loaded['cv2'].__version__}")
print(f"pyyaml={loaded['yaml'].__version__}")
print(f"psutil={loaded['psutil'].__version__}")
print(f"requests={loaded['requests'].__version__}")
print(f"packaging={loaded['packaging'].__version__}")
print(f"typing_extensions={getattr(loaded['typing_extensions'], '__version__', 'installed')}")
print(f"paramiko={paramiko.__version__ if paramiko else 'not-installed (ssh/scp fallback available)'}")
print("v3_core_imports=ok")
print("compileall=ok")
PY

  local required_commands=(cmake g++ pkg-config git rsync curl ssh scp ffmpeg ip ethtool jq)
  local missing_commands=()
  local command_name
  for command_name in "${required_commands[@]}"; do
    command -v "${command_name}" >/dev/null 2>&1 || missing_commands+=("${command_name}")
  done
  if [[ ${#missing_commands[@]} -gt 0 ]]; then
    log_error "缺少系统命令: ${missing_commands[*]}"
    log_error "请不要使用 --skip-apt，重新执行环境脚本"
    exit 1
  fi

  if ! command -v sshpass >/dev/null 2>&1; then
    log_warn "未安装 sshpass；使用 SSH 密码上传时需要 paramiko 或 sshpass"
  fi
  if ! python3 -c 'import testresources' >/dev/null 2>&1; then
    log_warn "系统缺少 testresources；这只会触发 launchpadlib 的 pip 元数据警告，不影响 VisionOps 运行"
  fi

  if ! pkg-config --exists opencv4 2>/dev/null; then
    log_warn "pkg-config 未找到 opencv4；Python cv2 可用，但编译 C++ Runtime/相机 Bridge 可能失败"
  fi
  if [[ ! -f /usr/include/rknn_api.h && ! -f /usr/local/include/rknn_api.h ]]; then
    log_warn "未检测到 rknn_api.h；编译真实 RKNN Runtime 前仍需安装板厂 RKNN runtime/SDK"
  fi
  if ! ldconfig -p 2>/dev/null | grep -q 'librknnrt\.so'; then
    log_warn "未检测到 librknnrt.so；运行真实 RKNN Runtime 前仍需安装板厂 RKNN runtime"
  fi
  if ! ldconfig -p 2>/dev/null | grep -q 'librga\.so'; then
    log_warn "未检测到 librga.so；启用 RGA 预处理前仍需安装板厂 RGA runtime"
  fi
  log_ok "环境验证通过"
}

print_summary() {
  cat <<EOF_SUMMARY

============================================================
VisionOps v3 边缘端环境创建完成
============================================================
仓库目录: ${ROOT}
Python venv: ${VENV}
运行用户:   ${SERVICE_USER}
Python:     ${VENV}/bin/python3

激活环境（仅用于人工调试）:
  source ${VENV}/bin/activate

再次验证（不安装任何内容）:
  sudo bash ${ROOT}/scripts/setup_edge_env.sh --verify-only

验证单条命令:
  ${VENV}/bin/python3 -c 'import cv2,numpy,yaml,psutil,requests; print(cv2.__version__)'

后续生产脚本默认读取:
  VISIONOPS_VENV=${VENV}

说明:
  - v3 边缘端 C++ RKNN Runtime 不需要 fastapi/uvicorn/rknn-toolkit-lite2。
  - numpy/cv2 使用 Ubuntu ARM64 二进制包，避免 pip OpenCV ABI 冲突。
  - RKNN、RGA、Orbbec SDK 和 HP60C SDK 属于板厂/相机运行库，
    本脚本只检查，不会用 apt 自动替代这些专有组件。
EOF_SUMMARY
}

main() {
  check_python_version
  if [[ ${VERIFY_ONLY} -eq 0 ]]; then
    install_system_packages
    create_or_update_venv
  fi
  verify_environment
  find "${ROOT}/scripts" "${ROOT}/production" -type f -name '*.sh' -exec chmod +x {} + 2>/dev/null || true
  print_summary
}

main "$@"
