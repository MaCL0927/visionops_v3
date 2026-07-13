#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 PC -> RK3576/LB3576 快速同步脚本
# 默认同步边缘端代码；可选同步标准模型包 model.rknn + model.yaml。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="${SCRIPT_DIR}"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/CMakeLists.txt" \
       && -d "${dir}/apps/collector_web" \
       && -d "${dir}/edge/runtime_cpp" \
       && -d "${dir}/production" ]]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"
TARGET_HOST=""
TARGET_USER=""
TARGET_PORT="22"
TARGET_DIR="/opt/visionops_v3"

MODE="code"                    # code | models | all
DRY_RUN="false"
DELETE_CODE="false"
SYNC_DOCS="false"
VERIFY_REMOTE="true"
MODEL_SELECTIONS=()

usage() {
  cat <<'USAGE'
用法：
  bash edge/deploy/push.sh --host <ip-or-host> --user <ssh-user> [options]

常用示例：
  # 默认：同步所有边缘端代码
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi

  # 先预览，再同步并清理旧代码文件
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi --dry-run --delete
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi --delete

  # 只同步所有标准模型包
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi --mode models

  # 只同步指定模型包，可重复指定 --model
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi \
    --mode models \
    --model carton_partition_check/current \
    --model carton_tube_check/current \
    --model tube_pick_vision/current

  # 同步代码和模型
  bash edge/deploy/push.sh --host 192.168.1.120 --user neardi --mode all

参数：
  --host <host>          目标 3576 IP 或主机名，必填
  --user <user>          SSH 用户名，必填
  --port <port>          SSH 端口，默认 22
  --target-dir <path>    远端目录，默认 /opt/visionops_v3
  --mode <mode>          code、models、all，默认 code
  --sync-models          兼容旧参数，等价于 --mode all
  --model <path>         指定 models/ 下的模型包，可重复
  --dry-run              仅预览，不真正写入
  --delete               清理受脚本管理的代码目录中的旧文件
  --sync-docs            额外同步 docs/architecture 和 docs/migration
  --no-verify            不执行同步后的关键文件检查
  -h, --help             显示帮助

默认同步：
  apps/collector_web、edge、interfaces、production
  configs/app、configs/edge、configs/runtime、configs/task
  通用 Runtime/Collector 启动脚本和边缘诊断工具
  CMakeLists.txt、README.md、AGENTS.md、.gitignore

默认不同步：
  apps/server_api、configs/server、training、tests、server_data
  服务端存储工具、编译目录、缓存、日志、实际 *.env
  PT/ONNX、归档文件；RKNN 模型需使用 models/all 模式
USAGE
}

info()  { printf '[INFO] %s\n' "$*"; }
ok()    { printf '[OK] %s\n' "$*"; }
warn()  { printf '[WARN] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*" >&2; }
die()   { error "$*"; exit 1; }

need_value() {
  [[ -n "${2:-}" ]] || die "$1 后缺少参数"
}

normalize_model_path() {
  local value="$1"
  value="${value#./}"
  value="${value#models/}"
  value="${value%/}"
  [[ -n "${value}" ]] || die "--model 不能是空路径"
  [[ "${value}" != /* && "${value}" != *".."* ]] \
    || die "--model 必须是 models/ 下的安全相对路径: ${value}"
  printf '%s\n' "${value}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        need_value "$1" "${2:-}"; TARGET_HOST="$2"; shift 2 ;;
    --user)        need_value "$1" "${2:-}"; TARGET_USER="$2"; shift 2 ;;
    --port)        need_value "$1" "${2:-}"; TARGET_PORT="$2"; shift 2 ;;
    --target-dir)  need_value "$1" "${2:-}"; TARGET_DIR="${2%/}"; shift 2 ;;
    --mode)        need_value "$1" "${2:-}"; MODE="$2"; shift 2 ;;
    --sync-models) MODE="all"; shift ;;
    --model)
      need_value "$1" "${2:-}"
      MODEL_SELECTIONS+=("$(normalize_model_path "$2")")
      shift 2
      ;;
    --dry-run)   DRY_RUN="true"; shift ;;
    --delete)    DELETE_CODE="true"; shift ;;
    --sync-docs) SYNC_DOCS="true"; shift ;;
    --no-verify) VERIFY_REMOTE="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "不支持的参数: $1；使用 --help 查看用法" ;;
  esac
done

[[ -n "${REPO_ROOT}" ]] \
  || die "无法定位仓库根目录，请将脚本放到 edge/deploy/push.sh"
[[ -n "${TARGET_HOST}" ]] || { usage; die "--host 必填"; }
[[ -n "${TARGET_USER}" ]] || { usage; die "--user 必填"; }
[[ "${TARGET_PORT}" =~ ^[0-9]+$ ]] || die "--port 必须是整数"
[[ "${TARGET_DIR}" == /* ]] || die "--target-dir 必须是绝对路径"
case "${MODE}" in code|models|all) ;; *) die "--mode 只能是 code、models 或 all" ;; esac
[[ ${#MODEL_SELECTIONS[@]} -eq 0 || "${MODE}" != "code" ]] \
  || die "使用 --model 时请同时指定 --mode models 或 --mode all"

command -v ssh >/dev/null 2>&1 || die "缺少 ssh"
command -v rsync >/dev/null 2>&1 || die "缺少 rsync"
cd "${REPO_ROOT}"

# 当前结构下全部边缘端代码。服务端、训练和存储维护工具不在此清单。
CODE_DIRS=(
  "apps/collector_web"
  "edge"
  "interfaces"
  "configs/app"
  "configs/edge"
  "configs/runtime"
  "configs/task"
  "production"
  "tools/config"
  "tools/interfaces"
)
CODE_FILES=(
  "apps/__init__.py"
  "scripts/start_runtime.sh"
  "scripts/start_collector.sh"
  "tools/benchmark_runtime.py"
  "CMakeLists.txt"
  "README.md"
  "AGENTS.md"
  ".gitignore"
)
if [[ "${SYNC_DOCS}" == "true" ]]; then
  CODE_DIRS+=("docs/architecture" "docs/migration")
fi

# 用于本地和远端校验，防止目录重构后脚本静默漏传关键模块。
REQUIRED_EDGE_FILES=(
  "apps/collector_web/backend/main.py"
  "apps/collector_web/frontend/index.html"
  "edge/runtime_cpp/CMakeLists.txt"
  "edge/camera_bridge/orbbec336l_bridge/visionops_orbbec336l_bridge.cpp"
  "edge/modbus_adapter/modbus_tcp_server.py"
  "interfaces/schemas/inference_result.schema.json"
  "production/carton_line/config/line.yaml"
  "production/carton_line/gateway/service.py"
  "production/carton_line/tasks/carton_partition_check/algorithm.py"
  "production/carton_line/tasks/carton_tube_check/algorithm.py"
  "production/carton_line/tasks/tube_pick_vision/service.py"
  "production/carton_line/tasks/tube_pick_vision/websocket_server.py"
  "production/carton_line/scripts/start_runtime.sh"
  "production/carton_line/scripts/start_gateway.sh"
  "production/carton_line/scripts/start_ws_pick.sh"
  "production/carton_line/deploy/install_services.sh"
)

COMMON_EXCLUDES=(
  --exclude ".git/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude ".cache/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "*.pyo"
  --exclude "*.swp"
  --exclude "*.swo"
  --exclude "*~"
  --exclude ".DS_Store"
  --exclude "Thumbs.db"
  --exclude "build/"
  --exclude "build-*/"
  --exclude "cmake-build-*/"
  --exclude "CMakeFiles/"
  --exclude "CMakeCache.txt"
  --exclude "Makefile"
  --exclude "*.log"
  --exclude "logs/"
  --exclude "run/"
  --exclude "runtime/"
  --exclude "tmp/"
  --exclude "*.pid"
  # 保留板端实际环境文件；*.env.example 不受影响。
  --exclude "*.env"
  --exclude ".env"
  --exclude "*.local.yaml"
  --exclude "*.override.yaml"
  # 代码模式不夹带模型、训练产物和归档。
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.onnx"
  --exclude "*.rknn"
  --exclude "*.tar"
  --exclude "*.tar.gz"
  --exclude "*.tgz"
  --exclude "*.zip"
  --exclude "*.7z"
)

# SSH 复用，避免每同步一个目录都重新输入密码。
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/visionops-push.XXXXXX")"
CONTROL_PATH="${CONTROL_DIR}/cm-%C"
REMOTE="${TARGET_USER}@${TARGET_HOST}"
SSH=(
  ssh
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=10
  -o ControlMaster=auto
  -o ControlPersist=60
  -o "ControlPath=${CONTROL_PATH}"
  -p "${TARGET_PORT}"
)
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=${CONTROL_PATH} -p ${TARGET_PORT}"

cleanup() {
  "${SSH[@]}" -O exit "${REMOTE}" >/dev/null 2>&1 || true
  rm -rf -- "${CONTROL_DIR}"
}
trap cleanup EXIT

remote_mkdir() {
  [[ "${DRY_RUN}" == "false" ]] || return 0
  local quoted
  printf -v quoted '%q' "$1"
  "${SSH[@]}" "${REMOTE}" "mkdir -p -- ${quoted}"
}

base_rsync_args() {
  RSYNC_ARGS=(
    -az
    --safe-links
    --partial
    --delay-updates
    --human-readable
    --info=stats2,progress2
    --rsh "${RSYNC_RSH}"
  )
  [[ "${DRY_RUN}" == "false" ]] || RSYNC_ARGS+=(--dry-run --itemize-changes)
}

sync_code_dir() {
  local rel="$1"
  base_rsync_args
  [[ "${DELETE_CODE}" == "false" ]] || RSYNC_ARGS+=(--delete-delay)
  remote_mkdir "${TARGET_DIR}/${rel}"
  rsync "${RSYNC_ARGS[@]}" "${COMMON_EXCLUDES[@]}" \
    "${REPO_ROOT}/${rel}/" "${REMOTE}:${TARGET_DIR}/${rel}/"
}

sync_code_file() {
  local rel="$1"
  local dest="${TARGET_DIR}/$(dirname "${rel}")"
  base_rsync_args
  remote_mkdir "${dest}"
  rsync "${RSYNC_ARGS[@]}" \
    "${REPO_ROOT}/${rel}" "${REMOTE}:${dest}/"
}

validate_code_manifest() {
  local item
  for item in "${CODE_DIRS[@]}"; do
    [[ -d "${REPO_ROOT}/${item}" ]] || die "缺少同步目录: ${item}"
  done
  for item in "${CODE_FILES[@]}" "${REQUIRED_EDGE_FILES[@]}"; do
    [[ -f "${REPO_ROOT}/${item}" ]] || die "缺少边缘端关键文件: ${item}"
  done
}

sync_code() {
  validate_code_manifest
  info "同步边缘端代码："
  local item
  for item in "${CODE_DIRS[@]}"; do printf '  - %s/\n' "${item}"; done
  for item in "${CODE_FILES[@]}"; do printf '  - %s\n' "${item}"; done

  for item in "${CODE_DIRS[@]}"; do
    info "同步目录 ${item}/"
    sync_code_dir "${item}"
  done
  for item in "${CODE_FILES[@]}"; do
    info "同步文件 ${item}"
    sync_code_file "${item}"
  done
  ok "边缘端代码同步完成"
}

all_model_packages() {
  local root="${REPO_ROOT}/models"
  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' yaml; do
    local dir="$(dirname "${yaml}")"
    [[ -f "${dir}/model.rknn" ]] || continue
    printf '%s\0' "${dir#${root}/}"
  done < <(find -L "${root}" -type f -name model.yaml -print0)
}

sync_model_package() {
  local rel="$1"
  local src="${REPO_ROOT}/models/${rel}"
  local dst="${TARGET_DIR}/models/${rel}"
  [[ -f "${src}/model.rknn" && -f "${src}/model.yaml" ]] \
    || die "模型包必须包含 model.rknn 和 model.yaml: models/${rel}"

  remote_mkdir "${dst}"
  base_rsync_args
  # -L 可将 current 等符号链接解析成真实模型文件。
  RSYNC_ARGS+=(-L)
  rsync "${RSYNC_ARGS[@]}" \
    --include "/model.rknn" \
    --include "/model.yaml" \
    --exclude "*" \
    "${src}/" "${REMOTE}:${dst}/"
}

sync_models() {
  local packages=()
  local item
  if [[ ${#MODEL_SELECTIONS[@]} -gt 0 ]]; then
    packages=("${MODEL_SELECTIONS[@]}")
  else
    while IFS= read -r -d '' item; do packages+=("${item}"); done < <(all_model_packages)
  fi
  [[ ${#packages[@]} -gt 0 ]] \
    || die "models/ 下未找到标准模型包；也可以用 --model 指定"

  info "同步模型包（仅 model.rknn + model.yaml）："
  for item in "${packages[@]}"; do printf '  - models/%s/\n' "${item}"; done
  for item in "${packages[@]}"; do sync_model_package "${item}"; done
  ok "模型同步完成"
}

verify_remote() {
  [[ "${VERIFY_REMOTE}" == "true" && "${DRY_RUN}" == "false" ]] || return 0
  info "检查远端关键文件..."
  local file quoted
  for file in "${REQUIRED_EDGE_FILES[@]}"; do
    printf -v quoted '%q' "${TARGET_DIR}/${file}"
    "${SSH[@]}" "${REMOTE}" "test -f ${quoted}" \
      || die "远端缺少关键文件: ${TARGET_DIR}/${file}"
  done
  ok "远端关键文件检查通过"
}

warn_legacy_dir() {
  [[ "${DRY_RUN}" == "false" ]] || return 0
  local quoted
  printf -v quoted '%q' "${TARGET_DIR}/collector_web"
  if "${SSH[@]}" "${REMOTE}" "test -d ${quoted}"; then
    warn "发现旧残留目录 ${TARGET_DIR}/collector_web"
    warn "当前 Collector 位于 ${TARGET_DIR}/apps/collector_web；确认无用后手动删除旧目录。"
  fi
}

info "仓库根目录: ${REPO_ROOT}"
info "目标设备: ${REMOTE}:${TARGET_PORT}"
info "目标目录: ${TARGET_DIR}"
info "同步模式: ${MODE}"
info "检查 SSH 连接..."
"${SSH[@]}" "${REMOTE}" "echo ok" >/dev/null
ok "SSH 连接成功"
remote_mkdir "${TARGET_DIR}"

case "${MODE}" in
  code)
    sync_code
    verify_remote
    warn_legacy_dir
    ;;
  models)
    sync_models
    ;;
  all)
    sync_code
    sync_models
    verify_remote
    warn_legacy_dir
    ;;
esac

printf '\n同步完成。3576 目标目录：%s\n\n' "${TARGET_DIR}"
cat <<'EOF_SUMMARY'
常用后续操作：
  # Python、Web 或产线任务代码变化：重启对应服务
  sudo systemctl restart visionops-v3-robot-gateway.service
  sudo systemctl restart visionops-v3-ws-pick.service
  sudo systemctl restart visionops-v3-collector-partition.service
  sudo systemctl restart visionops-v3-collector-tube.service
  sudo systemctl restart visionops-v3-collector-pick.service

  # C++ Runtime 变化：在板端重新编译后重启三个 Runtime
  cmake --build build-rknn -j4
  sudo systemctl restart visionops-v3-runtime-partition.service
  sudo systemctl restart visionops-v3-runtime-tube.service
  sudo systemctl restart visionops-v3-runtime-pick.service

说明：
  - 实际 *.env 不会被覆盖。
  - --delete 只清理代码目录，不会删除远端 models/。
  - 默认不传服务端、训练、测试、server_data 和编译产物。
EOF_SUMMARY