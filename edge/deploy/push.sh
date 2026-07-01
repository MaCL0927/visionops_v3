#!/usr/bin/env bash
set -euo pipefail

# VisionOps v3 边缘端代码同步脚本
#
# 目标：
# - 本地修改代码后，直接把 v3 边缘端所需代码同步到 RK3576 / LB3576
# - 不依赖先 push 到 GitHub 再在板端 git pull
# - 当前阶段先同步代码与配置；模型同步参数先预留，后续扩展

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_HOST=""
TARGET_USER=""
TARGET_PORT="22"
TARGET_DIR="/opt/visionops_v3"

DRY_RUN="false"
DELETE_REMOTE="false"
SYNC_MODELS="false"
SYNC_DOCS="false"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

usage() {
  cat <<'USAGE'
用法：
  bash edge/deploy/push.sh --host <ip-or-host> --user <ssh-user> [options]

示例：
  bash edge/deploy/push.sh --host 192.168.1.120 --user pc
  bash edge/deploy/push.sh --host 192.168.1.120 --user pc --dry-run
  bash edge/deploy/push.sh --host 192.168.1.120 --user pc --delete
  bash edge/deploy/push.sh --host 192.168.1.120 --user pc --port 2222

参数：
  --host <host>         目标 3576 IP 或主机名，必填
  --user <user>         目标 SSH 用户名，必填
  --port <port>         目标 SSH 端口，默认 22
  --target-dir <path>   目标目录，默认 /opt/visionops_v3
  --dry-run             仅预览 rsync 结果，不真正写入远端
  --delete              删除远端目录中本地已不存在的同步文件
  --sync-docs           同步 docs/ 中的交接与架构文档
  --sync-models         预留参数；当前仅提示，暂不真正同步模型
  -h, --help            显示帮助

当前默认同步内容：
  apps/collector_web
  edge/
  interfaces/
  configs/
  deploy/
  tools/
  README.md
  AGENTS.md
  CMakeLists.txt
  .gitignore

当前默认不同步内容：
  .git/
  build/
  training/
  tests/
  apps/server_api/
  models/
  __pycache__/
  *.pyc
  .pytest_cache/
  .mypy_cache/
USAGE
}

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log_error "缺少命令: $1"
    exit 1
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) TARGET_HOST="${2:-}"; shift 2 ;;
    --user) TARGET_USER="${2:-}"; shift 2 ;;
    --port) TARGET_PORT="${2:-22}"; shift 2 ;;
    --target-dir) TARGET_DIR="${2:-/opt/visionops_v3}"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --delete) DELETE_REMOTE="true"; shift ;;
    --sync-models) SYNC_MODELS="true"; shift ;;
    --sync-docs) SYNC_DOCS="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      log_error "不支持的参数: $1"
      usage
      exit 1
      ;;
  esac
done

[[ -n "${TARGET_HOST}" ]] || { log_error "--host 必填"; usage; exit 1; }
[[ -n "${TARGET_USER}" ]] || { log_error "--user 必填"; usage; exit 1; }

require_cmd ssh
require_cmd rsync

cd "${REPO_ROOT}"

log_info "仓库根目录: ${REPO_ROOT}"
log_info "目标设备: ${TARGET_USER}@${TARGET_HOST}:${TARGET_PORT}"
log_info "目标目录: ${TARGET_DIR}"

if [[ "${SYNC_MODELS}" == "true" ]]; then
  log_warn "--sync-models 当前仅预留接口，暂未实现模型包同步。"
  log_warn "后续建议单独同步 /opt/visionops_v3/models 下的标准模型包目录。"
fi

SSH_CMD=(ssh ${SSH_OPTS} -p "${TARGET_PORT}")
RSYNC_RSH="ssh ${SSH_OPTS} -p ${TARGET_PORT}"

log_info "检查远端连接..."
"${SSH_CMD[@]}" "${TARGET_USER}@${TARGET_HOST}" "echo ok" >/dev/null
log_ok "SSH 连接成功"

log_info "创建远端目标目录..."
"${SSH_CMD[@]}" "${TARGET_USER}@${TARGET_HOST}" "mkdir -p '${TARGET_DIR}'"

RSYNC_ARGS=(
  -az
  --info=stats2,progress2
  --human-readable
  --rsh "${RSYNC_RSH}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

if [[ "${DELETE_REMOTE}" == "true" ]]; then
  RSYNC_ARGS+=(--delete)
fi

RSYNC_ARGS+=(
  --exclude ".git/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "*.pyo"
  --exclude "*.swp"
  --exclude "build/"
  --exclude "build-rknn/"
  --exclude "cmake-build-*/"
  --exclude "training/"
  --exclude "tests/"
  --exclude "apps/server_api/"
  --exclude "models/"
  --exclude "*.pt"
  --exclude "*.onnx"
  --exclude "*.rknn"
  --exclude "*.tar"
  --exclude "*.tar.gz"
  --exclude "*.zip"
)

SYNC_ITEMS=(
  "apps/collector_web"
  "edge"
  "interfaces"
  "configs"
  "deploy"
  "tools"
  "README.md"
  "AGENTS.md"
  "CMakeLists.txt"
  ".gitignore"
)

if [[ "${SYNC_DOCS}" == "true" ]]; then
  SYNC_ITEMS+=("docs/architecture" "docs/handoff" "docs/migration")
fi

log_info "本次同步内容："
for item in "${SYNC_ITEMS[@]}"; do
  echo "  - ${item}"
done

log_info "开始同步到远端..."
for item in "${SYNC_ITEMS[@]}"; do
  rsync "${RSYNC_ARGS[@]}" --relative "./${item}" "${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}/"
done

log_ok "代码同步完成"

if "${SSH_CMD[@]}" "${TARGET_USER}@${TARGET_HOST}" "[ -d '${TARGET_DIR}/collector_web' ]"; then
  log_warn "检测到旧版脚本残留目录: ${TARGET_DIR}/collector_web"
  log_warn "该目录是早期同步脚本把 apps/collector_web 拍扁后的错误路径。"
  log_warn "当前运行中的 Collector 使用的是 ${TARGET_DIR}/apps/collector_web。"
  log_warn "确认无用后，可在板端手动清理：rm -rf '${TARGET_DIR}/collector_web'"
fi

cat <<EOF

后续可在 3576 上执行：

  cd ${TARGET_DIR}
  cmake -S . -B build-rknn \\
    -DVISIONOPS_ENABLE_RKNN=ON \\
    -DVISIONOPS_ENABLE_OPENCV=ON \\
    -DVISIONOPS_RKNN_INCLUDE_DIR=/path/to/rknn/include \\
    -DVISIONOPS_RKNN_LIBRARY=/path/to/librknnrt.so
  cmake --build build-rknn -j4

如果只改了 Python / 前端页面，也可以直接重启对应服务或重新启动 Collector Web。
EOF
