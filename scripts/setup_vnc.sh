#!/usr/bin/env bash
# ============================================================
# RK3576 Headless Remote Desktop
# Xfce + TigerVNC wrapper + systemd
# Version: 3.4.2
#
# 设计目标：
#   1) 面向无显示器 RK3576，创建独立 Xfce 虚拟桌面
#   2) 默认与已有 LightDM/Xorg/x11vnc/VNC 共存，不破坏原桌面
#   3) 自动寻找同时空闲的 X DISPLAY 和 VNC TCP 端口
#   4) 使用 TigerVNC vncserver wrapper + systemd
#   5) 可重复执行；记录上次选择的 DISPLAY，优先复用
#   6) 自动识别旧的远程桌面/X Server，并输出诊断信息
#
# 默认：
#   bash setup_vnc_v3.4.2.sh
#
# 自动选择 DISPLAY（默认在 :1 ~ :20 中寻找）：
#   VNC_DISPLAY=auto bash setup_vnc_v3.4.2.sh
#
# 强制指定 DISPLAY（若已被占用则安全退出，不抢占）：
#   VNC_DISPLAY=3 bash setup_vnc_v3.4.2.sh
#
# 修改分辨率：
#   VNC_GEOMETRY=1600x900 bash setup_vnc_v3.4.2.sh
#
# 使用清华 ARM Ubuntu 镜像：
#   USE_TUNA_MIRROR=1 bash setup_vnc_v3.4.2.sh
#
# 注意：
#   - 默认不会停止 LightDM、Xorg、x11vnc、xrdp 等已有方案。
#   - 不会直接删除正在使用的 /tmp/.X*-lock 或 X11 socket。
# ============================================================

set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}$*${NC}"; }
info() { echo -e "${BLUE}$*${NC}"; }
warn() { echo -e "${YELLOW}$*${NC}"; }
err()  { echo -e "${RED}$*${NC}" >&2; }

die() {
    err "ERROR: $*"
    exit 1
}

# -------------------- 可调参数 --------------------
VNC_DISPLAY_REQUEST="${VNC_DISPLAY:-auto}"
VNC_GEOMETRY="${VNC_GEOMETRY:-1920x1080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
AUTO_DISPLAY_MIN="${AUTO_DISPLAY_MIN:-1}"
AUTO_DISPLAY_MAX="${AUTO_DISPLAY_MAX:-20}"
USE_TUNA_MIRROR="${USE_TUNA_MIRROR:-0}"
STATE_FILE="${STATE_FILE:-/etc/default/rk3576-tigervnc-headless}"

[[ "$VNC_DISPLAY_REQUEST" == "auto" || "$VNC_DISPLAY_REQUEST" =~ ^[0-9]+$ ]] \
    || die "VNC_DISPLAY 必须为 auto 或非负整数"
[[ "$VNC_GEOMETRY" =~ ^[0-9]+x[0-9]+$ ]] \
    || die "VNC_GEOMETRY 格式应类似 1920x1080"
[[ "$VNC_DEPTH" =~ ^(16|24|32)$ ]] \
    || die "VNC_DEPTH 建议使用 16/24/32"
[[ "$AUTO_DISPLAY_MIN" =~ ^[0-9]+$ && "$AUTO_DISPLAY_MAX" =~ ^[0-9]+$ ]] \
    || die "AUTO_DISPLAY_MIN/MAX 必须为整数"
(( AUTO_DISPLAY_MIN <= AUTO_DISPLAY_MAX )) \
    || die "AUTO_DISPLAY_MIN 不能大于 AUTO_DISPLAY_MAX"

# 如果脚本被 sudo 调用，桌面用户仍使用原始登录用户。
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$(id -un)"
fi

[[ "$TARGET_USER" != "root" ]] \
    || die "请使用普通用户执行该脚本，不要直接以 root 登录后部署 VNC。"

TARGET_GROUP="$(id -gn "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[[ -n "$TARGET_HOME" && -d "$TARGET_HOME" ]] \
    || die "无法确定用户 ${TARGET_USER} 的 HOME"

VNC_DISPLAY=""
VNC_PORT=""
SERVICE_NAME=""

on_error() {
    local rc=$?
    local line=${BASH_LINENO[0]:-unknown}
    err "脚本在第 ${line} 行失败，退出码: ${rc}"
    if [[ -n "${SERVICE_NAME:-}" ]]; then
        err "可运行以下命令继续定位："
        err "  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
        err "  sudo journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
    fi
    exit "$rc"
}
trap on_error ERR

run_as_user() {
    if [[ "$(id -un)" == "$TARGET_USER" ]]; then
        HOME="$TARGET_HOME" "$@"
    else
        sudo -u "$TARGET_USER" -H "$@"
    fi
}

port_in_use() {
    local port="$1"
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"
}

display_has_lock_or_socket() {
    local d="$1"
    [[ -e "/tmp/.X${d}-lock" || -e "/tmp/.X11-unix/X${d}" ]]
}

display_has_server_process() {
    local d="$1"
    pgrep -a -f "(Xorg|Xwayland|Xtigervnc|Xvnc)([^[:digit:]]|.*[[:space:]]):${d}([[:space:]]|$)" \
        >/dev/null 2>&1
}

display_in_use() {
    local d="$1"
    display_has_lock_or_socket "$d" || display_has_server_process "$d"
}

our_vnc_running_on_display() {
    local d="$1"
    sudo systemctl is-active --quiet "vncserver@${d}.service" 2>/dev/null \
        && pgrep -u "$TARGET_USER" -f "Xtigervnc.*:${d}([[:space:]]|$)" >/dev/null 2>&1
}

display_reason() {
    local d="$1"
    local p=$((5900 + d))
    local reasons=()

    [[ -e "/tmp/.X${d}-lock" ]] && reasons+=("/tmp/.X${d}-lock")
    [[ -e "/tmp/.X11-unix/X${d}" ]] && reasons+=("/tmp/.X11-unix/X${d}")
    display_has_server_process "$d" && reasons+=("X server process")
    port_in_use "$p" && reasons+=("TCP ${p}")

    if ((${#reasons[@]} == 0)); then
        echo "空闲"
    else
        local IFS=', '
        echo "${reasons[*]}"
    fi
}

read_saved_display() {
    [[ -r "$STATE_FILE" ]] || return 1
    local d
    d="$(awk -F= '$1=="VNC_DISPLAY"{gsub(/"/,"",$2); print $2}' "$STATE_FILE" 2>/dev/null | tail -n1)"
    [[ "$d" =~ ^[0-9]+$ ]] || return 1
    echo "$d"
}

choose_display() {
    local requested="$VNC_DISPLAY_REQUEST"
    local saved=""

    if [[ "$requested" != "auto" ]]; then
        VNC_DISPLAY="$requested"
        VNC_PORT=$((5900 + VNC_DISPLAY))

        # 若这是脚本自己当前正常运行的实例，可以复用。
        if our_vnc_running_on_display "$VNC_DISPLAY"; then
            return 0
        fi

        if display_in_use "$VNC_DISPLAY" || port_in_use "$VNC_PORT"; then
            die "指定的 DISPLAY :${VNC_DISPLAY} 不可用：$(display_reason "$VNC_DISPLAY")。请改用 VNC_DISPLAY=auto 或指定其它显示号。"
        fi
        return 0
    fi

    saved="$(read_saved_display || true)"
    if [[ -n "$saved" ]]; then
        local saved_port=$((5900 + saved))
        if our_vnc_running_on_display "$saved"; then
            VNC_DISPLAY="$saved"
            VNC_PORT="$saved_port"
            info "检测到本脚本上次运行的 TigerVNC :${saved}，本次优先复用。"
            return 0
        fi
        if ! display_in_use "$saved" && ! port_in_use "$saved_port"; then
            VNC_DISPLAY="$saved"
            VNC_PORT="$saved_port"
            info "上次保存的 DISPLAY :${saved} 当前空闲，本次继续使用。"
            return 0
        fi
        warn "上次保存的 DISPLAY :${saved} 当前被占用：$(display_reason "$saved")"
        warn "将自动寻找新的空闲 DISPLAY。"
    fi

    local d p
    for ((d=AUTO_DISPLAY_MIN; d<=AUTO_DISPLAY_MAX; d++)); do
        p=$((5900 + d))
        if ! display_in_use "$d" && ! port_in_use "$p"; then
            VNC_DISPLAY="$d"
            VNC_PORT="$p"
            return 0
        fi
    done

    die "在 :${AUTO_DISPLAY_MIN} ~ :${AUTO_DISPLAY_MAX} 中未找到同时空闲的 DISPLAY/端口。"
}

show_existing_graphics() {
    local found=0

    if systemctl is-active --quiet display-manager.service 2>/dev/null; then
        found=1
        warn "检测到 display-manager 正在运行："
        systemctl status display-manager.service --no-pager -l 2>/dev/null \
            | sed -n '1,12p' || true
        echo
    fi

    local xprocs
    xprocs="$(ps -ef | grep -E 'Xorg|Xwayland|Xtigervnc|Xvnc' | grep -v grep || true)"
    if [[ -n "$xprocs" ]]; then
        found=1
        warn "检测到已有 X Server："
        echo "$xprocs"
        echo
    fi

    local remote
    remote="$(ps -ef | grep -E 'x11vnc|Xtigervnc|Xvnc|tightvnc|wayvnc|xrdp|gnome-remote-desktop|vino-server' \
        | grep -v grep || true)"
    if [[ -n "$remote" ]]; then
        found=1
        warn "检测到已有远程桌面相关进程："
        echo "$remote"
        echo
    fi

    if (( found == 1 )); then
        info "v3.4.2 默认采用共存模式：不会停止上述服务，只避开它们占用的 DISPLAY/端口。"
    else
        info "未检测到明显的已有图形/远程桌面服务。"
    fi
}

xfce_on_selected_display() {
    local pid
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        if tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null \
            | grep -qx "DISPLAY=:${VNC_DISPLAY}"; then
            return 0
        fi
    done < <(pgrep -u "$TARGET_USER" -f 'xfce4-session|startxfce4' 2>/dev/null || true)
    return 1
}

stop_managed_vnc_instances() {
    local unit desc
    while read -r unit; do
        [[ -n "$unit" ]] || continue
        desc="$(systemctl show -p Description --value "$unit" 2>/dev/null || true)"
        if [[ "$desc" == *"TigerVNC Headless Xfce Server display"* ]]; then
            warn "停止旧的一键脚本实例：${unit}"
            sudo systemctl disable --now "$unit" >/dev/null 2>&1 || true
        fi
    done < <(
        systemctl list-units --all --type=service --no-legend 'vncserver@*.service' 2>/dev/null \
            | awk '{print $1}' || true
    )
}

clear
info "========================================"
info " RK3576 TigerVNC Headless Desktop v3.4.2"
info " Auto DISPLAY + coexistence mode"
info "========================================"
echo "用户       : $TARGET_USER"
echo "HOME       : $TARGET_HOME"
echo "显示号请求 : $VNC_DISPLAY_REQUEST"
echo "分辨率     : $VNC_GEOMETRY"
echo "颜色深度   : $VNC_DEPTH"
echo

# ============================================================
# 1/9 基础环境
# ============================================================
log "[1/9] 检查系统环境..."

command -v sudo >/dev/null 2>&1 || die "未安装 sudo"
command -v apt-get >/dev/null 2>&1 || die "当前脚本面向 Ubuntu/Debian apt 系统"
command -v ss >/dev/null 2>&1 || die "找不到 ss 命令（通常由 iproute2 提供）"

ARCH="$(uname -m)"
OS_ID="unknown"
OS_VERSION="unknown"
CODENAME=""
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_VERSION="${VERSION_ID:-unknown}"
    CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
fi

echo "架构       : $ARCH"
echo "系统       : $OS_ID $OS_VERSION ${CODENAME:+($CODENAME)}"

# ============================================================
# 2/9 探测已有图形/远程桌面
# ============================================================
log "[2/9] 探测已有 X Server / 远程桌面..."
show_existing_graphics

# ============================================================
# 3/9 选择 DISPLAY + 端口
# ============================================================
log "[3/9] 选择 TigerVNC DISPLAY..."
choose_display
SERVICE_NAME="vncserver@${VNC_DISPLAY}.service"

echo "选择显示号 : :${VNC_DISPLAY}"
echo "选择端口   : ${VNC_PORT}"

# 解释前几个 DISPLAY 为什么跳过，便于排障。
if [[ "$VNC_DISPLAY_REQUEST" == "auto" ]]; then
    echo
    info "DISPLAY 扫描摘要："
    for ((d=AUTO_DISPLAY_MIN; d<VNC_DISPLAY; d++)); do
        echo "  :${d} -> 跳过：$(display_reason "$d")"
    done
    echo "  :${VNC_DISPLAY} -> 使用：空闲"
fi

# ============================================================
# 4/9 可选镜像源 + 软件安装
# ============================================================
log "[4/9] 检查软件源并安装 TigerVNC / Xfce / D-Bus..."

if [[ "$USE_TUNA_MIRROR" == "1" ]]; then
    if [[ "$ARCH" != "aarch64" ]]; then
        warn "USE_TUNA_MIRROR=1，但当前不是 aarch64；跳过 ARM ubuntu-ports 镜像配置。"
    elif [[ "$OS_ID" != "ubuntu" || -z "$CODENAME" ]]; then
        warn "无法可靠识别 Ubuntu codename；为安全起见不修改 sources.list。"
    else
        warn "将备份并切换 /etc/apt/sources.list 到清华 ubuntu-ports 镜像。"
        sudo cp -a /etc/apt/sources.list \
            "/etc/apt/sources.list.vnc-v342.$(date +%Y%m%d_%H%M%S).bak"
        sudo tee /etc/apt/sources.list >/dev/null <<EOF_TUNA
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ ${CODENAME} main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ ${CODENAME}-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ ${CODENAME}-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ ${CODENAME}-security main restricted universe multiverse
EOF_TUNA
    fi
else
    info "保持当前系统软件源不变。"
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tigervnc-standalone-server \
    tigervnc-common \
    xfce4 \
    xfce4-goodies \
    dbus-x11 \
    x11-xserver-utils \
    net-tools

VNCSERVER_BIN="$(command -v vncserver || true)"
XTIGERVNC_BIN="$(command -v Xtigervnc || true)"
STARTXFCE_BIN="$(command -v startxfce4 || true)"
DBUS_LAUNCH_BIN="$(command -v dbus-launch || true)"

[[ -x "$VNCSERVER_BIN" ]] || die "找不到 vncserver wrapper"
[[ -x "$XTIGERVNC_BIN" ]] || die "找不到 Xtigervnc"
[[ -x "$STARTXFCE_BIN" ]] || die "找不到 startxfce4"
[[ -x "$DBUS_LAUNCH_BIN" ]] || die "找不到 dbus-launch"

echo "vncserver  : $VNCSERVER_BIN"
echo "Xtigervnc  : $XTIGERVNC_BIN"
echo "startxfce4 : $STARTXFCE_BIN"

# ============================================================
# 5/9 VNC 密码
# ============================================================
log "[5/9] 配置 VNC 密码..."

sudo install -d -m 700 -o "$TARGET_USER" -g "$TARGET_GROUP" "$TARGET_HOME/.vnc"

if [[ -s "$TARGET_HOME/.vnc/passwd" ]]; then
    warn "检测到现有 $TARGET_HOME/.vnc/passwd，保留原密码。"
else
    info "请设置 VNC 连接密码："
    run_as_user vncpasswd
fi

sudo chown "$TARGET_USER:$TARGET_GROUP" "$TARGET_HOME/.vnc/passwd"
sudo chmod 600 "$TARGET_HOME/.vnc/passwd"

# ============================================================
# 6/9 Xfce xstartup
# ============================================================
log "[6/9] 创建 Headless Xfce 启动脚本..."

if [[ -f "$TARGET_HOME/.vnc/xstartup" ]]; then
    sudo cp -a "$TARGET_HOME/.vnc/xstartup" \
        "$TARGET_HOME/.vnc/xstartup.vnc-v342.$(date +%Y%m%d_%H%M%S).bak"
fi

sudo tee "$TARGET_HOME/.vnc/xstartup" >/dev/null <<'EOF_XSTARTUP'
#!/bin/sh
# TigerVNC Headless Xfce session

unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=XFCE
export DESKTOP_SESSION=xfce

[ -r "$HOME/.Xresources" ] && xrdb "$HOME/.Xresources"

exec dbus-launch --exit-with-session startxfce4
EOF_XSTARTUP

sudo chown "$TARGET_USER:$TARGET_GROUP" "$TARGET_HOME/.vnc/xstartup"
sudo chmod 755 "$TARGET_HOME/.vnc/xstartup"

# ============================================================
# 7/9 systemd
# ============================================================
log "[7/9] 创建 systemd 服务..."

# 清掉旧脚本留下的失败/运行实例，但不碰 x11vnc、LightDM、Xorg 等其它方案。
stop_managed_vnc_instances

# 兼容 v3.3 的错误实例名。
for d in $(seq "$AUTO_DISPLAY_MIN" "$AUTO_DISPLAY_MAX"); do
    sudo systemctl disable --now "vncserver@:${d}.service" >/dev/null 2>&1 || true
done

if [[ -f /etc/systemd/system/vncserver@.service ]]; then
    sudo cp -a /etc/systemd/system/vncserver@.service \
        "/etc/systemd/system/vncserver@.service.vnc-v342.$(date +%Y%m%d_%H%M%S).bak"
fi

sudo tee /etc/systemd/system/vncserver@.service >/dev/null <<EOF_SERVICE
[Unit]
Description=TigerVNC Headless Xfce Server display :%i
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=${TARGET_USER}
Group=${TARGET_GROUP}
WorkingDirectory=${TARGET_HOME}
Environment=HOME=${TARGET_HOME}
Environment=USER=${TARGET_USER}

ExecStartPre=-${VNCSERVER_BIN} -kill :%i
ExecStartPre=-${VNCSERVER_BIN} -list :* -cleanstale
ExecStart=${VNCSERVER_BIN} :%i -fg -autokill -cleanstale -localhost no -geometry ${VNC_GEOMETRY} -depth ${VNC_DEPTH} -SecurityTypes VncAuth -PasswordFile ${TARGET_HOME}/.vnc/passwd -xstartup ${TARGET_HOME}/.vnc/xstartup
ExecStop=${VNCSERVER_BIN} -kill :%i

Restart=on-failure
RestartSec=10
TimeoutStartSec=30
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF_SERVICE

# 持久化本次选择，下一次自动优先复用。
sudo tee "$STATE_FILE" >/dev/null <<EOF_STATE
# Generated by setup_vnc_v3.4.2.sh
VNC_DISPLAY=${VNC_DISPLAY}
VNC_PORT=${VNC_PORT}
VNC_GEOMETRY="${VNC_GEOMETRY}"
VNC_DEPTH=${VNC_DEPTH}
TARGET_USER="${TARGET_USER}"
EOF_STATE
sudo chmod 644 "$STATE_FILE"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

# ============================================================
# 8/9 启动验证
# ============================================================
log "[8/9] 启动并验证 VNC 服务..."

START_OK=1
if ! sudo systemctl restart "$SERVICE_NAME"; then
    START_OK=0
    err "systemd 启动 VNC 失败；继续执行诊断。"
fi

sleep 3

SERVICE_OK=0
PORT_OK=0
PROCESS_OK=0
XFCE_OK=0

sudo systemctl is-active --quiet "$SERVICE_NAME" && SERVICE_OK=1 || true
port_in_use "$VNC_PORT" && PORT_OK=1 || true
pgrep -u "$TARGET_USER" -f "Xtigervnc.*:${VNC_DISPLAY}([[:space:]]|$)" >/dev/null 2>&1 \
    && PROCESS_OK=1 || true
xfce_on_selected_display && XFCE_OK=1 || true

IP_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
IP_ADDR="${IP_ADDR:-<开发板IP>}"

# ============================================================
# 9/9 结果与常用命令
# ============================================================
log "[9/9] 部署结果"
info "========================================"

[[ "$SERVICE_OK" == "1" ]] \
    && log "✅ systemd 服务运行中 : $SERVICE_NAME" \
    || err "❌ systemd 服务未运行 : $SERVICE_NAME"

[[ "$PORT_OK" == "1" ]] \
    && log "✅ TCP ${VNC_PORT} 正在监听" \
    || err "❌ TCP ${VNC_PORT} 未监听"

[[ "$PROCESS_OK" == "1" ]] \
    && log "✅ Xtigervnc :${VNC_DISPLAY} 已运行" \
    || err "❌ 未检测到 Xtigervnc :${VNC_DISPLAY}"

[[ "$XFCE_OK" == "1" ]] \
    && log "✅ 已确认 Xfce 属于 DISPLAY=:${VNC_DISPLAY}" \
    || warn "⚠️ 暂未确认 DISPLAY=:${VNC_DISPLAY} 上的 Xfce 会话"

info "========================================"
echo "VNC Viewer 连接地址："
echo "  ${IP_ADDR}:${VNC_PORT}"
echo
echo "显示号形式（部分客户端支持）："
echo "  ${IP_ADDR}:${VNC_DISPLAY}"
echo
echo "配置记录："
echo "  ${STATE_FILE}"

info "========================================"
info "常用管理命令"
info "========================================"
cat <<EOF_COMMANDS

# 查看当前自动选择结果
cat ${STATE_FILE}

# 服务状态
sudo systemctl status ${SERVICE_NAME} --no-pager -l

# 最近日志
sudo journalctl -u ${SERVICE_NAME} -n 100 --no-pager

# 实时日志
sudo journalctl -u ${SERVICE_NAME} -f

# TigerVNC / Xfce 日志
tail -n 100 ~/.vnc/*.log

# 重启
sudo systemctl restart ${SERVICE_NAME}

# 停止
sudo systemctl stop ${SERVICE_NAME}

# 启动
sudo systemctl start ${SERVICE_NAME}

# 检查端口
ss -ltnp | grep ${VNC_PORT}

# 查看图形/远程桌面进程
ps -ef | grep -E 'Xorg|Xwayland|Xtigervnc|Xvnc|x11vnc|xfce4-session|startxfce4|lightdm' | grep -v grep

# 查看 DISPLAY :${VNC_DISPLAY} 的锁/socket
ls -l /tmp/.X${VNC_DISPLAY}-lock /tmp/.X11-unix/X${VNC_DISPLAY} 2>/dev/null

# 修改 VNC 密码
vncpasswd
sudo systemctl restart ${SERVICE_NAME}

# 手动前台启动（排障）
sudo systemctl stop ${SERVICE_NAME}
vncserver :${VNC_DISPLAY} -fg -autokill -cleanstale -verbose -localhost no -geometry ${VNC_GEOMETRY} -depth ${VNC_DEPTH} -SecurityTypes VncAuth -PasswordFile ~/.vnc/passwd -xstartup ~/.vnc/xstartup

# 检查防火墙
sudo ufw status
# sudo ufw allow ${VNC_PORT}/tcp

# 如果明确决定以后不再使用旧 x11vnc，请先找出它由哪个服务启动：
systemctl list-unit-files | grep -i x11vnc
systemctl list-units --all | grep -i x11vnc
# 确认后再手动 stop/disable；本脚本默认不会自动删除旧方案。

EOF_COMMANDS

if [[ "$SERVICE_OK" == "1" && "$PORT_OK" == "1" && "$PROCESS_OK" == "1" ]]; then
    info "========================================"
    log "✅ TigerVNC v3.4.2 部署完成"
    log "   连接：${IP_ADDR}:${VNC_PORT}"
    info "========================================"
    exit 0
fi

info "========================================"
err "部署尚未完全通过验证，自动输出诊断信息："
info "========================================"

sudo systemctl status "$SERVICE_NAME" --no-pager -l || true
echo
sudo journalctl -u "$SERVICE_NAME" -n 100 --no-pager || true
echo

warn "用户 VNC 日志："
run_as_user bash -lc 'ls -la ~/.vnc/; echo; tail -n 150 ~/.vnc/*.log 2>/dev/null || true' || true
echo

warn "当前 DISPLAY/端口状态："
echo "  :${VNC_DISPLAY} -> $(display_reason "$VNC_DISPLAY")"
ls -l "/tmp/.X${VNC_DISPLAY}-lock" "/tmp/.X11-unix/X${VNC_DISPLAY}" 2>/dev/null || true
echo

warn "相关进程："
ps -ef | grep -E 'Xorg|Xwayland|Xtigervnc|Xvnc|x11vnc|xfce4-session|startxfce4|lightdm' \
    | grep -v grep || true
echo

warn "可直接执行以下前台诊断命令（Ctrl+C 结束）："
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo "  vncserver :${VNC_DISPLAY} -fg -autokill -cleanstale -verbose -localhost no -geometry ${VNC_GEOMETRY} -depth ${VNC_DEPTH} -SecurityTypes VncAuth -PasswordFile ~/.vnc/passwd -xstartup ~/.vnc/xstartup"

exit 1