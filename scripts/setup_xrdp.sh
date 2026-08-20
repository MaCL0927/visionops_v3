#!/usr/bin/env bash
# ============================================================
# RK3576 Headless XRDP + Xfce
# Backend: Xvnc (TigerVNC, localhost only)
# Version: 1.1
#
# 适用场景：
#   - Ubuntu 20.04 / ARM64 RK3576 厂商镜像
#   - 系统 Xorg/xserver-common 可能被厂商固定版本
#   - 不希望为了 XRDP 强制升级/替换系统 Xorg 栈
#
# 对外：
#   - SSH :22
#   - XRDP :3389
#
# 内部：
#   - xrdp-sesman 按需启动 Xvnc
#   - Xvnc 仅监听 localhost，不对局域网暴露 590x
#   - Xfce 作为桌面环境
#
# 默认行为：
#   - 清理旧 x11vnc / 独立 TigerVNC systemd 服务 / TightVNC 等
#   - 保留 tigervnc-standalone-server 软件包作为 XRDP 的 Xvnc 后端
#   - 禁用 LightDM/GDM/SDDM，并切到 multi-user.target
#   - 不安装 xorgxrdp
#   - 不主动升级 xserver-xorg-core / xserver-common
#
# 使用：
#   chmod +x setup_xrdp_rk3576_v1.1.sh
#   bash setup_xrdp_rk3576_v1.1.sh
#
# 可选：
#   XRDP_PORT=3390 bash setup_xrdp_rk3576_v1.1.sh
#   CLEAN_OLD_REMOTE=0 bash setup_xrdp_rk3576_v1.1.sh
#   DISABLE_DISPLAY_MANAGER=0 bash setup_xrdp_rk3576_v1.1.sh
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
die()  { err "ERROR: $*"; exit 1; }

XRDP_PORT="${XRDP_PORT:-3389}"
CLEAN_OLD_REMOTE="${CLEAN_OLD_REMOTE:-1}"
DISABLE_DISPLAY_MANAGER="${DISABLE_DISPLAY_MANAGER:-1}"
OPEN_UFW="${OPEN_UFW:-1}"

[[ "$XRDP_PORT" =~ ^[0-9]+$ ]] || die "XRDP_PORT 必须是整数"
(( XRDP_PORT >= 1024 && XRDP_PORT <= 65535 )) || die "XRDP_PORT 必须在 1024~65535"
[[ "$CLEAN_OLD_REMOTE" =~ ^[01]$ ]] || die "CLEAN_OLD_REMOTE 只能为 0/1"
[[ "$DISABLE_DISPLAY_MANAGER" =~ ^[01]$ ]] || die "DISABLE_DISPLAY_MANAGER 只能为 0/1"
[[ "$OPEN_UFW" =~ ^[01]$ ]] || die "OPEN_UFW 只能为 0/1"

if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$(id -un)"
fi

[[ "$TARGET_USER" != "root" ]] || die "请使用普通用户执行脚本。"

TARGET_GROUP="$(id -gn "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[[ -n "$TARGET_HOME" && -d "$TARGET_HOME" ]] || die "无法确定 ${TARGET_USER} 的 HOME"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="/root/xrdp-rk3576-v11-backup-${STAMP}"

on_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    err "脚本在第 ${line} 行失败，退出码 ${rc}"
    err "排障命令："
    err "  sudo systemctl status xrdp --no-pager -l"
    err "  sudo systemctl status xrdp-sesman --no-pager -l"
    err "  sudo journalctl -u xrdp -u xrdp-sesman -n 150 --no-pager"
    err "  sudo tail -n 150 /var/log/xrdp.log /var/log/xrdp-sesman.log"
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

pkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

port_lines() {
    ss -ltnp 2>/dev/null | awk -v p=":${1}" '$4 ~ p"$" {print}'
}

port_in_use() {
    [[ -n "$(port_lines "$1")" ]]
}

disable_unit_if_exists() {
    local svc="$1"
    if systemctl list-unit-files "$svc" --no-legend 2>/dev/null | grep -q .; then
        sudo systemctl disable --now "$svc" >/dev/null 2>&1 || true
        sudo systemctl mask "$svc" >/dev/null 2>&1 || true
    fi
}

stop_remove_old_vnc_units() {
    local unit path
    while read -r unit; do
        [[ -n "$unit" ]] || continue
        warn "停止旧远程桌面服务：$unit"
        sudo systemctl disable --now "$unit" >/dev/null 2>&1 || true
    done < <(
        systemctl list-units --all --type=service --no-legend 2>/dev/null \
          | awk '{print $1}' \
          | grep -Ei 'x11vnc|vncserver@|tightvnc|wayvnc|vino' || true
    )

    sudo mkdir -p "$BACKUP_DIR/systemd"
    shopt -s nullglob
    for path in \
        /etc/systemd/system/x11vnc*.service \
        /etc/systemd/system/vncserver@.service \
        /etc/systemd/system/tigervnc*.service \
        /etc/systemd/system/tightvnc*.service \
        /etc/systemd/system/wayvnc*.service
    do
        [[ -e "$path" || -L "$path" ]] || continue
        warn "备份并移除旧自定义 unit：$path"
        sudo cp -a "$path" "$BACKUP_DIR/systemd/" 2>/dev/null || true
        sudo rm -f "$path"
    done
    shopt -u nullglob

    sudo systemctl daemon-reload
    sudo systemctl reset-failed || true
}

clean_old_remote_packages() {
    # 注意：TigerVNC standalone/common 不能卸载，因为本方案需要 Xvnc。
    local candidates=(
        x11vnc
        tightvncserver
        tightvnc-common
        wayvnc
        vino
        xorgxrdp
    )
    local installed=()
    local p
    for p in "${candidates[@]}"; do
        pkg_installed "$p" && installed+=("$p")
    done

    if ((${#installed[@]})); then
        warn "卸载不再使用的软件包：${installed[*]}"
        sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y "${installed[@]}"
    fi
}

clean_user_vnc_autostart() {
    local dir="$TARGET_HOME/.config/autostart"
    [[ -d "$dir" ]] || return 0

    sudo mkdir -p "$BACKUP_DIR/user-autostart"
    local f
    while IFS= read -r -d '' f; do
        if grep -Eqi 'x11vnc|tigervncserver|vncserver|tightvnc|wayvnc|vino-server' "$f"; then
            warn "备份并移除用户旧 VNC 自启动：$f"
            sudo cp -a "$f" "$BACKUP_DIR/user-autostart/" || true
            sudo rm -f "$f"
        fi
    done < <(find "$dir" -maxdepth 1 -type f -name '*.desktop' -print0 2>/dev/null)
}

clean_stale_x_locks() {
    local lock d pid sock
    shopt -s nullglob
    for lock in /tmp/.X*-lock; do
        [[ "$lock" =~ /tmp/\.X([0-9]+)-lock$ ]] || continue
        d="${BASH_REMATCH[1]}"
        pid="$(tr -dc '0-9' < "$lock" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            info "保留活动 X lock：$lock (PID=$pid)"
            continue
        fi
        sock="/tmp/.X11-unix/X${d}"
        warn "清理 stale X lock/socket：$lock $sock"
        sudo rm -f "$lock" "$sock"
    done
    shopt -u nullglob
}

set_ini_key_in_section() {
    local file="$1"
    local section="$2"
    local key="$3"
    local value="$4"

    sudo python3 - "$file" "$section" "$key" "$value" <<'PY'
import sys
p, section, key, value = sys.argv[1:]
lines = open(p, encoding="utf-8").read().splitlines()
target = f"[{section}]".lower()
in_section = False
done = False
out = []

for line in lines:
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        if in_section and not done:
            out.append(f"{key}={value}")
            done = True
        in_section = (s.lower() == target)

    if in_section and not done:
        # 允许替换注释或非注释形式
        stripped = s.lstrip(";#").strip()
        if stripped.lower().startswith(key.lower() + "="):
            out.append(f"{key}={value}")
            done = True
            continue

    out.append(line)

if in_section and not done:
    out.append(f"{key}={value}")
    done = True

if not done:
    raise SystemExit(f"section [{section}] not found in {p}")

open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
}

ensure_xvnc_sections() {
    grep -q '^\[Xvnc\]' /etc/xrdp/xrdp.ini \
        || die "/etc/xrdp/xrdp.ini 缺少 [Xvnc]"
    grep -q '^\[Xvnc\]' /etc/xrdp/sesman.ini \
        || die "/etc/xrdp/sesman.ini 缺少 [Xvnc]"

    # sesman 必须能找到 Xvnc
    command -v Xvnc >/dev/null 2>&1 || die "找不到 Xvnc"
}

write_xfce_startwm() {
    local wm="/etc/xrdp/startwm.sh"
    [[ -f "$wm" ]] || die "找不到 $wm"

    sudo cp -a "$wm" "${wm}.pre-rk3576-v11-${STAMP}.bak"

    sudo tee "$wm" >/dev/null <<'EOF'
#!/bin/sh
# RK3576 XRDP -> Xvnc -> Xfce

if [ -r /etc/default/locale ]; then
    . /etc/default/locale
    export LANG LANGUAGE
fi

unset DBUS_SESSION_BUS_ADDRESS
unset SESSION_MANAGER

export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=XFCE
export XDG_SESSION_DESKTOP=xfce
export DESKTOP_SESSION=xfce

exec dbus-launch --exit-with-session /usr/bin/startxfce4
EOF
    sudo chmod 755 "$wm"

    if [[ -f "$TARGET_HOME/.xsession" ]]; then
        sudo cp -a "$TARGET_HOME/.xsession" \
            "$TARGET_HOME/.xsession.pre-xrdp-v11-${STAMP}.bak"
    fi

    sudo tee "$TARGET_HOME/.xsession" >/dev/null <<'EOF'
#!/bin/sh
unset DBUS_SESSION_BUS_ADDRESS
unset SESSION_MANAGER
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=XFCE
export XDG_SESSION_DESKTOP=xfce
export DESKTOP_SESSION=xfce
exec dbus-launch --exit-with-session /usr/bin/startxfce4
EOF
    sudo chown "$TARGET_USER:$TARGET_GROUP" "$TARGET_HOME/.xsession"
    sudo chmod 755 "$TARGET_HOME/.xsession"

    # 避免旧的 XFCE session cache 恢复错误 DISPLAY/布局。
    if [[ -d "$TARGET_HOME/.cache/sessions" ]]; then
        sudo mkdir -p "$BACKUP_DIR"
        sudo cp -a "$TARGET_HOME/.cache/sessions" \
            "$BACKUP_DIR/xfce-sessions" 2>/dev/null || true
        sudo rm -rf "$TARGET_HOME/.cache/sessions"
    fi
    sudo install -d -m 700 -o "$TARGET_USER" -g "$TARGET_GROUP" \
        "$TARGET_HOME/.cache/sessions"
}

verify_public_listener() {
    local lines
    lines="$(port_lines "$XRDP_PORT")"
    [[ -n "$lines" ]] || return 1
    echo "$lines" | awk '{print $4}' \
        | grep -Eq '(^\*:'"$XRDP_PORT"'$|^0\.0\.0\.0:'"$XRDP_PORT"'$|^\[::\]:'"$XRDP_PORT"'$|^:::'"$XRDP_PORT"'$)'
}

clear
info "========================================"
info " RK3576 Headless XRDP + Xfce v1.1"
info " Backend: Xvnc / TigerVNC localhost-only"
info "========================================"
echo "用户                 : $TARGET_USER"
echo "HOME                 : $TARGET_HOME"
echo "XRDP 端口            : $XRDP_PORT"
echo "清理旧远程桌面       : $CLEAN_OLD_REMOTE"
echo "禁用本地显示管理器   : $DISABLE_DISPLAY_MANAGER"
echo

# ============================================================
# 1/10 系统 / Xorg 版本诊断
# ============================================================
log "[1/10] 检查系统与 Xorg 包状态..."

ARCH="$(uname -m)"
echo "架构: $ARCH"

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    echo "系统: ${ID:-unknown} ${VERSION_ID:-unknown} ${VERSION_CODENAME:-}"
fi

echo
info "当前 Xorg 包版本（仅诊断，不主动升级）："
dpkg-query -W -f='${Package}\t${Version}\n' \
    xserver-xorg-core xserver-common 2>/dev/null || true

echo
info "APT candidate："
apt-cache policy xserver-xorg-core xserver-common 2>/dev/null | sed -n '1,30p' || true

echo
info "APT hold："
apt-mark showhold 2>/dev/null || true
echo

# ============================================================
# 2/10 盘点旧服务
# ============================================================
log "[2/10] 盘点旧图形/远程桌面..."

ps -ef | grep -E \
'Xorg|Xwayland|x11vnc|Xtigervnc|Xvnc|tightvnc|wayvnc|vino|xrdp|lightdm|gdm|sddm' \
| grep -v grep || true
echo
ss -ltnp 2>/dev/null | grep -E ':(3389|590[0-9])([[:space:]]|$)' || true
echo

# ============================================================
# 3/10 清理旧“对外 VNC”服务
# ============================================================
log "[3/10] 清理旧远程服务..."

sudo systemctl stop xrdp.service xrdp-sesman.service >/dev/null 2>&1 || true

if [[ "$CLEAN_OLD_REMOTE" == "1" ]]; then
    sudo mkdir -p "$BACKUP_DIR"
    stop_remove_old_vnc_units
    clean_user_vnc_autostart
    clean_old_remote_packages
else
    warn "CLEAN_OLD_REMOTE=0：保留旧远程服务。"
fi

# ============================================================
# 4/10 Headless
# ============================================================
log "[4/10] 配置 Headless 启动模式..."

if [[ "$DISABLE_DISPLAY_MANAGER" == "1" ]]; then
    for dm in lightdm.service gdm.service gdm3.service sddm.service; do
        if systemctl list-unit-files "$dm" --no-legend 2>/dev/null | grep -q .; then
            warn "禁用并 mask：$dm"
            sudo systemctl disable --now "$dm" >/dev/null 2>&1 || true
            sudo systemctl mask "$dm" >/dev/null 2>&1 || true
        fi
    done

    sudo systemctl set-default multi-user.target
    clean_stale_x_locks
else
    warn "保留 display-manager。"
fi

# ============================================================
# 5/10 安装，不碰 xorgxrdp / xserver-xorg-core
# ============================================================
log "[5/10] 安装 XRDP + Xvnc backend + Xfce..."

sudo apt-get update

# --no-install-recommends 很重要：
# 避免 xrdp 的 Recommends 把 xorgxrdp / 新 Xorg 栈重新拉进来。
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    xrdp \
    tigervnc-standalone-server \
    tigervnc-common \
    xfce4 \
    xfce4-goodies \
    dbus-x11 \
    x11-xserver-utils \
    ssl-cert \
    net-tools

command -v xrdp >/dev/null || die "未找到 xrdp"
command -v xrdp-sesman >/dev/null || die "未找到 xrdp-sesman"
command -v Xvnc >/dev/null || die "未找到 Xvnc"
command -v startxfce4 >/dev/null || die "未找到 startxfce4"
command -v dbus-launch >/dev/null || die "未找到 dbus-launch"

echo "xrdp      : $(command -v xrdp)"
echo "sesman    : $(command -v xrdp-sesman)"
echo "Xvnc      : $(command -v Xvnc)"
echo "startxfce4: $(command -v startxfce4)"

# ============================================================
# 6/10 配置 XRDP -> Xvnc
# ============================================================
log "[6/10] 配置 XRDP 使用 Xvnc..."

sudo systemctl stop xrdp.service xrdp-sesman.service >/dev/null 2>&1 || true

[[ -f /etc/xrdp/xrdp.ini ]] || die "缺少 /etc/xrdp/xrdp.ini"
[[ -f /etc/xrdp/sesman.ini ]] || die "缺少 /etc/xrdp/sesman.ini"

sudo cp -a /etc/xrdp/xrdp.ini \
    "/etc/xrdp/xrdp.ini.pre-rk3576-v11-${STAMP}.bak"
sudo cp -a /etc/xrdp/sesman.ini \
    "/etc/xrdp/sesman.ini.pre-rk3576-v11-${STAMP}.bak"

ensure_xvnc_sections

# XRDP 对外端口
set_ini_key_in_section /etc/xrdp/xrdp.ini Globals port "$XRDP_PORT"

# 让登录默认选 Xvnc。
set_ini_key_in_section /etc/xrdp/xrdp.ini Globals autorun "Xvnc"

# 确保 Xvnc 是 sesman 管理的本机回环会话。
set_ini_key_in_section /etc/xrdp/xrdp.ini Xvnc ip "127.0.0.1"
set_ini_key_in_section /etc/xrdp/xrdp.ini Xvnc port "-1"

# sesman 的 Xvnc 启动参数由发行版默认配置提供。
# 强制校验关键参数存在，而不是改厂商 Xorg。
grep -A12 '^\[Xvnc\]' /etc/xrdp/sesman.ini | grep -Eq '^param=(Xvnc|/usr/bin/Xvnc)$' \
    || warn "sesman.ini [Xvnc] 的首个 param 不是常见 Xvnc；请留意后续验证日志。"

write_xfce_startwm

if id xrdp >/dev/null 2>&1; then
    sudo usermod -a -G ssl-cert xrdp
fi

# Linux 密码用于 XRDP 登录
PASS_STATE="$(passwd -S "$TARGET_USER" 2>/dev/null | awk '{print $2}' || true)"
if [[ "$PASS_STATE" != "P" ]]; then
    warn "用户 ${TARGET_USER} 当前密码状态为 '${PASS_STATE:-未知}'。"
    warn "XRDP 使用 Linux 密码；请执行 sudo passwd ${TARGET_USER}"
fi

# ============================================================
# 7/10 防火墙
# ============================================================
log "[7/10] 配置防火墙..."

if command -v ufw >/dev/null 2>&1; then
    UFW_STATE="$(sudo ufw status 2>/dev/null | head -n1 || true)"
    echo "$UFW_STATE"
    if [[ "$OPEN_UFW" == "1" && "$UFW_STATE" == *active* ]]; then
        sudo ufw allow "${XRDP_PORT}/tcp"
    fi
fi

# ============================================================
# 8/10 启动
# ============================================================
log "[8/10] 启动 XRDP..."

if port_in_use "$XRDP_PORT"; then
    err "启动前 ${XRDP_PORT} 已被占用："
    port_lines "$XRDP_PORT"
    die "请清理占用进程或改 XRDP_PORT"
fi

sudo systemctl daemon-reload
sudo systemctl unmask xrdp.service >/dev/null 2>&1 || true
sudo systemctl enable xrdp.service
sudo systemctl restart xrdp.service

if systemctl list-unit-files xrdp-sesman.service --no-legend 2>/dev/null | grep -q .; then
    sudo systemctl unmask xrdp-sesman.service >/dev/null 2>&1 || true
    sudo systemctl enable xrdp-sesman.service >/dev/null 2>&1 || true
    sudo systemctl restart xrdp-sesman.service
fi

sleep 3

# ============================================================
# 9/10 验证
# ============================================================
log "[9/10] 验证 XRDP..."

XRDP_OK=0
SESMAN_OK=0
PORT_OK=0
PUBLIC_OK=0
XVNC_OK=0
CONFIG_OK=0

sudo systemctl is-active --quiet xrdp.service && XRDP_OK=1 || true

if sudo systemctl is-active --quiet xrdp-sesman.service 2>/dev/null \
   || pgrep -x xrdp-sesman >/dev/null 2>&1; then
    SESMAN_OK=1
fi

port_in_use "$XRDP_PORT" && PORT_OK=1 || true
verify_public_listener && PUBLIC_OK=1 || true
command -v Xvnc >/dev/null && XVNC_OK=1 || true

if grep -q '^autorun=Xvnc$' /etc/xrdp/xrdp.ini \
   && grep -q '^\[Xvnc\]' /etc/xrdp/xrdp.ini \
   && grep -q '^\[Xvnc\]' /etc/xrdp/sesman.ini; then
    CONFIG_OK=1
fi

IP_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
IP_ADDR="${IP_ADDR:-<RK3576-IP>}"

info "========================================"
[[ "$XRDP_OK" == "1" ]] && log "✅ xrdp active" || err "❌ xrdp 未运行"
[[ "$SESMAN_OK" == "1" ]] && log "✅ xrdp-sesman active" || err "❌ sesman 未运行"
[[ "$PORT_OK" == "1" ]] && log "✅ TCP ${XRDP_PORT} LISTEN" || err "❌ TCP ${XRDP_PORT} 未监听"
[[ "$PUBLIC_OK" == "1" ]] && log "✅ ${XRDP_PORT} 对局域网接口开放" || err "❌ ${XRDP_PORT} 仅本机或未监听"
[[ "$XVNC_OK" == "1" ]] && log "✅ Xvnc backend 可用" || err "❌ Xvnc 不可用"
[[ "$CONFIG_OK" == "1" ]] && log "✅ XRDP autorun=Xvnc" || err "❌ Xvnc 配置未通过"
info "========================================"

echo
echo "监听状态："
port_lines "$XRDP_PORT" || true
echo

# ============================================================
# 10/10 使用说明
# ============================================================
log "[10/10] 连接信息"

echo
echo "Windows:"
echo "  Win + R -> mstsc"
echo "  Computer : ${IP_ADDR}:${XRDP_PORT}"
echo
echo "Ubuntu:"
echo "  Remmina -> RDP"
echo "  Server   : ${IP_ADDR}:${XRDP_PORT}"
echo
echo "登录："
echo "  Session  : Xvnc（已配置 autorun=Xvnc）"
echo "  Username : ${TARGET_USER}"
echo "  Password : ${TARGET_USER} 的 Linux 登录密码"
echo
echo "注意："
echo "  - 不需要 VNC Viewer"
echo "  - 不需要手动开放 5901/5902"
echo "  - Xvnc 仅作为 XRDP 的本机内部后端"
echo "  - 对外只需要 ${XRDP_PORT}/tcp"

cat <<EOF

# ---------- 常用命令 ----------

# 服务
sudo systemctl status xrdp --no-pager -l
sudo systemctl status xrdp-sesman --no-pager -l
sudo systemctl restart xrdp
sudo systemctl restart xrdp-sesman

# 日志
sudo tail -n 150 /var/log/xrdp.log
sudo tail -n 150 /var/log/xrdp-sesman.log
sudo journalctl -u xrdp -u xrdp-sesman -n 150 --no-pager

# 端口
sudo ss -ltnp | grep ${XRDP_PORT}

# 检查是否还有旧 VNC 对外服务
ps -ef | grep -E 'x11vnc|Xtigervnc|vncserver|tightvnc|wayvnc' | grep -v grep
sudo ss -ltnp | grep -E ':590[0-9]'

# 检查 Xvnc 后端
command -v Xvnc
grep -A12 '^\\[Xvnc\\]' /etc/xrdp/sesman.ini
grep -A8 '^\\[Xvnc\\]' /etc/xrdp/xrdp.ini
grep '^autorun=' /etc/xrdp/xrdp.ini

# 黑屏/登录即退出
tail -n 150 ~/.xsession-errors 2>/dev/null
ls -la ~/.xsession ~/.cache/sessions 2>/dev/null
sudo tail -n 150 /var/log/xrdp-sesman.log

# 用户密码
sudo passwd ${TARGET_USER}

# 当前 Headless target
systemctl get-default

# 厂商 Xorg 包版本（本方案不修改）
dpkg-query -W -f='\${Package}\\t\${Version}\\n' xserver-xorg-core xserver-common 2>/dev/null
apt-cache policy xserver-xorg-core xserver-common

# 旧配置备份
sudo ls -la ${BACKUP_DIR} 2>/dev/null

EOF

if [[ "$XRDP_OK" == "1" && "$SESMAN_OK" == "1" && "$PORT_OK" == "1" && "$PUBLIC_OK" == "1" && "$XVNC_OK" == "1" && "$CONFIG_OK" == "1" ]]; then
    info "========================================"
    log "✅ RK3576 XRDP + Xvnc + Xfce 部署完成"
    log "   ${IP_ADDR}:${XRDP_PORT}"
    info "========================================"
    exit 0
fi

info "========================================"
err "部署未完全通过，自动输出诊断："
info "========================================"

sudo systemctl status xrdp --no-pager -l || true
echo
sudo systemctl status xrdp-sesman --no-pager -l 2>/dev/null || true
echo
sudo journalctl -u xrdp -u xrdp-sesman -n 150 --no-pager || true
echo
sudo tail -n 150 /var/log/xrdp.log 2>/dev/null || true
echo
sudo tail -n 150 /var/log/xrdp-sesman.log 2>/dev/null || true
echo
grep -A15 '^\[Xvnc\]' /etc/xrdp/sesman.ini 2>/dev/null || true
echo
grep -A10 '^\[Xvnc\]' /etc/xrdp/xrdp.ini 2>/dev/null || true

exit 1
