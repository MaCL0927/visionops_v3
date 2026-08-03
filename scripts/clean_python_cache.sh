#!/bin/bash

# ============================================
# 配置区：在此列出需要保留的服务名称
# 格式：每个服务占一行，以 # 开头的行为注释
# 如果此列表为空，则删除所有 visionops 服务
# ============================================
KEEP_SERVICES=(
    # 保留的服务列表
    # 在此添加更多需要保留的服务
)

# ============================================
# 脚本开始 - 请勿修改以下内容
# ============================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VisionOps 服务清理脚本 (增强版)"
echo "=========================================="
echo ""

# 显示配置的保留服务
echo "配置保留的服务 (${#KEEP_SERVICES[@]} 个):"
if [ ${#KEEP_SERVICES[@]} -eq 0 ]; then
    echo "  (无 - 将删除所有 visionops 服务)"
else
    for svc in "${KEEP_SERVICES[@]}"; do
        echo "  ✓ $svc"
    done
fi
echo ""

# 获取所有 visionops 服务（改进的提取逻辑）
echo "正在扫描 visionops 服务..."
# 使用 awk 提取服务名：匹配包含 "visionops" 的行，提取第二列（服务名）
# 同时过滤掉空行和状态符号
ALL_SERVICES=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $2}' | grep -v '^$' | sort -u)

# 如果上面提取不到，回退到提取第一列并清理符号
if [ -z "$ALL_SERVICES" ]; then
    ALL_SERVICES=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $1}' | sed 's/^●//' | grep -v '^$' | sort -u)
fi

SERVICE_COUNT=$(echo "$ALL_SERVICES" | grep -c . || echo 0)

if [ $SERVICE_COUNT -eq 0 ]; then
    echo "未找到任何 visionops 服务"
    exit 0
fi

echo "找到 $SERVICE_COUNT 个 visionops 服务"
echo ""

# 构建要删除的服务列表
TO_DELETE=()
echo "检查服务..."
for svc in $ALL_SERVICES; do
    # 清理可能的空白字符
    svc=$(echo "$svc" | xargs)
    
    # 检查是否在保留列表中
    KEEP=false
    for keep in "${KEEP_SERVICES[@]}"; do
        if [ "$svc" == "$keep" ]; then
            KEEP=true
            break
        fi
    done
    
    if [ "$KEEP" = true ]; then
        echo "  ⏭ 保留: $svc"
    else
        echo "  ✗ 删除: $svc"
        TO_DELETE+=("$svc")
    fi
done

if [ ${#TO_DELETE[@]} -eq 0 ]; then
    echo ""
    echo "没有需要删除的服务"
    exit 0
fi

echo ""
echo "=========================================="
echo "将要删除 ${#TO_DELETE[@]} 个服务"
echo "=========================================="

# 确认操作
read -p "确认继续? (输入 y/Y 确认): " -r CONFIRM
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "开始删除服务..."

# 删除服务
for svc in "${TO_DELETE[@]}"; do
    echo "  处理: $svc"
    
    # 停止服务（忽略错误）
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        sudo systemctl stop "$svc" 2>/dev/null || true
        echo "    已停止"
    fi
    
    # 禁用服务（忽略错误）
    if systemctl is-enabled --quiet "$svc" 2>/dev/null; then
        sudo systemctl disable "$svc" 2>/dev/null || true
        echo "    已禁用"
    fi
    
    # 取消屏蔽（忽略错误）
    if systemctl status "$svc" 2>/dev/null | grep -q masked; then
        sudo systemctl unmask "$svc" 2>/dev/null || true
        echo "    已取消屏蔽"
    fi
    
    # 删除服务文件（更彻底的搜索）
    SVC_FILES=$(sudo find /etc/systemd/system /lib/systemd/system -name "$svc" -o -name "${svc%.service}" 2>/dev/null)
    if [ -n "$SVC_FILES" ]; then
        echo "$SVC_FILES" | while read -r file; do
            sudo rm -f "$file"
            echo "    已删除: $file"
        done
    else
        # 尝试查找包含服务名的文件（处理符号链接）
        SVC_FILES=$(sudo find /etc/systemd/system /lib/systemd/system -name "*${svc%.service}*" -type f 2>/dev/null | head -5)
        if [ -n "$SVC_FILES" ]; then
            echo "$SVC_FILES" | while read -r file; do
                sudo rm -f "$file"
                echo "    已删除: $file"
            done
        else
            echo "    未找到服务文件（可能已删除）"
        fi
    fi
    
    # 重置失败状态
    sudo systemctl reset-failed "$svc" 2>/dev/null || true
done

# 重新加载 systemd
echo ""
echo "重新加载 systemd..."
sudo systemctl daemon-reload
sudo systemctl reset-failed

# 额外清理：删除所有残留的 visionops 相关文件
echo ""
echo "清理残留的 visionops 配置文件..."
sudo find /etc/systemd/system -name "*visionops*" -type f -delete 2>/dev/null || true
sudo find /lib/systemd/system -name "*visionops*" -type f -delete 2>/dev/null || true

# 再次重新加载
sudo systemctl daemon-reload

# 显示结果
echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="

# 显示剩余服务
REMAINING=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $2}' | grep -v '^$')
if [ -z "$REMAINING" ]; then
    echo "✓ 未找到任何 visionops 服务"
else
    echo "⚠ 剩余 visionops 服务:"
    echo "$REMAINING" | while read -r svc; do
        echo "  ✓ $svc"
    done
    echo ""
    echo "如果仍有残留，请手动运行:"
    echo "  sudo find /etc/systemd/system /lib/systemd/system -name '*visionops*' -delete"
    echo "  sudo systemctl daemon-reload"
fi

echo ""
echo "=========================================="