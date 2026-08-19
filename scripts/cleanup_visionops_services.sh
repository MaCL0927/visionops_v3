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
# 方法1：提取第二列（服务名），过滤掉状态符号
ALL_SERVICES=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $2}' | grep -v '^$' | sort -u)

# 如果提取不到，回退到提取第一列并清理符号
if [ -z "$ALL_SERVICES" ]; then
    ALL_SERVICES=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $1}' | sed 's/^●//' | grep -v '^$' | sort -u)
fi

# 额外：直接扫描服务文件目录，找出所有 visionops 相关的 service 文件
SERVICE_FILES=$(sudo find /etc/systemd/system /lib/systemd/system -name "*visionops*.service" -type f 2>/dev/null | sed 's|.*/||' | sort -u)

# 合并两个来源的服务列表
ALL_SERVICES=$(echo -e "$ALL_SERVICES\n$SERVICE_FILES" | grep -v '^$' | sort -u)

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
    # 清理可能的空白字符和特殊符号
    svc=$(echo "$svc" | xargs)
    
    # 跳过空行
    [ -z "$svc" ] && continue
    
    # 检查是否在保留列表中
    KEEP=false
    for keep in "${KEEP_SERVICES[@]}"; do
        if [ "$svc" == "$keep" ] || [ "$svc" == "${keep%.service}" ]; then
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
echo "${TO_DELETE[@]}" | tr ' ' '\n' | sed 's/^/  - /'

# 确认操作
read -p "确认继续? (输入 y/Y 确认): " -r CONFIRM
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "开始删除服务..."

# 第一步：停止、禁用、屏蔽所有目标服务
for svc in "${TO_DELETE[@]}"; do
    echo "  处理: $svc"
    
    # 停止服务（忽略错误）
    sudo systemctl stop "$svc" 2>/dev/null || true
    echo "    已停止"
    
    # 禁用服务（忽略错误）
    sudo systemctl disable "$svc" 2>/dev/null || true
    echo "    已禁用"
    
    # 取消屏蔽（如果有）
    sudo systemctl unmask "$svc" 2>/dev/null || true
    echo "    已取消屏蔽"
    
    # 重置失败状态
    sudo systemctl reset-failed "$svc" 2>/dev/null || true
done

# 第二步：彻底删除所有 visionops 服务文件
echo ""
echo "删除服务文件..."
sudo find /etc/systemd/system /lib/systemd/system -name "*visionops*.service" -type f -delete 2>/dev/null || true
sudo find /etc/systemd/system /lib/systemd/system -name "*visionops*.socket" -type f -delete 2>/dev/null || true
sudo find /etc/systemd/system -name "*visionops*" -type f -delete 2>/dev/null || true

# 第三步：清理 systemd 的残留状态（关键步骤）
echo ""
echo "清理 systemd 残留状态..."
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 强制重置所有失败的服务状态
sudo systemctl reset-failed

# 清理 systemd 的隐藏状态目录
sudo rm -rf /run/systemd/system/*visionops* 2>/dev/null || true

# 第四步：重新生成 systemd 依赖关系
echo ""
echo "重新生成 systemd 依赖..."
sudo systemctl daemon-reexec 2>/dev/null || true

# 显示结果
echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="

# 检查服务文件是否还存在
echo ""
echo "检查残留的服务文件..."
REMAINING_FILES=$(sudo find /etc/systemd/system /lib/systemd/system -name "*visionops*" -type f 2>/dev/null)
if [ -n "$REMAINING_FILES" ]; then
    echo "⚠ 仍有残留文件:"
    echo "$REMAINING_FILES" | while read -r f; do
        echo "  - $f"
    done
    echo ""
    echo "建议手动删除:"
    echo "  sudo find /etc/systemd/system /lib/systemd/system -name '*visionops*' -delete"
else
    echo "✓ 服务文件已全部清除"
fi

# 显示剩余服务
REMAINING_SERVICES=$(systemctl list-units --type=service --all | grep -i visionops | awk '{print $2}' | grep -v '^$')
if [ -z "$REMAINING_SERVICES" ]; then
    echo "✓ 未找到任何 visionops 服务"
else
    echo "⚠ 剩余 visionops 服务:"
    echo "$REMAINING_SERVICES" | while read -r svc; do
        echo "  - $svc"
    done
    echo ""
    echo "如果仍有残留，请尝试重启 systemd 或重启系统"
fi

echo ""
echo "=========================================="