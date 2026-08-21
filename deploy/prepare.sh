#!/usr/bin/env bash
# ============================================================
# DeepSeek V4 Flash 双机部署 - 环境准备脚本
#
# 设计原则：
#   1. 整个脚本由普通用户运行，不要使用 sudo bash prepare.sh。
#   2. Docker、Python 虚拟环境和模型下载均以当前用户身份执行。
#   3. 只有创建受保护目录及配置 RoCE 网络时，才对单条命令使用 sudo。
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
PREPARE_VENV_DIR="${PREPARE_VENV_DIR:-$SCRIPT_DIR/.venv}"
CURRENT_USER="$(id -un)"
CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"

# --- 颜色 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { printf "${GREEN}[✓]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
error() { printf "${RED}[✗]${NC} %s\n" "$*" >&2; }
step()  { printf "\n${BLUE}▶ %s${NC}\n" "$*"; }
ask()   { printf "${CYAN}[?]${NC} %s\n" "$*"; }

usage() {
    cat <<'EOF'
用法：
  bash prepare.sh                  打开交互式菜单
  bash prepare.sh --check          只读检查，不修改系统
  bash prepare.sh --image          拉取基础镜像并构建稳定运行时
  bash prepare.sh --model          下载或续传模型
  bash prepare.sh --roce [角色]    配置双路 RoCE；角色为 head 或 worker
  bash prepare.sh --all [角色]     执行镜像、RoCE、模型三项准备
  bash prepare.sh --help           显示帮助

必须使用普通用户运行。脚本只会在创建受保护的模型目录和配置网络时请求 sudo。
EOF
}

require_regular_user() {
    if [ "$EUID" -eq 0 ]; then
        error "不要以 root 运行整个脚本。"
        echo "  请退出 root shell，然后以目标普通用户执行：bash prepare.sh"
        echo "  脚本会在配置 /data 或 RoCE 时自行调用 sudo。"
        return 1
    fi
}

validate_config() {
    local name
    for name in IMAGE BASE_IMAGE MODEL_PATH HEAD_IP WORKER_IP HEAD_IP_SECONDARY \
        WORKER_IP_SECONDARY NCCL_INTF NCCL_INTF_SECONDARY; do
        if [ -z "${!name:-}" ]; then
            error "config.sh 缺少必需配置：$name"
            return 1
        fi
    done

    if [[ "$MODEL_PATH" != /* ]]; then
        error "MODEL_PATH 必须是绝对路径：$MODEL_PATH"
        return 1
    fi

    case "$MODEL_PATH" in
        /|/data|/data/models|"$HOME")
            error "MODEL_PATH 过于宽泛，拒绝操作：$MODEL_PATH"
            return 1
            ;;
    esac
}

ensure_sudo() {
    if ! command -v sudo >/dev/null 2>&1; then
        error "当前操作需要管理员权限，但系统中没有 sudo"
        return 1
    fi
    if ! sudo -v; then
        error "无法获取 sudo 权限"
        return 1
    fi
}

run_privileged() {
    sudo -- "$@"
}

# ============================================================
# Docker：不使用 sudo docker，确保后续启动脚本也能由普通用户运行
# ============================================================
check_docker_access() {
    if ! command -v docker >/dev/null 2>&1; then
        error "未安装 Docker"
        return 1
    fi

    if docker info >/dev/null 2>&1; then
        return 0
    fi

    error "当前用户 $CURRENT_USER 无法访问 Docker daemon"
    if [ -S /var/run/docker.sock ]; then
        echo "  请执行：sudo usermod -aG docker $CURRENT_USER"
        echo "  然后注销并重新登录（或执行 newgrp docker 后重新运行脚本）。"
    else
        echo "  Docker socket 不存在，请先确认 Docker 服务已启动。"
    fi
    echo "  不建议用 sudo docker 绕过，否则后续启动仍会遇到权限问题。"
    return 1
}

do_pull_image() {
    step "1/3 准备 Docker 镜像：${IMAGE}"
    check_docker_access || return 1

    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        warn "镜像已存在，跳过下载"
        docker image ls "$IMAGE" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'
        return 0
    fi

    if [ "$BUILD_STABLE_RUNTIME" = "1" ] && [ "$IMAGE" != "$BASE_IMAGE" ]; then
        if [ ! -x "$SCRIPT_DIR/stable-runtime/build.sh" ]; then
            error "缺少稳定运行时构建脚本：$SCRIPT_DIR/stable-runtime/build.sh"
            return 1
        fi
        info "将拉取 digest 固定的基础镜像并构建原生 Anthropic 长 Agent 稳定薄层"
        BASE_IMAGE="$BASE_IMAGE" IMAGE="$IMAGE" \
            "$SCRIPT_DIR/stable-runtime/build.sh"
    else
        echo "开始下载 Docker 镜像（约 20 GB）..."
        if ! docker pull "$IMAGE"; then
            error "镜像下载失败，请检查网络或镜像地址"
            return 1
        fi
    fi
    info "镜像准备完成：$IMAGE"
}

# ============================================================
# 模型目录：仅在必要时 sudo，最终归当前普通用户所有
# ============================================================
ensure_model_directory() {
    if [ ! -d "$MODEL_PATH" ]; then
        if mkdir -p -- "$MODEL_PATH" 2>/dev/null; then
            info "已创建模型目录：$MODEL_PATH"
        else
            warn "创建 $MODEL_PATH 需要管理员权限；只会提权创建这一目录"
            ensure_sudo || return 1
            run_privileged mkdir -p -- "$MODEL_PATH"
            run_privileged chown "$CURRENT_UID:$CURRENT_GID" -- "$MODEL_PATH"
        fi
    fi

    local foreign_owner=""
    foreign_owner="$(find "$MODEL_PATH" -xdev ! -uid "$CURRENT_UID" -print -quit 2>/dev/null || true)"
    if [ -n "$foreign_owner" ]; then
        warn "模型目录中存在非 $CURRENT_USER 所有的文件：$foreign_owner"
        warn "将只修正 $MODEL_PATH 内的所有权，不会修改整个 /data"
        ensure_sudo || return 1
        run_privileged chown -R "$CURRENT_UID:$CURRENT_GID" -- "$MODEL_PATH"
    fi

    if [ ! -w "$MODEL_PATH" ]; then
        warn "模型目录不可写，将只为当前用户修正该目录权限"
        ensure_sudo || return 1
        run_privileged chown "$CURRENT_UID:$CURRENT_GID" -- "$MODEL_PATH"
        run_privileged chmod u+rwx -- "$MODEL_PATH"
    fi

    if [ ! -w "$MODEL_PATH" ]; then
        error "模型目录仍不可写：$MODEL_PATH"
        return 1
    fi
    info "模型目录可写，所有者：$(stat -c '%U:%G' "$MODEL_PATH")"
}

ensure_modelscope() {
    if ! command -v python3 >/dev/null 2>&1; then
        error "未安装 python3"
        return 1
    fi

    if [ ! -x "$PREPARE_VENV_DIR/bin/python" ]; then
        step "创建当前用户专用 Python 虚拟环境：$PREPARE_VENV_DIR"
        if ! python3 -m venv "$PREPARE_VENV_DIR"; then
            error "无法创建 Python 虚拟环境"
            echo "  Ubuntu/DGX OS 可执行：sudo apt install python3-venv"
            return 1
        fi
    fi

    if ! "$PREPARE_VENV_DIR/bin/python" -c 'import modelscope' >/dev/null 2>&1; then
        echo "在项目虚拟环境中安装 ModelScope（不会写入系统 Python）..."
        "$PREPARE_VENV_DIR/bin/python" -m pip install --upgrade 'modelscope>=1.39,<2'
    fi

    if [ ! -x "$PREPARE_VENV_DIR/bin/modelscope" ]; then
        error "ModelScope CLI 安装不完整：$PREPARE_VENV_DIR/bin/modelscope"
        return 1
    fi
    info "ModelScope 已就绪：$PREPARE_VENV_DIR"
}

model_is_complete() {
    python3 - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "config.json",
    root / "tokenizer.json",
    root / "tokenizer_config.json",
    root / "model.safetensors.index.json",
]
missing = [str(path.name) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    print("缺少基础文件：" + ", ".join(missing))
    raise SystemExit(1)

try:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
    print(f"权重索引无效：{exc}")
    raise SystemExit(1)

missing_shards = [name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing_shards:
    preview = ", ".join(missing_shards[:5])
    suffix = " ..." if len(missing_shards) > 5 else ""
    print(f"缺少 {len(missing_shards)} 个权重分片：{preview}{suffix}")
    raise SystemExit(1)

print(f"基础文件及 {len(shards)} 个权重分片齐全")
PY
}

check_model_free_space() {
    local probe="$MODEL_PATH"
    while [ ! -e "$probe" ] && [ "$probe" != "/" ]; do
        probe="$(dirname "$probe")"
    done

    local available_kib required_kib
    available_kib="$(df -Pk "$probe" | awk 'NR==2 {print $4}')"
    required_kib=$((180 * 1024 * 1024))
    if [ -n "$available_kib" ] && [ "$available_kib" -lt "$required_kib" ]; then
        warn "可用空间不足 180 GiB；完整下载可能失败（当前约 $((available_kib / 1024 / 1024)) GiB）"
    fi
}

do_download_model() {
    step "2/3 准备模型权重：$MODEL_ID"
    ensure_model_directory || return 1

    local force_download=0 check_result=""
    if check_result="$(model_is_complete 2>&1)"; then
        local model_size
        model_size="$(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}')"
        warn "模型已完整：$MODEL_PATH（${model_size:-未知大小}；$check_result）"
        read -r -p "     是否强制重新下载并覆盖同名文件？[y/N] " yn
        if [[ ! "${yn:-}" =~ ^[Yy]$ ]]; then
            info "跳过下载"
            return 0
        fi
        force_download=1
    elif [ -n "$(find "$MODEL_PATH" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        warn "检测到未完成的模型目录：$check_result"
        info "将保留已有文件，并由 ModelScope 续传缺失内容"
    fi

    check_model_free_space
    ensure_modelscope || return 1

    local download_args=(download "$MODEL_ID" --local-dir "$MODEL_PATH")
    if [ "$force_download" -eq 1 ]; then
        download_args+=(--force)
    fi

    echo "开始从 ModelScope 下载模型（完整模型约 156 GB）..."
    echo "目标路径：$MODEL_PATH"
    if ! "$PREPARE_VENV_DIR/bin/modelscope" "${download_args[@]}"; then
        error "模型下载失败；已下载文件会保留，下次运行可继续"
        return 1
    fi

    if ! check_result="$(model_is_complete 2>&1)"; then
        error "下载结束但完整性检查未通过：$check_result"
        return 1
    fi

    info "模型下载并验证完成：$(du -sh "$MODEL_PATH" | awk '{print $1}')，$check_result"
}

# ============================================================
# RoCE：只在这里请求 sudo，并通过 NetworkManager 持久化
# ============================================================
interface_has_ip() {
    local interface="$1" expected_ip="$2"
    ip -o -4 addr show dev "$interface" scope global 2>/dev/null \
        | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$expected_ip"
}

select_node_role() {
    local allow_prompt="${1:-yes}"

    case "${NODE_ROLE:-}" in
        head|worker) return 0 ;;
        '') ;;
        *)
            error "NODE_ROLE 只能是 head 或 worker：$NODE_ROLE"
            return 1
            ;;
    esac

    if interface_has_ip "$NCCL_INTF" "$HEAD_IP" \
        || interface_has_ip "$NCCL_INTF_SECONDARY" "$HEAD_IP_SECONDARY"; then
        NODE_ROLE="head"
        info "根据现有 RoCE IP 自动识别本机角色：head"
        return 0
    fi
    if interface_has_ip "$NCCL_INTF" "$WORKER_IP" \
        || interface_has_ip "$NCCL_INTF_SECONDARY" "$WORKER_IP_SECONDARY"; then
        NODE_ROLE="worker"
        info "根据现有 RoCE IP 自动识别本机角色：worker"
        return 0
    fi

    if [ "$allow_prompt" != "yes" ]; then
        error "无法根据当前地址判断节点角色；请传入 head 或 worker"
        return 1
    fi

    echo ""
    echo "  本机是 Head 还是 Worker？"
    echo "    1) Head   ($HEAD_IP / $HEAD_IP_SECONDARY)"
    echo "    2) Worker ($WORKER_IP / $WORKER_IP_SECONDARY)"
    read -r -p "  请选择 [1/2]：" role_choice
    case "${role_choice:-}" in
        1) NODE_ROLE="head" ;;
        2) NODE_ROLE="worker" ;;
        *) error "无效选择"; return 1 ;;
    esac
}

validate_roce_interface() {
    local interface="$1" expected_ip="$2"
    if ! ip link show dev "$interface" >/dev/null 2>&1; then
        error "RoCE 网卡不存在：$interface"
        return 1
    fi

    local mtu
    mtu="$(cat "/sys/class/net/$interface/mtu")"
    if ! interface_has_ip "$interface" "$expected_ip"; then
        local actual
        actual="$(ip -o -4 addr show dev "$interface" scope global 2>/dev/null | awk '{print $4}' | paste -sd, -)"
        error "$interface 地址不匹配：期望 $expected_ip/24，实际 ${actual:-未配置}"
        return 1
    fi
    if [ "$mtu" != "9000" ]; then
        error "$interface MTU 不匹配：期望 9000，实际 $mtu"
        return 1
    fi
    info "$interface：$expected_ip/24，MTU 9000"
}

configure_roce_nm() {
    local interface="$1" address="$2" connection_name="$3"

    run_privileged nmcli device set "$interface" managed yes

    if nmcli -g connection.id connection show "$connection_name" >/dev/null 2>&1; then
        run_privileged nmcli connection modify "$connection_name" \
            connection.interface-name "$interface" \
            connection.autoconnect yes \
            802-3-ethernet.mtu 9000 \
            ipv4.method manual \
            ipv4.addresses "$address/24" \
            ipv4.gateway "" \
            ipv4.dns "" \
            ipv4.never-default yes \
            ipv6.method disabled
    else
        run_privileged nmcli connection add \
            type ethernet \
            ifname "$interface" \
            con-name "$connection_name" \
            connection.autoconnect yes \
            802-3-ethernet.mtu 9000 \
            ipv4.method manual \
            ipv4.addresses "$address/24" \
            ipv4.never-default yes \
            ipv6.method disabled
    fi

    run_privileged nmcli --wait 30 connection up "$connection_name"
}

configure_roce_transient() {
    local interface="$1" address="$2"
    local cidr

    while IFS= read -r cidr; do
        [ "$cidr" = "$address/24" ] && continue
        run_privileged ip address del "$cidr" dev "$interface"
    done < <(ip -o -4 addr show dev "$interface" scope global 2>/dev/null | awk '{print $4}')

    if ! interface_has_ip "$interface" "$address"; then
        run_privileged ip address add "$address/24" dev "$interface"
    fi
    run_privileged ip link set dev "$interface" mtu 9000 up
}

do_setup_roce() {
    step "3/3 配置双路 RoCE 网络"
    select_node_role yes || return 1

    local primary_ip secondary_ip
    if [ "$NODE_ROLE" = "head" ]; then
        primary_ip="$HEAD_IP"
        secondary_ip="$HEAD_IP_SECONDARY"
    else
        primary_ip="$WORKER_IP"
        secondary_ip="$WORKER_IP_SECONDARY"
    fi
    info "本机角色：$NODE_ROLE"

    local interface
    for interface in "$NCCL_INTF" "$NCCL_INTF_SECONDARY"; do
        if ! ip link show dev "$interface" >/dev/null 2>&1; then
            error "RoCE 网卡不存在：$interface"
            echo "  可用网卡："
            ip -br link show | awk '$1 != "lo" {print "    " $1}'
            return 1
        fi
    done

    ensure_sudo || return 1

    if command -v nmcli >/dev/null 2>&1 && nmcli general status >/dev/null 2>&1; then
        info "使用 NetworkManager 创建持久化连接 roce-primary / roce-secondary"
        configure_roce_nm "$NCCL_INTF" "$primary_ip" roce-primary
        configure_roce_nm "$NCCL_INTF_SECONDARY" "$secondary_ip" roce-secondary
    else
        warn "NetworkManager 不可用，只能应用临时 IP；重启后需要重新配置"
        configure_roce_transient "$NCCL_INTF" "$primary_ip"
        configure_roce_transient "$NCCL_INTF_SECONDARY" "$secondary_ip"
    fi

    validate_roce_interface "$NCCL_INTF" "$primary_ip" || return 1
    validate_roce_interface "$NCCL_INTF_SECONDARY" "$secondary_ip" || return 1
    info "RoCE 配置完成；NCCL 保持动态 GID 选择，不固定 NCCL_IB_GID_INDEX"
}

# ============================================================
# 只读预检查
# ============================================================
do_check() {
    step "只读环境检查（不会调用 sudo，也不会修改系统）"
    local failed=0 check_result=""

    if check_docker_access; then
        info "Docker：普通用户可访问"
    else
        failed=1
    fi

    if [ -d "$MODEL_PATH" ]; then
        if check_result="$(model_is_complete 2>&1)"; then
            info "模型：$check_result"
        else
            error "模型不完整：$check_result"
            failed=1
        fi
        if [ -w "$MODEL_PATH" ]; then
            info "模型目录：当前用户可写（$(stat -c '%U:%G' "$MODEL_PATH")）"
        else
            warn "模型目录当前用户不可写；执行 --model 时会仅针对该目录请求 sudo 修正"
        fi
    else
        warn "模型目录尚不存在：$MODEL_PATH"
        failed=1
    fi

    if select_node_role no; then
        local primary_ip secondary_ip
        if [ "$NODE_ROLE" = "head" ]; then
            primary_ip="$HEAD_IP"
            secondary_ip="$HEAD_IP_SECONDARY"
        else
            primary_ip="$WORKER_IP"
            secondary_ip="$WORKER_IP_SECONDARY"
        fi
        validate_roce_interface "$NCCL_INTF" "$primary_ip" || failed=1
        validate_roce_interface "$NCCL_INTF_SECONDARY" "$secondary_ip" || failed=1
    else
        failed=1
    fi

    if [ "$failed" -eq 0 ]; then
        info "环境检查通过"
    else
        error "环境检查发现问题"
        return 1
    fi
}

# ============================================================
# 一键执行及菜单
# ============================================================
do_all() {
    echo ""
    echo "========================================"
    echo " 一键执行全部环境准备"
    echo "========================================"

    do_pull_image || { error "镜像准备失败，中止"; return 1; }
    do_setup_roce || { error "RoCE 配置失败，中止"; return 1; }
    do_download_model || { error "模型准备失败，中止"; return 1; }

    echo ""
    info "全部环境准备完成"
    echo "下一步："
    echo "  Head：   bash start-head.sh"
    echo "  Worker： bash start-worker.sh"
}

show_menu() {
    clear 2>/dev/null || true
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║   DeepSeek V4 Flash 双机部署 - 环境准备         ║"
    echo "╠════════════════════════════════════════════════╣"
    printf "║  运行用户：%-36s║\n" "$CURRENT_USER"
    printf "║  模型目录：%-36s║\n" "$MODEL_PATH"
    echo "╠════════════════════════════════════════════════╣"
    echo "║  0) 只读环境检查                               ║"
    echo "║  1) 准备稳定 Docker 镜像（普通用户）           ║"
    echo "║  2) 下载/续传模型（普通用户）                   ║"
    echo "║  3) 持久化双路 RoCE（仅此步骤调用 sudo）        ║"
    echo "║  9) 一键全部执行（1 → 3 → 2）                  ║"
    echo "║  q) 退出                                        ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
}

run_menu() {
    local choice
    while true; do
        show_menu
        read -r -p "  请选择 > " choice
        case "${choice:-}" in
            0) do_check || true ;;
            1) do_pull_image || true ;;
            2) do_download_model || true ;;
            3) do_setup_roce || true ;;
            9) do_all || true ;;
            q|Q) echo "退出"; return 0 ;;
            *) echo "无效选择" ;;
        esac

        if [[ "${choice:-}" != 9 && "${choice:-}" != q && "${choice:-}" != Q ]]; then
            echo ""
            read -r -p "按 Enter 返回菜单..." _
        fi
    done
}

main() {
    require_regular_user || exit 1
    validate_config || exit 1

    case "${1:-}" in
        '') run_menu ;;
        --check) do_check ;;
        --image) do_pull_image ;;
        --model) do_download_model ;;
        --roce)
            NODE_ROLE="${2:-${NODE_ROLE:-}}"
            do_setup_roce
            ;;
        --all)
            NODE_ROLE="${2:-${NODE_ROLE:-}}"
            do_all
            ;;
        --help|-h) usage ;;
        *)
            error "未知参数：$1"
            usage
            exit 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
