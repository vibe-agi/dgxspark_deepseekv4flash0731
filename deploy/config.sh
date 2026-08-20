#!/bin/bash
# ============================================================
# DeepSeek V4 Flash 双机部署 - 配置文件
# 在 Head 和 Worker 节点上使用相同的配置
# ============================================================

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 机器专属值放在不入 Git 的 config.local.sh。示例文件使用 `${VAR:-value}`，
# 因此显式环境变量仍能在单次启动时取得最高优先级。
if [ -f "$CONFIG_DIR/config.local.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONFIG_DIR/config.local.sh"
fi

# --- Docker 镜像 (必须预先拉取) ---
# IMAGE: Docker 镜像 (registry 地址，新设备上 docker pull 即可获取)
IMAGE="${IMAGE:-deepseek-v4-flash:0.1.1-stable-nvfp4-20260819}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1}"
BUILD_STABLE_RUNTIME="${BUILD_STABLE_RUNTIME:-1}"
# 回滚镜像: IMAGE="$BASE_IMAGE"

# --- 模型权重路径 (host 上的绝对路径) ---
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MODEL_PATH="${MODEL_PATH:-/data/models/deepseek-ai/DeepSeek-V4-Flash-0731}"
# 容器内挂载路径(无需修改)
MODEL_MOUNT="${MODEL_MOUNT:-/models/dsv4}"

# --- 网络配置 ---
HEAD_IP="${HEAD_IP:-10.10.12.11}"             # Head 节点 RoCE IP
WORKER_IP="${WORKER_IP:-10.10.12.21}"         # Worker 节点 RoCE IP
HEAD_IP_SECONDARY="${HEAD_IP_SECONDARY:-10.10.13.11}"
WORKER_IP_SECONDARY="${WORKER_IP_SECONDARY:-10.10.13.21}"
MASTER_PORT="${MASTER_PORT:-25000}"
NCCL_INTF="${NCCL_INTF:-enp1s0f0np0}"
NCCL_INTF_SECONDARY="${NCCL_INTF_SECONDARY:-enP2p1s0f0np0}"
NCCL_IB_HCAS="${NCCL_IB_HCAS:-=rocep1s0f0:1,roceP2p1s0f0:1}"
ROCE_SUBNET="${ROCE_SUBNET:-10.10.12.0/23}"

# 节点角色 (本机是 Head 还是 Worker — prepare.sh 会询问)
NODE_ROLE="${NODE_ROLE:-}"        # "head" 或 "worker"；prepare.sh 也会自动检测/询问

# --- 缓存与临时目录 ---
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}" # 当前运行用户的 HF/FlashInfer 缓存
TMP_DIR="${TMP_DIR:-${HOME}/dspark-tmp}"            # 当前运行用户的临时目录

# --- vLLM 参数 (无需修改) ---
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}" # 1M 上下文
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}" # 最多 6 条活跃序列；保留同步调度以兼顾稳定性
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
GPU_MEM="${GPU_MEM:-0.835}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-5}"
MAX_CUDAGRAPH="${MAX_CUDAGRAPH:-$((MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)))}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
# Single-user/full-window profile: let one long prefill consume the batch.
# Shared services or concurrent subagents should override this to 1024 and may
# also prefer MAX_BATCHED_TOKENS=8192 with GPU_MEM=0.78 for a wider safety margin.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-0}"
VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# Only enable this together with the Issue #26 coordinator hotfix.  Passing it
# to the unpatched/base runtime can create unsafe hybrid-SWA warm-cache hits.
VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
PORT="${PORT:-8888}"
