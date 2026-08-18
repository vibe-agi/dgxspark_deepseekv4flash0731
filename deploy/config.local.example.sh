#!/usr/bin/env bash
# Copy to config.local.sh on both nodes and edit locally.
# config.local.sh is ignored by Git. Keep hostnames, management IPs, mount paths,
# registry credentials, and other machine-specific values there.

MODEL_PATH="${MODEL_PATH:-/data/models/deepseek-ai/DeepSeek-V4-Flash-0731}"

# 1: pull BASE_IMAGE and build the PR #50686 overlay locally (recommended).
# 0: pull IMAGE directly, useful when you publish the stable image yourself.
BUILD_STABLE_RUNTIME="${BUILD_STABLE_RUNTIME:-1}"

# Example RoCE data-plane values. These are not client/API addresses.
HEAD_IP="${HEAD_IP:-10.10.12.11}"
WORKER_IP="${WORKER_IP:-10.10.12.21}"
HEAD_IP_SECONDARY="${HEAD_IP_SECONDARY:-10.10.13.11}"
WORKER_IP_SECONDARY="${WORKER_IP_SECONDARY:-10.10.13.21}"

NCCL_INTF="${NCCL_INTF:-enp1s0f0np0}"
NCCL_INTF_SECONDARY="${NCCL_INTF_SECONDARY:-enP2p1s0f0np0}"
NCCL_IB_HCAS="${NCCL_IB_HCAS:-=rocep1s0f0:1,roceP2p1s0f0:1}"
ROCE_SUBNET="${ROCE_SUBNET:-10.10.12.0/23}"

# Six active sequences with synchronous scheduling. Lower to 1 for strict
# single-request isolation or when diagnosing prefix-cache/concurrency issues.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-5}"
