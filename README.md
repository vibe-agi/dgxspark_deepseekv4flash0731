# Dual DGX Spark Deployment: DeepSeek V4 Flash 0731

Run **DeepSeek V4 Flash 0731** with a 1M context limit across **two NVIDIA DGX Spark** (GB10, 128GB unified memory each) workstations interconnected via dual-path **RoCE**. The stability-first profile uses FP8 KV cache and a reproducible vLLM tokenizer backport for long agent histories.

> 🇨🇳 中文版见下方 | Gitee 镜像：[gitee.com/alexlu0912_admin/dgxspark_deepseekv4flash0731](https://gitee.com/alexlu0912_admin/dgxspark_deepseekv4flash0731)

---

## What This Repository Fixes / 本项目解决的问题

This repository turns the original two-node recipe into a reproducible, long-agent-stable deployment:

- **Long agent histories degrade or leak protocol markup**: the pinned runtime mishandles adjacent assistant text/reasoning/tool-call records. A thin Docker overlay backports vLLM PR #50686, validates it during the build, and includes a split-history regression probe. Model weights are not modified.
- **Head/Worker containers exit during distributed startup**: the scripts enforce unique ranks, Worker-first startup, dual-HCA RoCE, matching IP/MTU checks, and dynamic IPv4 GID selection instead of a fragile hard-coded GID index.
- **Setup assumes the operator is root**: `prepare.sh` now runs as an ordinary user and requests targeted `sudo` only for the protected model directory and RoCE configuration; it no longer recursively changes ownership of all `/data`.
- **A configured 1M limit is mistaken for proven capacity**: the stable profile uses FP8 KV, chunked prefill, prefix caching, `MAX_NUM_SEQS=6`, and synchronous scheduling. It has passed a `999,860`-token single request and six concurrent `79,134`-token shared-prefix requests; six active sequences do not mean 6 × 1M KV capacity.
- **Client and cluster addresses are confused**: OpenAI-compatible clients use the Head management/LAN endpoint, while the example `10.10.12.0/24` and `10.10.13.0/24` networks remain dedicated to Head/Worker RoCE traffic.

In short: it fixes long-session prompt encoding, makes dual-node startup repeatable, removes root-only setup assumptions, and records a tested stability baseline instead of relying only on advertised configuration values.

**中文摘要：**本项目修复了长 Agent 多轮后的协议标签泄露/胡言乱语，解决了双机 NCCL/GID/启动顺序导致的容器退出，将 root 视角的准备脚本改为普通用户可用，并给出经长上下文与 6 路并发验证的 FP8 稳定基线。

---

## Hardware Topology

```text
┌──────────────────────────┐       dual RoCE / MTU 9000       ┌──────────────────────────┐
│ DGX Spark Head           │◄───────────────────────────────►│ DGX Spark Worker         │
│ management: site-local   │   primary + secondary HCA       │ management: site-local   │
│ RoCE examples: .12.11    │                                 │ RoCE examples: .12.21    │
│                .13.11    │                                 │                .13.21    │
└──────────────────────────┘                                 └──────────────────────────┘
```

- **TP=2** across 2 GB10 GPUs (1 per node), **PP=1**, **NNODES=2**
- **Network**: two RoCE v2 paths, dynamic GID selection, MTU 9000
- **Runtime**: FP8 KV, 1M model limit, DSpark speculative decoding
- **Clients**: use the Head management/LAN address, not the RoCE data-plane address

---

## Directory Structure

```
dgxspark_deepseekv4flash0731/
├── README.md                # This file (English + 中文)
├── deploy/                  # ✅ Recommended — latest dual-node deploy scripts
│   ├── config.sh            #    Configuration (image, model path, IPs)
│   ├── config.local.example.sh # Machine-local override template
│   ├── prepare.sh           #    Environment setup (interactive menu)
│   ├── start-head.sh        #    Head node startup script
│   ├── start-worker.sh      #    Worker node startup script
│   ├── stable-runtime/      #    Reproducible vLLM PR #50686 overlay
│   ├── stability-probe.py   #    Multi-turn / long-context regression
│   ├── STABILITY.md         #    Root cause, validation, rollback
│   └── README.md            #    Detailed deployment guide
├── legacy/                  # Legacy reference files from dsv4dspark
│   ├── setup-roce.sh        #    RoCE network auto-config
│   ├── preflight.sh         #    Pre-deployment checks
│   ├── start-all.sh         #    One-click orchestration
│   ├── docker-compose.yml   #    Docker Compose config
│   ├── benchmark-matrix.py  #    Performance benchmark tool
│   ├── *.env                #    Environment variable templates
│   └── *.md                 #    Legacy docs
├── vllm-args.md             # vLLM parameter reference
└── troubleshooting.md       # Troubleshooting guide
```

---

## Quick Start

### Prerequisites

| Item | Details |
|------|---------|
| Base Image | `ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1`; `prepare.sh --image` builds the stable overlay |
| Model Weights | `DeepSeek-V4-Flash-0731` full directory, same path on both nodes |
| OS | DGX Spark (NVIDIA GB10, 128GB), Ubuntu 24.04+, with NVIDIA drivers/docker/nvidia-container-toolkit |

### 1. Environment Setup (Interactive Menu)

```bash
cd deploy && bash prepare.sh
```

Menu options:
```
0) Read-only environment check
1) Build stable Docker image (pulls the base image first)
2) Download model weights  (~156 GB, ModelScope)
3) Persist both RoCE paths  (the only sudo step)
9) Run all
```

### 2. Configuration

Keep tracked defaults generic. Copy the ignored local override and edit that on each node:

```bash
cd deploy
cp config.local.example.sh config.local.sh
```

Do not commit management IPs, SSH aliases, credentials, or personal paths.

### 3. Configure RoCE Network

On each node, let `prepare.sh` create persistent primary and secondary NetworkManager profiles:

```bash
bash prepare.sh --roce head
bash prepare.sh --roce worker
```

### 4. Launch

```bash
# Start Worker first
cd deploy && bash start-worker.sh

# Then start Head
cd deploy && bash start-head.sh
```

### 5. Verify Deployment

#### 5.1 Health Check — Model List

```bash
API_HOST=head-management-hostname
curl -s http://${API_HOST}:8888/v1/models | python3 -m json.tool
```

Expected:
```json
{
  "object": "list",
  "data": [{ "id": "deepseek-v4-flash", "object": "model", "owned_by": "vllm" }]
}
```

> ❌ If `Connection refused`: wait for Head to finish loading (~5-10 min), check `docker logs -f vllm_anemll | grep "Ulysses"`.

#### 5.2 Basic Chat Test

```bash
curl -s http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
    "max_tokens": 128,
    "temperature": 0.6
  }' | python3 -m json.tool
```

#### 5.3 Streaming Test

```bash
curl -s http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Write a haiku about AI."}],
    "max_tokens": 200,
    "stream": true
  }'
```

#### 5.4 Throughput Benchmark

```bash
# Single request latency
curl -s -o /dev/null -w "TTFT: %{time_starttransfer}s | Total: %{time_total}s\n" \
  http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain quantum entanglement in detail."}],
    "max_tokens": 500
  }'

# Concurrent load test (requires vllm benchmark tools)
pip install vllm 2>/dev/null
vllm bench serve \
  --host ${API_HOST} --port 8888 \
  --model deepseek-v4-flash \
  --num-prompts 20 --request-rate 2 \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash-0731
```

Expected benchmarks (TP=2 dual-node):

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | ~1s |
| Throughput (single 500-token request) | **40-60 tok/s** |
| End-to-end latency (500 tokens) | ~10-15s |

#### 5.5 Verify NCCL Communication

```bash
docker logs vllm_anemll 2>&1 | grep -E "NCCL.*comm|NCCL.*rank|NCCL.*Channel"
# Expected: rank 0 nranks 2 (two nodes connected)
```

> If only `rank 0 nranks 1` appears, Worker failed to join — check Worker logs and RoCE connectivity.

#### 5.6 Monitoring Quick Reference

```bash
docker exec vllm_anemll nvidia-smi           # GPU utilization
docker logs -f vllm_anemll                    # Real-time logs
docker stats vllm_anemll                      # Container resource usage
docker logs vllm_anemll 2>&1 | grep -i "worker\|node_rank.*1"  # Worker status
```

#### 5.7 Common Verification Failures

| Symptom | Cause | Solution |
|---------|-------|----------|
| `curl` no response | Head still loading | Wait 5-10 min, check for `Ulysses model is ready` in logs |
| `model not found` | Wrong served-model-name | Use `deepseek-v4-flash` |
| Garbled output | tokenizer not loaded | Check `--tokenizer-mode deepseek_v4` |
| Very low speed (<10 tok/s) | Worker offline or NCCL degraded | Check Worker logs + RoCE |
| `context length exceeds` | Input exceeds limit | Default 1M context, check input size |

---

## Lessons Learned

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| **NODE_RANK conflict** | Worker times out after 5 min | Both set to `--node-rank 0` | Worker must be `--node-rank 1` |
| **Missing `/cache/huggingface` mount** | JIT recompiles every restart | flashinfer autotune cache lost | Mount host directory for persistence |
| **Missing `--gpus all`** | `Failed to infer device type` | No GPU visible in container | Always add `--gpus all` |
| **`-p` + `--network host` conflict** | Port mapping ignored | Docker ignores `-p` in host mode | Use iptables REDIRECT |
| **RoCE IP lost after reboot** | transient `ip addr add` only | no persistent NM profile | run `prepare.sh --roce <role>` during maintenance |
| **Wrong fixed GID index** | NCCL communicator fails on one node | GID table positions differ | do not set `NCCL_IB_GID_INDEX`; select by AF/range |
| **Long agent degrades** | repeated tools or protocol markup | split assistant turns hit old tokenizer | build the stable PR #50686 overlay |

### Performance (Measured)

GB10 provides 128 GB unified memory per node. Observed model loading used about 79 GiB per node; the remaining unified-memory budget is also shared by KV cache, CUDA graphs, JIT workspaces, the runtime, and the OS, so it is not a dedicated 49 GB KV pool.

| Scenario | Throughput | TTFT | Memory |
|----------|-----------|------|--------|
| Code generation (500 tokens) | **~61 tok/s** | ~960ms | 79 GB/node |
| Long text generation (500 tokens) | ~45 tok/s | ~1s | 79 GB/node |

---

## Key vLLM Parameters

### Context Length: `--max-model-len 1048576`

Model natively supports 1M token context (`max_position_embeddings: 1048576` in `config.json`, extended 16× from 65536 via YaRN). No sliding window or extrapolation tricks needed.

### GPU Memory: `--gpu-memory-utilization 0.78`

**This is a safety ceiling, not process RSS.** Unified memory, CUDA graphs, JIT workspaces, and host caches all consume the same pool. The default has completed a near-1M single request; do not infer safe concurrency from weight size alone.

| Value | Effect |
|-------|--------|
| 0.78 (current) | Ceiling ~100 GB, actual ~79 GB, safe |
| 0.75 | Minimum viable; lower may be rejected |
| 0.85 | More aggressive; may OOM at peak concurrency |

### Concurrency: `--max-num-seqs 6` with synchronous scheduling

The validated default keeps up to six sequences active while using `--no-async-scheduling`. A dual-Spark test passed six concurrent requests of `79,134` prompt tokens each. This is a scheduler cap, not capacity for six independent 1M contexts; total live KV still has to fit the measured cache pool.

| Value | Use Case | Memory Pressure |
|-------|----------|-----------------|
| 1 | Isolation / diagnosing a concurrency issue | Lowest scheduler risk |
| 2–5 | Conservative shared service | Validate with the real history shape |
| 6 (default) | Multi-user service | Validated at 6 × 79,134-token shared-prefix requests |
| 7+ | Throughput experiment | Not validated by this profile |

### Full Parameter Reference

```bash
--tensor-parallel-size 2          # TP=2 across 2 GPUs (1 per node)
--pipeline-parallel-size 1        # PP=1
--nnodes 2                        # 2 nodes
--kv-cache-dtype fp8              # Validated stable profile
--block-size 256                  # KV Cache block size
--max-model-len 1048576           # 1M token context (native YaRN)
--max-num-seqs 6                  # Up to 6 active sequences; not 6 × 1M capacity
--max-num-batched-tokens 8192     # Prefill batch (small chunks save memory)
--gpu-memory-utilization 0.78     # Memory safety ceiling (~79GB / 128GB unified)
--enable-chunked-prefill          # Chunk long prompts to avoid prefill OOM
--enable-prefix-caching           # Reuse KV Cache for shared prefixes
--no-async-scheduling             # Avoid long-prefix scheduler races
--speculative-config dspark       # DGX Spark hardware speculative decode
--moe-backend flashinfer_b12x     # B300 GB10-specific MoE backend
```

---

## NCCL Configuration

Dual-node inference depends on **NCCL** over RoCE for cross-node GPU communication.

### Why `--network host` is Mandatory

```
Container net stack  ──NCCL RDMA──►  Physical NIC (enp1s0f0np0)
     ↑                                    ↑
  Bridge mode:                         RoCE requires direct
  RDMA cannot NAT                      physical HCA access
```

NCCL's RDMA data path needs direct physical NIC access. Docker bridge/NAT virtual IPs cause `mlx5` HCA binding failures. `--network host` is the only reliable approach.

### NCCL Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_IB_HCA` | `=rocep1s0f0:1,roceP2p1s0f0:1` | Exact-match both HCA paths |
| `NCCL_IB_MERGE_NICS` | `1` | Merge both paths |
| `NCCL_NET` | `IB` | Force IB/RoCE transport |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | RoCE v2 (UDP encapsulation) |
| `NCCL_CROSS_NIC` | `1` | Allow cross-NIC (single-card compat) |
| `NCCL_CUMEM_ENABLE` | `0` | Disable CUDA mempool (GB10 compat) |
| `NCCL_NVLS_ENABLE` | `0` | Disable NVLink Sharp (GB10 unsupported) |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | Ignore CPU affinity (container required) |
| `NCCL_DEBUG` | `WARN` | Log level (use `INFO` for debugging) |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET` | Force IPv4 |
| `NCCL_IB_ADDR_RANGE` | `10.10.12.0/23` | Restrict dynamic GID selection to both example subnets |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | NCCL communication interface |
| `GLOO_SOCKET_IFNAME` | `enp1s0f0np0` | Gloo communication interface |
| `TP_SOCKET_IFNAME` | `enp1s0f0np0` | PyTorch TP communication interface |

### Dynamic GID selection

The scripts deliberately do not set `NCCL_IB_GID_INDEX`. MTU does not determine
the position of a GID in each host's table. NCCL selects an IPv4/RoCE-v2 GID from
`NCCL_IB_ADDR_RANGE`, allowing the two nodes to use different local indices.

### NCCL Initialization Flow

```
1. Worker starts and waits for Rank 0
2. Head starts and both nodes load weights
3. NCCL init handshake completes (Head ↔ Worker)
   ├─ TCP handshake (MASTER_ADDR:MASTER_PORT)
   ├─ GLOO topology discovery (GLOO_SOCKET_IFNAME)
   └─ NCCL IB connection (dual HCA + dynamically selected GID)
4. Both nodes assigned NCCL rank → sync complete → inference begins
```

> ⚠️ **Critical**: Head `NODE_RANK=0`, Worker `NODE_RANK=1`. Both must be unique.

### NCCL Debugging

```bash
# Check RoCE devices
ibv_devinfo
ibv_devinfo -d rocep1s0f0 -v | grep GID

# View GID table
show_gids | grep -A3 "rocep1s0f0"

# Verbose NCCL logging
docker run ... -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=NET,INIT ...
```

### Common NCCL Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Got completion with error` | RoCE link down / loose cable | Check physical + bidirectional ping |
| `GID index not found` | A stale fixed GID index is still configured | remove `NCCL_IB_GID_INDEX`; verify AF/range and interface IPs |
| `NCCL timeout` | Worker NODE_RANK=0 | Change Worker to `--node-rank 1` |
| `Socket connection refused` | Worker connecting before Head ready | Wait for Head to finish loading |
| `mlx5_0:1 got error from peer` | One-way connectivity | Check firewall, bidirectional ping |

---

## Obtaining Model Weights

### Option 1: HuggingFace (Recommended)

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

> **Note**: The official Flash model directory is about **156 GB** and uses mixed-precision weights. Reserve 180–200 GB+ disk space. KV cache dtype is a separate runtime choice.

### Option 2: ModelScope (Faster in China)

```bash
pip install modelscope
modelscope download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

### Sync to Worker Node

```bash
# From Head, transfer over RoCE
rsync -avz --progress \
  /data/models/deepseek-ai/DeepSeek-V4-Flash-0731/ \
  worker-user@10.10.12.21:/data/models/deepseek-ai/DeepSeek-V4-Flash-0731/
```

---

## Author

[@alexlu0912_admin](https://gitee.com/alexlu0912_admin)

---

## License

MIT

---

---

# DGX Spark 双机部署 DeepSeek V4 Flash 0731

两台 NVIDIA DGX Spark（GB10，每节点 128 GB 统一内存）通过双路 RoCE 互联，运行 **DeepSeek V4 Flash 0731**。稳定优先配置使用 FP8 KV、1M 上限和可复现的 vLLM tokenizer 回移补丁。

> 🇬🇧 English version above | GitHub：[github.com/vibe-agi/dgxspark_deepseekv4flash0731](https://github.com/vibe-agi/dgxspark_deepseekv4flash0731)

---

## 硬件拓扑

```text
┌──────────────────────────┐       双路 RoCE / MTU 9000       ┌──────────────────────────┐
│ DGX Spark Head           │◄───────────────────────────────►│ DGX Spark Worker         │
│ 管理地址：现场配置        │   主 HCA + 第二 HCA             │ 管理地址：现场配置        │
│ RoCE 示例：.12.11/.13.11 │                                 │ RoCE 示例：.12.21/.13.21 │
└──────────────────────────┘                                 └──────────────────────────┘
```

- **TP=2** 跨两块 GB10 GPU（每节点 1 块），**PP=1**，**NNODES=2**
- **网络**：两条 RoCE v2 路径，动态 GID，MTU 9000
- **运行时**：FP8 KV、1M 上限、DSpark speculative decoding
- **客户端**：使用 Head 管理网/LAN 地址，不使用 RoCE 数据面地址

---

## 目录结构

```
dgxspark_deepseekv4flash0731/
├── README.md                # 本文件（中英双语）
├── deploy/                  # ✅ 推荐使用 — 最新双机部署脚本
│   ├── config.sh            #    配置文件 (镜像、模型路径、IP)
│   ├── config.local.example.sh # 本机覆盖模板
│   ├── prepare.sh           #    环境准备脚本 (交互式菜单)
│   ├── start-head.sh        #    Head 节点启动脚本
│   ├── start-worker.sh      #    Worker 节点启动脚本
│   ├── stable-runtime/      #    vLLM PR #50686 稳定薄层
│   ├── stability-probe.py   #    多轮/长上下文回归
│   ├── STABILITY.md         #    根因、验证、回滚
│   └── README.md            #    详细部署说明书
├── legacy/                  # 旧版 dsv4dspark 参考文件
│   ├── setup-roce.sh        #    RoCE 网络自动配置
│   ├── preflight.sh         #    部署前预检
│   ├── start-all.sh         #    一键编排
│   ├── docker-compose.yml   #    Docker Compose 配置
│   ├── benchmark-matrix.py  #    性能测试
│   ├── *.env                #    环境变量模板
│   └── *.md                 #    旧版文档
├── vllm-args.md             # vLLM 参数详解
└── troubleshooting.md       # 故障排查指南
```

---

## 快速开始

### 前置条件

| 项目 | 说明 |
|------|------|
| 基础镜像 | `ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1`；`prepare.sh --image` 构建稳定薄层 |
| 模型权重 | `DeepSeek-V4-Flash-0731` 完整目录，两台节点同一路径 |
| 系统 | DGX Spark (NVIDIA GB10, 128GB 统一内存), Ubuntu 24.04+, 自带驱动/docker/nvidia-container-toolkit |

### 1. 环境准备（交互式菜单）

```bash
cd deploy && bash prepare.sh
```

菜单：
```
0) 只读环境检查
1) 准备稳定 Docker 镜像（先拉基础镜像）
2) 下载模型权重       (~156 GB, ModelScope)
3) 持久化双路 RoCE    (仅此步骤 sudo)
9) 一键全部执行
```

### 2. 配置

保留公开默认值不动，在两台机器分别创建已被 Git 忽略的本机覆盖：

```bash
cd deploy
cp config.local.example.sh config.local.sh
```

不要提交管理网地址、SSH 别名、凭据或个人目录。

### 3. 配置 RoCE 网络

维护窗口中由脚本为两台节点分别建立持久化的主/第二 RoCE profile：

```bash
bash prepare.sh --roce head
bash prepare.sh --roce worker
```

### 4. 启动

```bash
# Worker 先启动
cd deploy && bash start-worker.sh

# Head 随后
cd deploy && bash start-head.sh
```

### 5. 验证部署

#### 5.1 模型列表

```bash
API_HOST=head-management-hostname
curl -s http://${API_HOST}:8888/v1/models | python3 -m json.tool
```

> ❌ 返回 `Connection refused`：等 Head 加载完成（约 5-10 分钟），`docker logs -f vllm_anemll | grep "Ulysses"` 出现即就绪。

#### 5.2 基础对话

```bash
curl -s http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
    "max_tokens": 128,
    "temperature": 0.6
  }' | python3 -m json.tool
```

#### 5.3 流式输出

```bash
curl -s http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "写一首五言绝句"}],
    "max_tokens": 200,
    "stream": true
  }'
```

#### 5.4 吞吐基准

```bash
# 单请求延迟
curl -s -o /dev/null -w "TTFT: %{time_starttransfer}s | Total: %{time_total}s\n" \
  http://${API_HOST}:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "解释量子纠缠"}],
    "max_tokens": 500
  }'
```

参考值：TTFT ~1s，吞吐 40-60 tok/s（500 token 输出）。

#### 5.5 验证 NCCL

```bash
docker logs vllm_anemll 2>&1 | grep -E "NCCL.*comm|NCCL.*rank|NCCL.*Channel"
# 预期：rank 0 nranks 2（两节点连接成功）
```

#### 5.6 监控速查

```bash
docker exec vllm_anemll nvidia-smi           # GPU 使用率
docker logs -f vllm_anemll                    # 实时日志
docker stats vllm_anemll                      # 容器资源占用
docker logs vllm_anemll 2>&1 | grep -i "worker\|node_rank.*1"  # Worker 在线状态
```

#### 5.7 常见验证失败

| 现象 | 原因 | 排查 |
|------|------|------|
| `curl` 无响应 | Head 尚未加载完 | 等 5-10 分钟，看日志是否出现 `Ulysses model is ready` |
| `model not found` | 模型名不匹配 | 使用 `deepseek-v4-flash` |
| 乱码/空白 | tokenizer 未加载 | 检查 `--tokenizer-mode deepseek_v4` |
| 速度 <10 tok/s | Worker 离线或 NCCL 降级 | 检查 Worker 日志 + RoCE |
| `context length exceeds` | 输入超限 | 默认 1M context，检查输入长度 |

---

## 踩坑经验

| 问题 | 现象 | 根因 | 解决 |
|------|------|------|------|
| **NODE_RANK 冲突** | Worker 5 分钟后超时退出 | Head 和 Worker 的 `--node-rank` 都设成 0 | Worker 必须 `--node-rank 1` |
| **缺少 `/cache/huggingface` 挂载** | 每次重启 JIT 编译超慢 | flashinfer autotune 缓存丢失 | 挂载 host 目录持久化 |
| **缺少 `--gpus all`** | `Failed to infer device type` | 容器内无 GPU 可见 | `docker run` 必须加 `--gpus all` |
| **`-p` + `--network host` 互斥** | 端口映射不生效 | Docker 在 host 网络模式下忽略 `-p` | 用 iptables REDIRECT |
| **RoCE IP 重启后丢失** | 只做了临时 `ip addr add` | 没有 NetworkManager 持久连接 | 运行 `prepare.sh --roce <role>` |
| **固定 GID 索引错误** | 某个节点 NCCL communicator 失败 | 两台机的 GID 表位置不一致 | 删除 `NCCL_IB_GID_INDEX`，按 AF/地址范围动态选择 |
| **长 Agent 会话劣化** | 重复工具、泄露协议标签或胡言乱语 | 旧 tokenizer 错误处理拆分的 assistant 记录 | 构建带 vLLM PR #50686 的稳定薄层 |

### 实测性能

GB10 每节点有 128 GB 统一内存。实测模型加载约占 79 GiB/节点；剩余预算还要由 KV cache、CUDA Graph、JIT workspace、运行时与操作系统共享，不能全部当作 KV cache。

| 场景 | 吞吐 | TTFT | 显存 |
|------|------|------|------|
| 代码生成 (500 tokens) | **~61 tok/s** | ~960ms | 79 GB/节点 |
| 长文本生成 (500 tokens) | ~45 tok/s | ~1s | 79 GB/节点 |

---

## vLLM 关键参数

### 上下文长度：`--max-model-len 1048576`

模型原生支持 1M token 上下文（`config.json` 中 `max_position_embeddings: 1048576`，通过 YaRN 从 65536 扩展 16 倍）。无需滑动窗口或外推技巧。

### 显存管理：`--gpu-memory-utilization 0.78`

**这是安全上限，不是实际占用。** 统一内存还要容纳 CUDA Graph、JIT workspace 和系统缓存。当前启动日志报告 GPU KV cache 容量约 `1,414,392` tokens；它可以接纳一条接近 1M 的序列，但不可能同时容纳 6 条 1M 序列。

| 值 | 效果 |
|------|------|
| 0.78（当前） | 上限 ~100 GB，实际 79 GB，安全 |
| 0.75 | 最低可用值，再低 vLLM 可能拒绝启动 |
| 0.85 | 更激进，极限并发可能 OOM |

### 并发控制：`--max-num-seqs 6`

最多保持 6 条活跃序列，并配合 `--no-async-scheduling`。实测已通过 6 路、每路 `79,134` prompt tokens 的共享前缀请求。这是调度上限，不是 6 × 1M 容量：

| 值 | 适用场景 | 显存压力 |
|------|----------|---------|
| 1 | 隔离/诊断并发相关问题 | 最低调度风险 |
| 2–5 | 保守的共享服务 | 应使用真实会话形状压测 |
| 6（当前） | 多用户服务 | 已验证 6 × 79,134-token 共享前缀请求 |
| 7+ | 吞吐实验 | 本配置尚未验证 |

### 完整参数速查

```bash
--tensor-parallel-size 2          # TP=2 跨 2 张 GPU（两节点各一）
--pipeline-parallel-size 1        # PP=1
--nnodes 2                        # 2 个节点
--kv-cache-dtype fp8              # 当前经验证的 KV cache 类型
--block-size 256                  # KV Cache block 大小
--max-model-len 1048576           # 1M token 上下文（YaRN 原生）
--max-num-seqs 6                  # 最多 6 条活跃序列，不等于 6 × 1M
--max-num-batched-tokens 8192     # prefill 批次（小块省显存）
--gpu-memory-utilization 0.78     # 显存安全上限（~79GB / 128GB 统一内存）
--enable-chunked-prefill          # 长 prompt 分块防 OOM
--enable-prefix-caching           # 共享前缀复用 KV Cache
--no-async-scheduling             # 关闭异步调度，保留 6 序列上限
--speculative-config dspark       # DGX Spark 硬件推测解码
--moe-backend flashinfer_b12x     # B300 GB10 专用 MoE
```

---

## NCCL 配置详解

双机推理依赖 **NCCL** 通过 RoCE 实现跨节点 GPU 通信。

### 为什么必须用 `--network host`

```
容器网络栈  ──NCCL RDMA──►  物理网卡 (enp1s0f0np0)
     ↑                            ↑
  docker bridge 模式时            RoCE 必须直通
  RDMA 无法穿透 NAT              物理 HCA 设备
```

NCCL RDMA 数据路径必须直通物理网卡，Docker bridge/NAT 虚拟 IP 会导致 `mlx5` HCA 绑定失败。`--network host` 是唯一可靠方式。

### NCCL 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `NCCL_IB_HCA` | `=rocep1s0f0:1,roceP2p1s0f0:1` | exact-match 指定双 HCA 路径 |
| `NCCL_IB_MERGE_NICS` | `1` | 合并两条 RoCE 路径 |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET` | 只选 IPv4 GID |
| `NCCL_IB_ADDR_RANGE` | `10.10.12.0/23` | 将动态 GID 选择限定在示例双子网 |
| `NCCL_NET` | `IB` | 强制 IB/RoCE 传输层 |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | RoCE v2 (UDP 封装) |
| `NCCL_CROSS_NIC` | `1` | 允许跨 NIC 通信 |
| `NCCL_CUMEM_ENABLE` | `0` | 禁用 CUDA 内存池直通 (GB10 兼容) |
| `NCCL_NVLS_ENABLE` | `0` | 禁用 NVLink Sharp (GB10 不支持) |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | 忽略 CPU 亲和性 (容器内必需) |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | NCCL 通信绑定网卡 |
| `GLOO_SOCKET_IFNAME` | `enp1s0f0np0` | Gloo 通信网卡 |
| `TP_SOCKET_IFNAME` | `enp1s0f0np0` | TP 通信网卡 |

### 动态 GID 选择

脚本故意不设置 `NCCL_IB_GID_INDEX`。MTU 不能决定 GID 在每台机表中的位置；NCCL 根据 `AF_INET` 和 `NCCL_IB_ADDR_RANGE` 动态选择 RoCE v2 GID，允许两台节点使用不同的本地索引。

### NCCL 初始化流程

```
1. Worker 先启动，等待 Rank 0
2. Head 后启动，两个节点协同加载权重
3. Head 完成 → NCCL init 握手 (Head ↔ Worker)
   ├─ TCP 握手 (MASTER_ADDR:MASTER_PORT)
   ├─ GLOO 拓扑发现 (GLOO_SOCKET_IFNAME)
   └─ NCCL IB 建连（双 HCA + 动态 GID）
4. 两节点各分配 NCCL rank → 同步完成 → 开始推理
```

> ⚠️ **关键**：Head `NODE_RANK=0`，Worker `NODE_RANK=1`，必须唯一。

### NCCL 排查

```bash
# 查看 RoCE 设备
ibv_devinfo
ibv_devinfo -d rocep1s0f0 -v | grep GID

# 查看 GID 表
show_gids | grep -A3 "rocep1s0f0"

# 开启详细 NCCL 日志
docker run ... -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=NET,INIT ...
```

### 常见 NCCL 错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Got completion with error` | RoCE 不通/网线松 | 检查物理连接 + 双向 ping |
| `GID index not found` | 仍残留固定 GID 索引 | 删除 `NCCL_IB_GID_INDEX`，检查 AF/地址范围与网卡 IP |
| `NCCL timeout` | Worker NODE_RANK=0 | Worker 改为 `--node-rank 1` |
| `Socket connection refused` | Head 未就绪 Worker 先连 | 等 Head 加载完再启 Worker |
| `mlx5_0:1 got error from peer` | 单向通另一向不通 | 防火墙检查，双向 ping |

---

## 模型权重获取

### 方式一：HuggingFace（推荐）

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

> 这里使用官方 Flash 0731 混合精度权重，总大小约 **156 GB**，建议预留 180–200 GB。`--kv-cache-dtype fp8` 描述的是运行时 KV cache，不是对权重的二次量化。两台节点都需下载。

### 方式二：ModelScope（国内镜像，更快）

```bash
pip install modelscope
modelscope download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

### 同步到 Worker

```bash
# 从 Head 通过 RoCE 直传
rsync -avz --progress \
  /data/models/deepseek-ai/DeepSeek-V4-Flash-0731/ \
  worker-user@10.10.12.21:/data/models/deepseek-ai/DeepSeek-V4-Flash-0731/
```

---

## 作者

[@alexlu0912_admin](https://gitee.com/alexlu0912_admin)

---

## 许可证

MIT
