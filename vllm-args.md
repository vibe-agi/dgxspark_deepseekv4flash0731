# vLLM 参数详解 — DeepSeek V4 Flash 双机部署

## 模型与分布式参数

### `--tensor-parallel-size 2`

张量并行度。DeepSeek V4 Flash 0731 官方 Flash 权重约 156 GB，单台 DGX Spark GB10 的 128 GB 统一内存无法稳定容纳权重、KV cache 和运行时开销，因此两台以 TP=2 协作。

### `--pipeline-parallel-size 1` / `--nnodes 2`

不切 pipeline stage；Head 和 Worker 两个节点各持有一个 TP rank。Head 必须是 `--node-rank 0`，Worker 必须是 `--node-rank 1`。

## KV cache 与长上下文

### `--kv-cache-dtype nvfp4_ds_mla`

当前生产基线使用 padded NVFP4 DS-MLA KV cache，并通过稳定镜像中的 Issue #22 补丁路由到已验证的快速内核。这里的布局仍为 584 bytes，与 `fp8_ds_mla` 的 padded 布局相同，不等于 KV 占用减半，也不会改变或二次量化模型权重。

### `--block-size 256`

KV cache block 大小。当前镜像与 1M profile 的已验证值，改动后要重做冷启动和长上下文测试。

### `--max-model-len 1048576`

单条序列的配置上限为 1M tokens。这不代表多条序列可以同时各占 1M；实际同时容量取决于启动日志中的 GPU KV cache token 数。

### `--enable-chunked-prefill`

将长 prompt 分块 prefill，避免一次将整个 prompt 塞入调度批次。当前单用户满窗档与 `--max-num-batched-tokens 16384` 配合使用；共享服务回退档为 8192。

### `--long-prefill-token-threshold 0`

`0` 允许一条长 prefill 吃满 batch，适合当前只有一条活跃请求的满窗场景。88K 冷 Prefill 相比 threshold 1024 从 52.635s 降到 45.100s。一个人同时运行多个 subagent 也属于并发；此时应回退到 `1024`，避免一条 prefill 让其他请求的 decode 饥饿。

### `--enable-prefix-caching`

共享会话前缀可复用 KV cache。它可以大幅降低重复 Agent 历史的 prefill 时间，但不能消除首次 prefill 或独立长会话的 KV 容量需求。

## 显存与并发

### `--gpu-memory-utilization 0.835`

vLLM 可用统一内存的预算上限。它不是进程 RSS，也不能用“128 GB 减去权重”直接推导并发容量。CUDA Graph、JIT workspace、NCCL buffer 和系统缓存也使用同一内存池。16K batch 在 0.78 下只剩 8.22 GiB KV，低于 1M 所需的 10.91 GiB；0.835 下可用 KV 为 15.33 GiB，并已完成 `1,039,984`-token 满窗请求。

### `--max-num-seqs 6`

最多保持 6 条活跃序列。当前配置已通过 6 路、每路 `79,134` prompt tokens 的共享前缀压测。这是调度上限，不是 6 × 1M 的 KV 容量承诺。遇到疑似并发竞争时，可临时用 `MAX_NUM_SEQS=1` 做隔离诊断。

### `--no-async-scheduling`

关闭 vLLM 的异步调度器，但不会将 `max-num-seqs=6` 改成串行。当前将“6 条活跃序列 + 同步调度”作为长 Agent 会话的稳定基线。

### `VLLM_USE_BREAKABLE_CUDAGRAPH=0`

明确使用 regular CUDA Graph。Anemll 自动启用的 breakable graph 在此双机实测更慢；`MAX_CUDAGRAPH` 默认按 `MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1)` 推导，6×5 时为 36。

## DGX Spark 专有参数

### `--speculative-config dspark`

DGX Spark 定制的 DSpark 推测解码配置。当前 `num_speculative_tokens=5`。

### `--moe-backend flashinfer_b12x`

GB10/SM120 上的 FlashInfer MoE 后端。相关 autotune 缓存应持久化，避免每次重建容器都重新预热。

## NCCL / RoCE

`VLLM_HOST_IP`、Gloo 和 TP socket 绑定主 RoCE IP/网卡。NCCL 则 exact-match 两条 HCA 路径，并通过 IPv4 地址范围动态选 GID：

```bash
NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1'
NCCL_IB_MERGE_NICS=1
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_IB_ADDR_RANGE=10.10.12.0/23
NCCL_NET=IB
NCCL_IB_ROCE_VERSION_NUM=2
NCCL_CROSS_NIC=1
NCCL_SOCKET_IFNAME=enp1s0f0np0
```

不要根据 MTU 猜测并固定 `NCCL_IB_GID_INDEX`；两台机的 GID 表索引可以不同。

## 调优建议

| 问题 | 调整顺序 |
|------|----------|
| 16K batch 启动时报 1M KV 不足 | 保持 `GPU_MEM=0.835`；或整体回退为 `0.78 + batch 8192 + threshold 1024` |
| 启动 OOM | 先看是否有其他 GPU/统一内存进程；需要降内存时同时缩小 batch，并重做 1M gate |
| NCCL 初始化失败 | 检查双向 IP/MTU/HCA，清除固定 `NCCL_IB_GID_INDEX` |
| 长 Agent 输出协议标签 | 确认稳定镜像包含 vLLM PR #50686，运行 `stability-probe.py` |
| 疑似并发竞争 | 先设 `LONG_PREFILL_TOKEN_THRESHOLD=1024`，必要时再临时设 `MAX_NUM_SEQS=1`；若消失，按 2→4→6 重放真实会话 |
| 需要更高吞吐 | 先用 6 路长历史建立基线，再逐级增加；不根据权重占用推导“安全并发” |
