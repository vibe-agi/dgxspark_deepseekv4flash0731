# vLLM 参数详解 — DeepSeek V4 Flash 双机部署

## 模型与分布式参数

### `--tensor-parallel-size 2`

张量并行度。DeepSeek V4 Flash 0731 官方 Flash 权重约 156 GB，单台 DGX Spark GB10 的 128 GB 统一内存无法稳定容纳权重、KV cache 和运行时开销，因此两台以 TP=2 协作。

### `--pipeline-parallel-size 1` / `--nnodes 2`

不切 pipeline stage；Head 和 Worker 两个节点各持有一个 TP rank。Head 必须是 `--node-rank 0`，Worker 必须是 `--node-rank 1`。

## KV cache 与长上下文

### `--kv-cache-dtype fp8`

当前生产基线使用 FP8 KV cache，已完成接近 1M token 的单请求验证。这个参数仅描述运行时 KV cache，不会将模型权重改成 NVFP4。

### `--block-size 256`

KV cache block 大小。当前镜像与 1M profile 的已验证值，改动后要重做冷启动和长上下文测试。

### `--max-model-len 1048576`

单条序列的配置上限为 1M tokens。这不代表多条序列可以同时各占 1M；实际同时容量取决于启动日志中的 GPU KV cache token 数。

### `--enable-chunked-prefill`

将长 prompt 分块 prefill，避免一次将整个 prompt 塞入调度批次。与 `--max-num-batched-tokens 8192` 配合使用。

### `--enable-prefix-caching`

共享会话前缀可复用 KV cache。它可以大幅降低重复 Agent 历史的 prefill 时间，但不能消除首次 prefill 或独立长会话的 KV 容量需求。

## 显存与并发

### `--gpu-memory-utilization 0.78`

vLLM 可用统一内存的预算上限。它不是进程 RSS，也不能用“128 GB 减去权重”直接推导并发容量。CUDA Graph、JIT workspace、NCCL buffer 和系统缓存也使用同一内存池。

### `--max-num-seqs 6`

最多保持 6 条活跃序列。当前配置已通过 6 路、每路 `79,134` prompt tokens 的共享前缀压测。这是调度上限，不是 6 × 1M 的 KV 容量承诺。遇到疑似并发竞争时，可临时用 `MAX_NUM_SEQS=1` 做隔离诊断。

### `--no-async-scheduling`

关闭 vLLM 的异步调度器，但不会将 `max-num-seqs=6` 改成串行。当前将“6 条活跃序列 + 同步调度”作为长 Agent 会话的稳定基线。

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
| 启动 OOM | 先看是否有其他 GPU/统一内存进程，再将 `GPU_MEM` 从 0.78 降到 0.75 |
| NCCL 初始化失败 | 检查双向 IP/MTU/HCA，清除固定 `NCCL_IB_GID_INDEX` |
| 长 Agent 输出协议标签 | 确认稳定镜像包含 vLLM PR #50686，运行 `stability-probe.py` |
| 疑似并发竞争 | 临时设 `MAX_NUM_SEQS=1`；若消失，按 2→4→6 重放真实会话 |
| 需要更高吞吐 | 先用 6 路长历史建立基线，再逐级增加；不根据权重占用推导“安全并发” |
