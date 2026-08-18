# DeepSeek V4 Flash 双机部署包

两台 DGX Spark 通过双路 RoCE 互联运行 DeepSeek V4 Flash（1M context，TP=2 跨节点）。

> 默认使用 FP8 KV、`MAX_NUM_SEQS=6`、同步调度，以及同时修复长历史编码和 DSML 工具协议漂移的本地稳定薄层镜像。机器专属值写入被 Git 忽略的 `config.local.sh`。修复原理和回归方法见 [STABILITY.md](STABILITY.md)。

---

## 1. 硬件拓扑

```text
┌──────────────────────────┐       双路 RoCE / MTU 9000       ┌──────────────────────────┐
│ DGX Spark Head           │◄───────────────────────────────►│ DGX Spark Worker         │
│ 10.10.12.11（主）        │  enp1s0f0np0 / rocep1s0f0      │ 10.10.12.21（主）        │
│ 10.10.13.11（第二路）    │  enP2p1s0f0np0 / roceP2p1s0f0  │ 10.10.13.21（第二路）    │
└──────────────────────────┘                                └──────────────────────────┘
```

- **TP=2 跨两个节点的 2 张 GPU**，PP=1，NNODES=2
- **网络**：两条 RoCE v2 数据路径，动态选择 GID，不固定 `NCCL_IB_GID_INDEX`

---

## 2. 你需要预先准备好的

| 项目 | 说明 |
|------|------|
| **基础镜像** | `ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1`；`prepare.sh --image` 会构建稳定薄层 |
| **模型权重** | `DeepSeek-V4-Flash-0731` 目录，两台节点放在**同一路径** (获取方式见下方) |

---

## 3. 部署前准备

### 3.0 一键环境准备 (推荐)

```bash
# 必须直接以普通用户运行；不要写成 sudo bash prepare.sh
bash prepare.sh

# 也可以先执行只读检查
bash prepare.sh --check

# 非交互：拉取基础镜像并构建稳定薄层
bash prepare.sh --image
```

交互式菜单，按需选择执行：

```
0) 只读环境检查
1) 准备稳定 Docker 镜像（普通用户）
2) 下载/续传模型（普通用户、项目 .venv）
3) 持久化双路 RoCE（仅此步骤调用 sudo）
9) 一键全部执行（1 → 3 → 2）
```

权限边界：

- Docker 操作、Python 虚拟环境和 ModelScope 下载均使用当前普通用户；
- 默认模型目录仍是 `/data/models/deepseek-ai/DeepSeek-V4-Flash-0731`。如果普通用户不能创建它，脚本只用 `sudo` 创建这个精确目录，并将该目录交给当前用户，不会递归修改整个 `/data`；
- 只有 RoCE 的 NetworkManager/IP/MTU 配置需要管理员权限；
- 如果误用 `sudo bash prepare.sh`，脚本会立即拒绝执行，避免产生 root 所有的模型、缓存和虚拟环境；
- 已有的部分模型不会被删除，ModelScope 会续传；完整性检查以权重索引和全部分片为准。

两台 DGX Spark 都需要运行 `prepare.sh`，Head 和 Worker 各执行一次。也可非交互执行：

```bash
bash prepare.sh --all head      # Head
bash prepare.sh --all worker    # Worker
```

### 3.1 手动方式 (备选)

两台节点分别执行：

```bash
# 当前用户必须能访问 Docker；若不能，先执行下列命令并重新登录
sudo usermod -aG docker "$USER"

docker pull ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1
BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 \
  IMAGE=deepseek-v4-flash:0.1.1-stable-20260818 \
  ./stable-runtime/build.sh
```

### 3.2 配置 RoCE IP

推荐让脚本通过 NetworkManager 创建 `roce-primary` 和 `roce-secondary` 两个持久化连接：

```bash
bash prepare.sh --roce head      # Head 上执行
bash prepare.sh --roce worker    # Worker 上执行
```

这一步会针对两张 RoCE 逻辑网卡逐条调用 `sudo nmcli`，配置 `/24` 地址、MTU 9000、`never-default` 和开机自动连接。若 NetworkManager 不可用，脚本会退回临时 `ip` 配置并明确警告重启后会丢失。

### 3.3 修改配置文件

复制本地覆盖模板（`config.local.sh` 已被 Git 忽略），再按本机修改：

```bash
cp config.local.example.sh config.local.sh
```

不要把管理网地址、SSH 用户、私有镜像凭据或个人目录写入已跟踪的 `config.sh`。公开默认值位于 `config.sh`：

```bash
IMAGE="deepseek-v4-flash:0.1.1-stable-20260818"
MODEL_PATH="/data/models/deepseek-ai/DeepSeek-V4-Flash-0731"
HEAD_IP="10.10.12.11"
WORKER_IP="10.10.12.21"
HEAD_IP_SECONDARY="10.10.13.11"
WORKER_IP_SECONDARY="10.10.13.21"
NCCL_INTF="enp1s0f0np0"
NCCL_INTF_SECONDARY="enP2p1s0f0np0"
```

其余参数无需修改。

### 3.4 同步部署包到两台节点

```bash
# 将整个 deploy 目录复制到 Worker
scp -r ./ worker.example:~/dgxspark_deepseekv4flash0731/deploy/
```

---

## 4. 部署步骤

### Step 1 — 启动 Worker

在 **Worker 节点** 上**先启动**：

```bash
cd ~/dgxspark_deepseekv4flash0731/deploy
bash start-worker.sh
```

### Step 2 — 启动 Head

Worker 就绪后，在 **Head 节点** 上启动：

```bash
cd ~/dgxspark_deepseekv4flash0731/deploy
bash start-head.sh
```

> 💡 监控加载进度：`docker logs -f vllm_anemll | grep -E "Ulysses|Loading model|E2E|est.|GPU"`

### Step 3 — 验证部署

#### ① 确认服务就绪

```bash
# 监控日志，出现 "Ulysses model is ready" 即就绪
docker logs -f vllm_anemll 2>&1 | grep -E "Ulysses|Loading model|E2E|est."
```

#### ② 模型列表

```bash
API_HOST=head-management-hostname
curl -s http://${API_HOST}:8888/v1/models | python3 -m json.tool
```

#### ③ 基础对话

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

#### ④ 流式输出

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

#### ⑤ 吞吐基准

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

#### ⑥ 验证 NCCL 跨节点通信

```bash
docker logs vllm_anemll 2>&1 | grep -E "NCCL.*rank.*nranks"
# 预期: rank 0 nranks 2 — 确认双节点已互联
```

#### 验证失败速查

| 现象 | 排查 |
|------|------|
| Connection refused | Head 未加载完，等 5-10 分钟 |
| model not found | 模型名是 `deepseek-v4-flash` |
| 速度 < 10 tok/s | Worker 离线，`docker logs` 查 Worker |
| 乱码/空白 | tokenizer 不正确 |

---

## 5. 文件清单

```
deploy/
├── config.sh                    # 可公开默认值
├── config.local.example.sh      # 机器专属覆盖模板
├── stable-runtime/              # PR #50686 可复现薄层镜像
├── stability-probe.py           # 多轮/长上下文回归探针
├── STABILITY.md                 # 根因、验证与回滚
├── prepare.sh         # 环境准备脚本 (交互式菜单)
├── start-head.sh      # Head 节点一键启动
├── start-worker.sh    # Worker 节点一键启动
└── README.md          # 本文件
```

---

## 6. 常用操作

```bash
# 查看日志
docker logs -f vllm_anemll

# 停止服务
docker stop vllm_anemll && docker rm vllm_anemll

# 进入容器调试
docker exec -it vllm_anemll bash

# 查看 GPU
docker exec vllm_anemll nvidia-smi
```

---

## 7. 关键参数

### 上下文长度 — 1M token 原生支持

模型 `config.json` 中 `max_position_embeddings: 1048576`，通过 **YaRN**（factor=16）从 65536 扩展到 1M，无需滑动窗口或外推技巧。对应启动参数 `--max-model-len 1048576`。

### 显存管理 — 安全上限 0.78，实际仅 79 GB

`--gpu-memory-utilization 0.78` 是 vLLM 建立执行与 KV cache 预算时使用的上限，不等于容器 RSS，也不能简单用“总内存减权重”推导安全并发。统一内存、CUDA graph、JIT workspace 和系统缓存都会影响启动余量；当前默认值已经完成接近 1M 的单请求验证。

| 值 | 效果 |
|------|------|
| 0.78（默认） | 已验证的稳定基线 |
| 0.75 | KV pool 更小，适合排查内存压力 |
| 0.80+ | 需要重新做冷启动、长上下文和并发压测 |

### 并发 — `max-num-seqs=6` + 同步调度

当前默认允许最多 6 条活跃序列，但关闭 vLLM 异步调度，以兼顾多用户吞吐和长 Agent 会话稳定性。双 Spark 实测已通过 6 路、每路 `79,134` prompt tokens 的共享前缀压测。`6` 只是调度上限，不代表 KV pool 能同时容纳 6 条 1M 序列；长下文总容量仍受 KV cache 预算约束。排查时可在 `config.local.sh` 临时设 `MAX_NUM_SEQS=1`。

### 完整参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-model-len` | `1048576` | 1M token 上下文（YaRN 原生） |
| `--gpu-memory-utilization` | `0.78` | 显存池上限（实际模型占用约 79GB / 128GB 统一内存） |
| `--max-num-seqs` | `6` | 最多 6 条活跃序列；不等于 6 × 1M KV 容量 |
| `--max-num-batched-tokens` | `8192` | prefill 小块处理（省显存） |
| `--kv-cache-dtype` | `fp8` | 当前生产配置；已验证接近 1M token |
| `--block-size` | `256` | KV Cache block 大小 |
| `--enable-chunked-prefill` | ✅ | 长 prompt 分块防 OOM |
| `--enable-prefix-caching` | ✅ | 共享前缀复用 |
| `--no-async-scheduling` | ✅ | 稳定优先，避免长前缀竞争 |
| `--speculative-config` | `dspark` | DGX Spark 硬件推测解码 |
| `--moe-backend` | `flashinfer_b12x` | B300 GB10 专用 MoE |
| `--headless` | Worker 专用 | Worker 无 API 服务 |

---

## 8. NCCL 配置

双节点推理依赖 NCCL 通过 RoCE 跨节点通信。脚本自动处理以下配置，无需手动干预。

### 动态选择 GID

脚本不设置 `NCCL_IB_GID_INDEX`。MTU 与 GID 表没有固定映射，两台机器强制相同索引曾导致 NCCL communicator 初始化失败；现在由 NCCL 根据 IPv4 地址范围动态选择 RoCE v2 GID。

### 核心 NCCL 环境变量

| 变量 | 值 | 作用 |
|------|-----|------|
| `NCCL_IB_HCA` | `=rocep1s0f0:1,roceP2p1s0f0:1` | exact-match 指定双 HCA 路径 |
| `NCCL_IB_MERGE_NICS` | `1` | 合并两条 RoCE 路径 |
| `NCCL_IB_ADDR_RANGE` | `10.10.12.0/23` | 允许两个 RoCE 子网 |
| `NCCL_NET` | `IB` | 强制 IB/RoCE 传输 |
| `NCCL_CROSS_NIC` | `1` | 允许跨 NIC |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | 容器内必需 |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | 通信绑定网卡 |

### 为什么必须 `--network host`

NCCL RDMA 数据路径 → 物理 HCA (`mlx5`) → 物理网卡。Docker bridge/NAT 模式会创建虚拟 IP，导致 RDMA 绑定失败。

---

## 9. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 模型加载卡住不动 | 等待 Worker 加入 | 正常现象 — 确认 Worker 已启动 |
| Worker 报 NCCL timeout | RoCE 不通 | 检查网线、`ping <head_ip>` |
| NCCL GID/communicator 初始化失败 | 固定 GID 索引与节点实际 GID 表不一致 | 删除 `NCCL_IB_GID_INDEX`，保留动态选择 |
| OOM | 显存不够 | 降低 `GPU_MEM` 或 `MAX_NUM_SEQS` |
| Container 启动即退出 | 镜像/模型路径不对 | `docker logs vllm_anemll` 查看错误 |

---

## 10. 模型权重获取

```bash
# 推荐：由 prepare.sh 创建项目 .venv、处理目录权限并验证 48 个权重分片
bash prepare.sh --model

# 对应的 ModelScope 新版 CLI 语法（不再使用旧的 --model 参数）
./.venv/bin/modelscope download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

> 官方 Flash 模型权重实际约 **156 GB**，建议预留 180–200 GB 磁盘。两台节点都需下载。

---

## 作者

[@alexlu0912_admin](https://gitee.com/alexlu0912_admin)
