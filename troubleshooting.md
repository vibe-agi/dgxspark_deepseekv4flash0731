# 故障排查与部署踩坑记录

本文档记录 2026-08-04 起两台 DGX Spark 双机部署 DeepSeek V4 Flash 全过程中遇到的**所有问题、根因与解决方案**。

---

## 目录

1. [NCCL 通信故障](#1-nccl-通信故障)
2. [容器 GPU 不可见](#2-容器-gpu-不可见)
3. [Worker 启动后自动退出](#3-worker-启动后自动退出)
4. [每次重启容器加载超慢](#4-每次重启容器加载超慢)
5. [端口映射不生效](#5-端口映射不生效)
6. [RoCE IP 重启后丢失](#6-roce-ip-重启后丢失)
7. [容器卡在 NCCL init / 死锁](#7-容器卡在-nccl-init--死锁)
8. [启动顺序问题](#8-启动顺序问题)
9. [Worker 必须从 docker run 启动](#9-worker-必须从-docker-run-启动)
10. [长 Agent 会话逐渐乱码](#10-长-agent-会话逐渐乱码)

---

## 1. NCCL 通信故障

### 现象
```
NCCL WARN NET/IB : Got completion with error
NCCL WARN NET/IB : mlx5_0:1 got error from peer
```

### 根因
1. **RoCE IP 未配** — `enp1s0f0np0` 接口无 IP 地址
2. **MTU 不匹配** — 默认 1500，RoCE 需要 9000
3. **固定 GID 索引错误** — 两台机的 GID 表位置不一致，同一个索引在某个节点不是所需的 IPv4/RoCE-v2 GID

### 解决
```bash
# 两节点都执行
ip addr add 10.10.12.11/24 dev enp1s0f0np0   # Head
ip addr add 10.10.12.21/24 dev enp1s0f0np0   # Worker
ip link set enp1s0f0np0 mtu 9000

# 验证
ibv_devinfo | grep -A5 mlx5_0
ping 10.10.12.21   # 从 Head ping Worker
```

生产脚本不再设置 `NCCL_IB_GID_INDEX`，而是使用 `NCCL_IB_ADDR_FAMILY=AF_INET` 和数据面地址范围让 NCCL 动态选择。MTU 应在两端保持一致；它不能用来推导 GID 表索引。

---

## 2. 容器 GPU 不可见

### 现象
```
RuntimeError: Failed to infer device type
```

### 根因
`docker run` 忘了 `--gpus all`

### 解决
确保 docker run 命令包含：
```bash
docker run ... --gpus all ...
```

---

## 3. Worker 启动后自动退出

### 现象
Worker 容器启动 5 分钟后自动退出，日志显示 NCCL 超时。

### 根因
**`NODE_RANK` 冲突**。Head 和 Worker 的 `--node-rank` 都设成了 0。
vLLM 检测到两个节点都声称自已是 rank 0，NCCL 无法建立连接。

### 解决
```bash
# Head
--node-rank 0

# Worker (必须不同于 Head!)
--node-rank 1
```

**教训**: 双机部署时，`NODE_RANK` 必须从 0 开始，每个节点唯一。

---

## 4. 每次重启容器加载超慢

### 现象
容器第一次启动模型加载约 3 分钟，但重启后需要 10-15 分钟。

### 根因
FlashInfer 的 autotune 结果（JIT 编译缓存）存储在 `~/.cache/flashinfer/`，
容器内路径是 `/root/.cache/flashinfer/`。没挂载 host 目录时，每次重建容器都重新编译。

### 解决
```bash
docker run ... \
  -v /cache/huggingface:/cache/huggingface \
  -v /root/.cache/flashinfer:/root/.cache/flashinfer \
  ...
```

---

## 5. 端口映射不生效

### 现象
```bash
docker run -p 8080:8080 --network host ...
```
Docker 提示 `-p` 和 `--network host` 冲突，端口无法访问。

### 根因
Docker 在 host 网络模式下**完全忽略** `-p`。容器直接使用 host 网络栈。

### 解决

**方案 A**: 直接用 host 端口（推荐）
容器内 vLLM 监听 8080，host 上直接 `curl localhost:8080`

**方案 B**: 用 iptables 转发到特定容器（如果不用 host 网络）
```bash
# 找到容器 IP
CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id>)

# 转发
iptables -t nat -A PREROUTING -p tcp --dport 4108 -j DNAT --to-destination $CONTAINER_IP:8080
iptables -t nat -A POSTROUTING -p tcp -d $CONTAINER_IP --dport 8080 -j MASQUERADE
```

**教训**: 双机部署场景，`--network host` 是最简单可靠的选择（NCCL RDMA 数据路径必须直通物理网卡）。

---

## 6. RoCE IP 重启后丢失

### 现象
`ip addr add` 配置的 IP 在重启后丢失。

### 根因
NetworkManager 覆盖了手动配置。

### 解决

让部署脚本通过 NetworkManager 创建带 IP、MTU 9000、`never-default` 和开机自启的持久连接：

```bash
bash prepare.sh --roce head      # Head
bash prepare.sh --roce worker    # Worker
```

---

## 7. 容器卡在 NCCL init / 死锁

### 现象
两个节点都启动后，一个或两个卡在 `NCCL INFO Channel 00 : ...` 不动。

### 可能原因
1. **端口被防火墙阻挡** — NCCL 使用的随机端口被 iptables 阻挡
2. **双向不通** — 只有单向 ping 通（比如从 Head 能 ping Worker，反向不行）
3. **IB 设备名不一致** — 一个节点是 `mlx5_0`，另一个不同

### 排查
```bash
# 双向 ping 测试
ping 10.10.12.21   # Head 上
ping 10.10.12.11   # Worker 上

# IB 设备检查
ibv_devinfo
```

---

## 8. 启动顺序问题

### 推荐顺序
1. **Worker 先启** — `bash start-worker.sh`
2. **Head 后跟** — `bash start-head.sh`

### 原因
vLLM 的 NCCL init 只在 Head 连接时触发，Worker 先就位等待 Head 连接更稳定。
Worker 先启动约 10-30 秒后 Head 再启动即可。

---

## 9. Worker 必须从 docker run 启动

### 现象
用 `docker exec` 在 Worker 已有容器内运行 vLLM 时提示 GPU 被占用。

### 根因
每个 docker run 创建一个隔离的 GPU 上下文。`docker exec` 在已有容器内无法独立获取 GPU。

### 解决
Worker 必须独立 `docker run` 一个新的 vLLM 容器，不能用 `docker exec` 或 docker-compose 的 `exec`。

---

## 10. 长 Agent 会话逐渐乱码

### 现象

短对话正常，但经过多轮 reasoning/tool call 之后出现重复操作、虚构的 system reminder，或泄露 `<antml:...>`、`<｜DSML｜...>` 等协议标签。问题往往在客户端显示上下文只使用 20%–30% 时就出现。

### 根因

Agent 网关可能将一个 assistant turn 重放为相邻的文本、reasoning 和 tool-call 记录。固定 Anemll `0.1.1` 镜像中的 DeepSeek-V4 tokenizer 早于 vLLM PR #50686，会分别渲染这些相邻 assistant 记录，长时间积累后产生错误 reasoning/EOS 边界。这不是“1M KV cache 已经用完”。

### 解决

1. 在两台节点构建并使用 `stable-runtime/` 薄层镜像；它只应用 PR #50686 tokenizer 补丁，不修改模型权重。
2. 使用 FP8 KV、prefix caching、chunked prefill、`MAX_NUM_SEQS=6` 和 `--no-async-scheduling`。
3. 运行协议回归：

```bash
./stability-probe.py \
  --base-url http://127.0.0.1:8888 \
  --rounds 32 \
  --filler-repetitions 800 \
  --repeat 1 \
  --concurrency 6 \
  --timeout 1800
```

已验证的双 Spark 基线中，6 路请求每路为 `79,134` prompt tokens，全部返回精确标记，无协议标签泄露。若只有真实业务会话仍失败，将该请求形状脱敏后加入回归；不要仅在网关层逐个替换特殊标签。
