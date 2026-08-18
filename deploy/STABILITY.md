# Long-agent stability profile

This profile targets long, tool-heavy agent sessions on two DGX Spark nodes. It
keeps the 1M model limit, FP8 KV cache, prefix caching, chunked prefill, and
DSpark speculative decoding while removing two failure modes seen in long
OpenAI-compatible histories.

## Why short chats can pass while long agents fail

Some Anthropic/OpenAI gateways replay one logical assistant turn as adjacent
assistant records: visible text, reasoning, and tool calls may arrive as separate
items. The DeepSeek-V4 tokenizer in the pinned Anemll `0.1.1` image predates
[vLLM PR #50686](https://github.com/vllm-project/vllm/pull/50686). It renders
those records independently and can accumulate malformed reasoning/EOS boundaries
over many tool rounds. The eventual symptoms include repeated actions, leaked
protocol markup, and nonsensical output even though the KV cache is not full.

The thin image in `stable-runtime/` applies that single upstream tokenizer fix
to the pinned base image and validates it during the Docker build. It does not
change or requantize model weights.

The stability profile also uses:

```text
--max-num-seqs 6
--no-async-scheduling
--enable-prefix-caching
--enable-chunked-prefill
--kv-cache-dtype fp8
```

Synchronous scheduling is deliberate, but it is not the same as serial execution:
vLLM may keep up to six sequences active while avoiding its asynchronous scheduler.
This profile was validated with six concurrent requests of `79,134` prompt tokens
each. `MAX_NUM_SEQS=6` is a scheduling cap, not a promise that the KV pool can hold
six independent 1M-token contexts. Use `MAX_NUM_SEQS=1` as an isolation/diagnostic
fallback if a workload still exposes a concurrency-dependent failure.

## Build

Run on both nodes:

```bash
cd deploy
bash prepare.sh --image
```

Or build the overlay directly:

```bash
cd deploy/stable-runtime
BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 \
IMAGE=deepseek-v4-flash:0.1.1-stable-pr50686 \
./build.sh
```

Both nodes must use images built from the same base image and patch.

## Launch

Copy `config.local.example.sh` to the ignored `config.local.sh` on each node and
set machine-specific RoCE addresses/interfaces there. Start Worker first, then
Head:

```bash
# Worker
cd ~/dgxspark_deepseekv4flash0731/deploy
bash start-worker.sh

# Head
cd ~/dgxspark_deepseekv4flash0731/deploy
bash start-head.sh
```

Client traffic must use the Head management/LAN address. `HEAD_IP` in
`config.sh` is the RoCE data-plane address for distributed initialization, not a
universal client endpoint.

## Regression probe

The included probe covers Responses API, split assistant/tool history, serial
replay, and concurrent submissions:

```bash
cd deploy
./stability-probe.py \
  --base-url http://127.0.0.1:8888 \
  --rounds 24 \
  --repeat 3 \
  --concurrency 6
```

To exercise roughly the part of a 1M window where the original long-agent issue
was observed:

```bash
./stability-probe.py \
  --base-url http://127.0.0.1:8888 \
  --rounds 32 \
  --filler-repetitions 2500 \
  --repeat 2 \
  --concurrency 1 \
  --timeout 1800
```

One tested dual-Spark deployment produced `242,334` prompt tokens. The first
prefill took `181.673s`; an identical prefix completed in `5.914s`, confirming
that prefix caching remained active. This is a reproducibility data point, not a
universal benchmark.

With `MAX_NUM_SEQS=6` and synchronous scheduling, the same deployment also passed
six concurrent requests of `79,134` prompt tokens each. After one `46.649s`
uncached prefill, the six shared-prefix requests completed in `1.888–2.182s` and
all returned the exact expected marker. This validates that workload shape; it
does not establish six-way capacity at the full 1M-token limit.

The same hardware/configuration had separately completed a `999,860`-token FP8
needle request. That demonstrates capacity for one test prompt; it does not mean
all 1M workloads have equal quality or latency.

## Rollback

To run the unpatched base image without editing tracked files, set the same
overrides on Worker and Head and retain Worker-first startup order:

```bash
IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 \
MAX_NUM_SEQS=1 \
bash start-worker.sh

IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 \
MAX_NUM_SEQS=1 \
bash start-head.sh
```

The `MAX_NUM_SEQS=1` override above is a conservative diagnostic fallback. The
tracked production default remains `6` with `--no-async-scheduling`.

For production changes, preserve the previous stopped containers under explicit
rollback names until the new profile has completed a soak period.

## Public/private configuration boundary

Safe to commit:

- the exact upstream patch and its license notice;
- Dockerfile/build verification;
- example RoCE subnets and interface names;
- generic startup, prepare, and regression scripts;
- aggregate test results without host identity.

Keep out of the public repository:

- management/LAN/Tailscale addresses and SSH host aliases;
- usernames, home-directory layouts, and private repository paths;
- API keys, registry credentials, client configurations, and raw prompts/logs;
- machine-specific `config.local.sh`.
