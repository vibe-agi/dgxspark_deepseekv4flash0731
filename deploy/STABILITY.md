# Long-agent stability profile

This profile targets long, tool-heavy agent sessions on two DGX Spark nodes. It
keeps the 1M model limit, padded NVFP4 DS-MLA KV cache, prefix caching, chunked
prefill, and DSpark speculative decoding while removing the tokenizer, parser,
streaming, and cold-JIT failure modes reproduced in long native-Anthropic agent
histories.

## Why short chats can pass while long agents fail

Some Anthropic/OpenAI gateways replay one logical assistant turn as adjacent
assistant records: visible text, reasoning, and tool calls may arrive as separate
items. The DeepSeek-V4 tokenizer in the pinned Anemll `0.1.1` image predates
[vLLM PR #50686](https://github.com/vllm-project/vllm/pull/50686). It renders
those records independently and can accumulate malformed reasoning/EOS boundaries
over many tool rounds. The eventual symptoms include repeated actions, leaked
protocol markup, and nonsensical output even though the KV cache is not full.

Empty historical reasoning introduces another failure: tool-enabled prompt
encoding can turn an assistant record with no replayable reasoning into repeated
`<think></think>` blocks. At long context, those blocks bias the next turn toward
an immediate thinking close and little or no useful reasoning.

There is also an independent producer-side failure. DeepSeek V4 normally emits
tool calls with full-width `<｜DSML｜...>` markers, but long agent runs can drift
to semantically equivalent ASCII `<|DSML|...>` markers or abbreviated closing
tags such as `</parameter>`, omit/corrupt the outer wrapper, or place a truncated
marker in the following tool name. The pinned parser recognizes only the ideal
form, so syntax can leak into text or the client receives an invalid tool name.
Large string arguments also used a repeated whole-buffer conversion path, which
can appear as “says it will call Bash, but never acts.”

The thin image in `stable-runtime/` pins the base digest and vLLM commit, applies
twelve exact (`--fuzz=0`) protocol/tokenizer/parser/sampler/mHC patches, and retains only the MiaAI
Issue #22 NVFP4 kernel-dispatch change from that external bundle. The parser
recovery path is provisional, lossless on failure, bounded, and allow-listed by
the tools declared in the current request. Build-time verification covers every
character boundary of canonical/recovered calls, contaminated/undeclared names,
1 MiB multiline/control-character arguments, mHC dispatch, empty thinking,
and the NVFP4 marker. See `stable-runtime/PATCH_REVIEW.md`. Model weights are not
changed or requantized.

The pinned native Anthropic endpoint also accepted but silently discarded the
top-level `thinking` request object. Passing the corresponding
`thinking_token_budget` farther into DSpark's mandatory Model Runner V2 instead
produced HTTP 500. Patch `0010` parses the Anthropic controls and preserves
explicit effort. Patch `0011` backports vLLM's merged V2 sampler implementation,
which tracks each request's reasoning state and enforces the exact numeric cap.
Patch `0012` counts reasoning that starts in the prompt template, fixing zero
`reasoning_tokens` usage on OpenAI Responses.

The tracked single-user/full-window profile also uses:

```text
--max-num-seqs 6
--no-async-scheduling
--enable-prefix-caching
--enable-chunked-prefill
--kv-cache-dtype nvfp4_ds_mla
--max-num-batched-tokens 16384
--long-prefill-token-threshold 0
--gpu-memory-utilization 0.835
```

Synchronous scheduling is deliberate, but it is not the same as serial execution:
vLLM may keep up to six sequences active while avoiding its asynchronous scheduler.
This profile was validated with six concurrent requests of `79,134` prompt tokens
each. `MAX_NUM_SEQS=6` is a scheduling cap, not a promise that the KV pool can hold
six independent 1M-token contexts. Use `MAX_NUM_SEQS=1` as an isolation/diagnostic
fallback if a workload still exposes a concurrency-dependent failure.

`LONG_PREFILL_TOKEN_THRESHOLD=0` optimizes one active cold prefill. A single
operator launching several subagents is still concurrent. For a shared service
or decode-fairness validation, set the same overrides on both nodes:

```text
GPU_MEM=0.78
MAX_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024
```

## Build

Run on both nodes:

```bash
cd deploy
bash prepare.sh --image
```

Or build the overlay directly:

```bash
cd deploy/stable-runtime
IMAGE=deepseek-v4-flash:0.1.9-stable-20260821 \
./build.sh
```

Both nodes must use images built from the same base image and patch.
The build script supplies the digest-pinned base by default; do not replace it
with the mutable `:0.1.1` tag in a reproducible build.

The runtime exposes native `/v1/messages` and `/v1/messages/count_tokens`.
Claude Code can connect directly to the Head management/LAN/VPN endpoint; an
Anthropic-to-OpenAI conversion proxy is not part of this stability path.

For thinking depth, prefer Anthropic `output_config.effort`, OpenAI Chat
`reasoning_effort`, or OpenAI Responses `reasoning.effort`. Anthropic
`budget_tokens` and the vLLM Chat extension `thinking_token_budget` are exact
reasoning cutoffs on this patched V2 profile. OpenAI Responses exposes effort,
not a standard reasoning-only numeric cap; its `max_output_tokens` covers both
reasoning and visible output.

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

## 2026-08-19 NVFP4 isolation result

The final image was selected by changing one runtime layer at a time. A candidate
image containing the complete MiaAI hotfix set stalled on the third sequential
8K cold request and logged `No available shared memory broadcast block found in
60 seconds`. It is not the production profile. Removing the scheduler,
spin-wait, hybrid-cache and optional performance changes while retaining only
Issue #22 eliminated the stall.

Matched cold-prefix measurements on this two-node cluster were:

| Workload | Regular-Graph FP8 | Issue #22-only NVFP4 |
| --- | ---: | ---: |
| C1, 256-token prompt, decode median | 72.3 tok/s | 78.0 tok/s |
| C1, 8K prompt, decode median | 87.2 tok/s | 79.1 tok/s (10/10 completed) |
| C6, 256-token prompt, per-stream median | 46.1 tok/s | 43.2 tok/s |
| C6, 256-token prompt, warm aggregate | 193–203 tok/s | 190–198 tok/s |
| C6, 8K prompt, warm aggregate | 21.1 tok/s (one trial) | 29.1 tok/s |

The run-to-run spread is larger than the apparent C1 differences, so these data
do **not** justify a general “NVFP4 is faster” claim. They show comparable decode
throughput and, more importantly, stable completion under the tested workloads.
The padded `nvfp4_ds_mla` and `fp8_ds_mla` layouts also consumed essentially the
same KV memory at `GPU_MEM=0.78` (about 1.4M tokens in the startup log).

MTP-6 was also rejected: C1 p256 fell to `73.2 tok/s`, C6 p256 fell to
`41.9 tok/s`, and Graph capture grew from about `0.33 GiB` to `1.15 GiB`.
Its small p8192 gain (`81.9` vs `79.1`) did not offset those regressions. The
production default remains MTP-5 and requests a derived capture ceiling of 36.
The pinned vLLM normalizes that request to a largest full-graph shape of 32;
single-sequence MTP-5 decode remains covered, while the six-way lane requires a
separate `40`-ceiling memory/throughput A/B before changing the stable profile.

Long-context/protocol gates on the selected profile:

- 128K C1: 2/2 completed, `74.8s` TTFT, `80.2 tok/s` decode median;
- 900K C1: completed 128/128 output tokens, `931.0s` TTFT and `73.5 tok/s`
  decode; running/waiting/KV metrics returned to zero afterward;
- native Anthropic: 239,869 input tokens across 24 history rounds, including a
  historical tool result, returned the exact structured tool call in `171.4s`;
- no DSML/Anthropic XML leakage, shared-memory timeout, NCCL error, or leftover
  running/waiting request was observed after those gates.

## Regression probe

Use the native Anthropic gates for the patched protocol path:

```bash
python3 stable-runtime/probes/anthropic_long_agent.py \
  --base-url http://HEAD-CLIENT-IP:8888/v1 \
  --target-tokens 32000 \
  --effort max

python3 stable-runtime/probes/anthropic_compaction_round.py \
  --base-url http://HEAD-CLIENT-IP:8888/v1 \
  --target-tokens 1040000 \
  --turns 96 \
  --effort max
```

The second command verifies a full-context tool call, a client-owned compaction
summary with durable anchors, and a post-compaction tool call. It is not a claim
that vLLM itself decides when to compact.

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

After adding the tolerant DSML parser, the reproducible probe passed an
`34,011`-token split assistant/tool history and returned only the expected marker.
A separate Claude-compatible gateway check passed at `33,343` effective input
tokens in both streaming and non-streaming modes; both responses contained a
structured tool call and no DSML/XML text. These checks target the 20–30% client
meter range where protocol leakage was originally reported; they do not replace a
longer soak test.

The same hardware/configuration had separately completed a `999,860`-token FP8
needle request. That demonstrates capacity for one test prompt; it does not mean
all 1M workloads have equal quality or latency.

The final 2026-08-21 gate ran on the reviewed `0.1.9` overlay. A native
Anthropic request with `1,040,105` input tokens completed in `1,127.003s`
without OOM, queueing, or container restart. The compaction request then
replayed `1,040,118` input tokens, hit `1,039,872`, recomputed only 246
(`99.976349%` hit), preserved all four anchors, and completed in `11.293s`.
After the client replaced its transcript with the summary, a 658-token request
completed a valid structured tool call in `5.579s`. This directly validates
near-full-window capacity, prefix-cache reuse, client-side compaction, and the
post-compaction tool path for the selected NVFP4 profile.

Prefix caching does not reduce request-body upload. A stateless client still
retransmits its complete history; on a relayed overlay path, a 1.04 MB history
added tens of seconds outside the engine despite a greater than 99.9% KV hit.
Use a direct management/LAN or direct overlay path when interpreting client
latency.

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

- the exact tokenizer/parser patches and their license notice;
- Dockerfile/build verification;
- example RoCE subnets and interface names;
- generic startup, prepare, and regression scripts;
- aggregate test results without host identity.

Keep out of the public repository:

- management/LAN/Tailscale addresses and SSH host aliases;
- usernames, home-directory layouts, and private repository paths;
- API keys, registry credentials, client configurations, and raw prompts/logs;
- machine-specific `config.local.sh`.
