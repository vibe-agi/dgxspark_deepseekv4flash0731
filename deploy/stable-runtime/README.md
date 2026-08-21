# Stable native-Anthropic runtime overlay

This directory builds the runtime used to keep long DeepSeek-V4 Flash agent
sessions usable on two DGX Spark nodes. It fixes the failure modes observed in
real Claude/Anthropic sessions: discarded thinking controls, malformed
historical thinking blocks, leaked DSML, missing/corrupt tool wrappers, delayed
large tool arguments, contaminated tool names, and first-request mHC JIT
stalls. Model weights are unchanged.

The image is pinned to:

- Anemll base image
  `ghcr.nju.edu.cn/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`;
- vLLM commit
  [`752a3a504`](https://github.com/vllm-project/vllm/commit/752a3a504485790a2e8491cacbb35c137339ad34);
- overlay version `0.1.9-stable-20260821`.

All patches are applied with `--fuzz=0`. A base-image drift therefore fails the
build instead of silently relocating a hunk.

## Patch set

1. `0001`: merge adjacent assistant records before prompt encoding (vLLM
   [#50686](https://github.com/vllm-project/vllm/pull/50686));
2. `0002`: avoid historical empty `<think></think>` blocks (vLLM
   [#52254](https://github.com/vllm-project/vllm/pull/52254));
3. `0003`: accept equivalent full-width, ASCII, and abbreviated DSML closers;
4. `0004`: warm the four mHC TileLang operations present on the pinned NVIDIA
   code path (a pinned-base adaptation of vLLM
   [#52941](https://github.com/vllm-project/vllm/pull/52941));
5. `0005`: provisionally recover a complete invoke whose outer wrapper was
   omitted/corrupted, but only after validating the declared tool and closing
   marker (vLLM [#52645](https://github.com/vllm-project/vllm/pull/52645)
   through `c848ab5`);
6. `0006`: stream large string tool arguments in linear time with JSON escaping
   and schema gating (vLLM
   [#52865](https://github.com/vllm-project/vllm/pull/52865));
7. `0007`: bound and harden malformed-prefix/recovery state discovered during
   whole-series review;
8. `0008`: repair a truncated DSML prefix inside an invoke name only when the
   remaining suffix exactly matches a tool declared by the current request;
   every other undeclared name is suppressed;
9. `0009`: keep emitted long-string JSON deltas list-backed and materialize
   once, removing the remaining repeated-prefix copy from the fast path;
10. `0010`: accept Anthropic manual/adaptive thinking, preserve explicit
    `output_config.effort`, and translate protocol-native controls into vLLM's
    common reasoning request fields instead of silently discarding them (vLLM
    [#29915](https://github.com/vllm-project/vllm/issues/29915) and
    [#50473](https://github.com/vllm-project/vllm/issues/50473));
11. `0011`: enforce per-request numeric thinking budgets in Model Runner V2's
    normal and speculative sampler paths (pinned-base backport of merged vLLM
    [#46727](https://github.com/vllm-project/vllm/pull/46727), commit
    `72c0d6765793e4c7242c3586274af3e1a8aca170`);
12. `0012`: seed Responses reasoning-token accounting from the parser's
    prompt-opened reasoning state, matching the DeepSeek-V4 chat template
    (bounded local fix for vLLM
    [#49711](https://github.com/vllm-project/vllm/issues/49711) and proposed
    [#49743](https://github.com/vllm-project/vllm/pull/49743)).

The MiaAI Issue #22 overlay additionally routes padded `nvfp4_ds_mla` through
the validated fast MLA kernel path. In this runtime, `nvfp4_ds_mla` and
`fp8_ds_mla` use the same padded 584-byte DeepSeek-V4 MLA layout; it is not a
generic half-size 4-bit KV cache. See [PATCH_REVIEW.md](PATCH_REVIEW.md) for the
review invariants, tests, and remaining limitations.

## Build on both nodes

Run this directory's build on Head and Worker:

```bash
cd deploy/stable-runtime
./build.sh
```

The default image is `deepseek-v4-flash:0.1.9-stable-20260821`. Docker image IDs
may differ because local build metadata differs; the hashes of the patched vLLM
files must match across both nodes.

## Runtime profile

The tracked single-user/full-window defaults are:

```text
max_model_len                  1048576
max_num_seqs                   6
max_num_batched_tokens         16384
long_prefill_token_threshold   0
gpu_memory_utilization         0.835
kv_cache_dtype                 nvfp4_ds_mla
MTP draft depth                5
requested CUDA Graph ceiling  seqs * (MTP + 1) = 36
async scheduling               disabled
breakable CUDA Graph           disabled (regular graphs)
prefix cache retention env     disabled
```

On the pinned vLLM build, a requested ceiling of `36` is normalized to capture
sizes `[1, 2, 4, 8, 16, 24, 32]`. One-sequence MTP-5 decode is covered by the
8-token graph; a six-sequence step can exceed the largest full graph and use a
piecewise/eager fallback. Do not describe `36` as an actually captured shape.
Rounding the request to `40` may improve the six-way lane, but needs a separate
memory and concurrency A/B and is not part of the single-user 1M profile.

Threshold `0` lets one active 1M prefill consume the 16K batch. This is the
right lane for one user filling the context window; it deliberately favors
time-to-first-token over fairness. For a shared service or concurrent subagents,
use the paired fallback:

```bash
GPU_MEM=0.78 \
MAX_BATCHED_TOKENS=8192 \
LONG_PREFILL_TOKEN_THRESHOLD=1024 \
bash start-head.sh
```

Do not set `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unless the matching hybrid-SWA
coordinator fix has also been installed and validated. The stable image does
neither.

## Start and connect

Start Worker first, then Head:

```bash
# Worker
cd deploy
bash start-worker.sh

# Head
cd deploy
bash start-head.sh
```

The Head/Worker RoCE addresses are cluster-internal. Claude Code, Cherry Studio,
OpenCode, or another client connects to the Head's reachable management/LAN/VPN
address on port `8888`, not to a RoCE-only address.

The runtime exposes native Anthropic endpoints. UniClaudeProxy or another
Anthropic-to-OpenAI conversion layer is not required:

```bash
curl http://HEAD-CLIENT-IP:8888/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "deepseek-v4-flash",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Reply only OK."}]
  }'
```

## Thinking compatibility

Use the protocol-native effort field when the client exposes one:

| Endpoint | Recommended request field |
| --- | --- |
| `/v1/messages` | `"thinking":{"type":"adaptive"}, "output_config":{"effort":"max"}` |
| `/v1/chat/completions` | `"reasoning_effort":"max"` |
| `/v1/responses` | `"reasoning":{"effort":"max"}` |

Manual `thinking: {"type":"enabled","budget_tokens":N}` and the vLLM Chat
extension `thinking_token_budget: N` are also accepted. Patch `0011` enforces
these as exact reasoning-token cutoffs in Model Runner V2's normal and
speculative sampler paths. A positive budget without explicit effort activates
the neutral `high` DeepSeek prompt; zero disables thinking for the Chat
extension. Explicit effort wins. Budgets below `1024` are useful for testing the
boundary but can force a mid-thought transition and damage final/tool quality.

OpenAI Responses uses standard `reasoning: {"effort":"..."}`. It has no
reasoning-only numeric budget in this mapping: `max_output_tokens` limits the
combined reasoning and visible output. Patch `0012` makes
`usage.output_tokens_details.reasoning_tokens` count the DeepSeek template's
prompt-opened reasoning span.

The live `/openapi.json` also advertises these controls: `ChatCompletionRequest`
contains `reasoning_effort` and `thinking_token_budget`, `ResponsesRequest`
contains `reasoning`, and the Anthropic Messages/count-token request schemas
contain `thinking` plus `output_config`. Generated clients therefore do not have
to rely on undocumented extra JSON fields.

Validate all paths with:

```bash
python3 probes/thinking_compat.py \
  --base-url http://HEAD-CLIENT-IP:8888
```

For Claude Code configurations that use a model suffix to advertise the window,
`deepseek-v4-flash[1m]` may be used as the client-side model label when the
client strips/aliases it to the served `deepseek-v4-flash` model.

## Reproducible gates

First run a smaller native-Anthropic history/tool gate:

```bash
python3 probes/anthropic_long_agent.py \
  --base-url http://HEAD-CLIENT-IP:8888/v1 \
  --model deepseek-v4-flash \
  --target-tokens 32000 \
  --effort max
```

Then validate a nearly full window, one client-side compaction round, and a
post-compaction tool call:

```bash
python3 probes/anthropic_compaction_round.py \
  --base-url http://HEAD-CLIENT-IP:8888/v1 \
  --model deepseek-v4-flash \
  --target-tokens 1040000 \
  --turns 96 \
  --effort max
```

Compaction is intentionally client-side. vLLM serves and counts the Anthropic
messages; the agent client decides when to summarize and replace its transcript.
The probe rejects raw DSML/Anthropic marker leakage and verifies four anchors
before and after compaction. It deliberately keeps the same ordered tool
inventory in the full, compact, and post-compact phases so vLLM can reuse the
message prefix just as it can for a real agent whose tools did not change.

The reviewed `0.1.9` image passed this exact gate with 1,040,105 full-context
input tokens in 1,127.003s. Its 1,040,118-token compaction request hit
1,039,872 cached tokens (99.976349%), recomputed 246, and completed in 11.293s.
The 658-token post-compaction request completed a structured tool call in
5.579s while retaining all four anchors. See [PATCH_REVIEW.md](PATCH_REVIEW.md)
for the full accounting and log checks.

## Rollback

The launch scripts accept one-shot overrides without editing tracked files:

```bash
KV_CACHE_DTYPE=fp8 \
IMAGE=deepseek-v4-flash:0.1.8-stable-20260821 \
bash start-worker.sh

KV_CACHE_DTYPE=fp8 \
IMAGE=deepseek-v4-flash:0.1.8-stable-20260821 \
bash start-head.sh
```

Keep the previous stopped Head and Worker containers under explicit names until
the full-window gate and a real agent soak have completed.

## Deliberate limits

- The patch series targets the exact pinned vLLM commit; it is not promised to
  apply to arbitrary future vLLM images.
- Raw DSML is delimiter based. A string argument containing the exact DSML
  `</...parameter>` delimiter is intrinsically ambiguous without an upstream
  escaping/length-framing protocol change.
- `max_num_seqs=6` does not imply six simultaneous 1M contexts. Capacity must
  be read from the startup `GPU KV cache size` and `Maximum concurrency` logs.
