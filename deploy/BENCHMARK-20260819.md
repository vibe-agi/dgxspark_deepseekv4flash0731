# Dual DGX Spark optimization report — 2026-08-19/20

This report records the A/B that selected the production profile. It contains
no management addresses, SSH aliases, usernames, or raw private prompts.

## Method

- two DGX Spark GB10 nodes, TP=2, PP=1, dual RoCE, text-only serve;
- official DeepSeek-V4-Flash-0731 weights, Anemll `0.1.1` base;
- 1,048,576 model limit, six active sequences, synchronous scheduling;
- unique cold prefixes, fixed 128-token output, `thinking=false`;
- cell values are median decode tok/s after first token; aggregate values are
  all completed streams divided by wall time;
- cold JIT/autotune trial is retained in the range rather than silently removed.

## Runtime-layer isolation

| Profile | C1 p256 | C1 p8192 | C6 p256 / stream | Warm C6 aggregate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| FP8, Anemll auto/breakable Graph | 75.4 | 81.2 | 44.3 | 191–207 | baseline |
| FP8, regular Graph | 72.3 | 87.2 | 46.1 | 194–203 | stable control |
| NVFP4 + complete candidate hotfix set | 78.2 | two passes, then stall | — | — | rejected |
| NVFP4, Issue #22 only, MTP-5 | 78.0 | 79.1 (10/10) | 43.2 | 190–198 | selected |
| NVFP4, Issue #22 only, MTP-6 | 73.2 | 81.9 | 41.9 | 151–193 | rejected |

The complete candidate bundle stalled on the third sequential 8K cold request
and logged `No available shared memory broadcast block found in 60 seconds`.
The selected image removes the scheduler, spin-wait, hybrid-cache and optional
performance changes and retains only the NVFP4 Issue #22 dispatch fix plus the
already-proven tokenizer/DSML fixes.

The FP8/NVFP4 differences are smaller than the run-to-run spread, so this test
does not support a blanket “NVFP4 is faster” claim. NVFP4 was selected because
the isolated path was stable and matches the maintained DSpark recipe, not
because it doubled capacity or decode speed. Both padded DeepSeek-V4 layouts are
584 bytes and produced roughly a 1.4M-token KV pool at `GPU_MEM=0.78`.

## Prefill fairness and concurrency

With `LONG_PREFILL_TOKEN_THRESHOLD=1024`, Issue #22-only NVFP4 completed:

| Workload | TTFT | Per-stream decode | Aggregate |
| --- | ---: | ---: | ---: |
| C6 × 8K, warm | 22.8s | 37.9 tok/s | 29.1 tok/s |
| C1 × 128K, 2/2 | 74.8s | 80.2 tok/s median | 1.7 tok/s end-to-end |

After each gate, running requests, waiting requests, and KV utilization returned
to zero. No shared-memory, NCCL, or Python traceback was present in the logs.

## Long-agent and near-full-window gates

| Gate | Result |
| --- | --- |
| Native OpenAI smoke | exact expected text |
| Native Anthropic smoke | exact expected text |
| Native Anthropic forced tool call | structured `tool_use`, exact JSON input |
| 24-round Anthropic agent/tool history | PASS at 239,869 input tokens in 171.4s; no DSML/XML leakage |
| 900K cold prompt | PASS; 931.0s TTFT, 73.5 decode tok/s, 128/128 output tokens |

The 900K result demonstrates usable near-full-window capacity for one request.
It does not imply six simultaneous 1M contexts or guarantee retrieval quality
for every 900K prompt.

## MTP-5 versus MTP-6

MTP-6 was not retained. Its short-prompt and six-way results were lower, its
measured draft acceptance was about 66% for this sweep, and Graph capture grew
from roughly 0.33 GiB to 1.15 GiB. The small 8K C1 gain did not compensate for
the other regressions. Production remains MTP-5 with capture size
`6 × (5 + 1) = 36`.

Later startup-log review showed that this pinned vLLM normalizes a requested
ceiling of `36` to actual capture sizes ending at `32`. The historical A/B below
used the requested value shown; it is not evidence that a 36-token full graph
was captured.

## Hardware state

During a long prefill both GPUs reported P0, about 96% utilization, CPU governors
were `performance`, and all NVIDIA software/hardware power and thermal throttle
reasons were inactive. The observed decode ceiling is therefore not explained
by an accidental low-power profile.

## 2026-08-19 concurrency-safe baseline

```text
KV_CACHE_DTYPE=nvfp4_ds_mla
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024
GPU_MEM=0.78
MTP_NUM_TOKENS=5
MAX_CUDAGRAPH=36              # derived: seqs * (MTP + 1)
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=(unset)
async scheduling=(disabled)
```

This remains the fairness-oriented fallback for a shared service or concurrent
subagents. It prevents one long prefill from consuming the whole scheduling
batch.

## 2026-08-20 single-user full-window A/B

The deployment is normally used by one operator and does not require KV
persistence across restarts. A second A/B therefore optimized the cold-prefill
lane while retaining the same image, model, TP=2 topology, MTP-5, synchronous
scheduler, prefix caching, and native Anthropic endpoint. Each measured prompt
used a unique prefix; one-time JIT trials were excluded from the steady-state
comparison.

| Profile | 88K cold prefill | Computed prefill rate | Result |
| --- | ---: | ---: | --- |
| batch 8192, threshold 1024, memory 0.78 | 52.635s | 1,673 tok/s | original |
| batch 8192, threshold 0, memory 0.78 | 45.100s | 1,952 tok/s | +16.7% rate |
| batch 16384, threshold 0, memory 0.835 | 42.434s | 2,075 tok/s | selected |

The selected lane reduced 88K prefill latency by 19.4% and increased computed
prefill throughput by about 24%. `MAX_BATCHED_TOKENS=16384` could not retain the
1M model limit at `GPU_MEM=0.78`: only 8.22 GiB remained for KV while vLLM
required 10.91 GiB. The paired `GPU_MEM=0.835` profile started with 15.33 GiB
available KV memory and passed the 1M capacity check.

Native Anthropic append-only reuse was then validated at two scales:

| Input | Cold prefill | Appended-turn cache hit | Appended prefill | Appended TTFT |
| --- | ---: | ---: | ---: | ---: |
| 44,037 tokens | 20.044s | 99.38% | 0.295s | 0.364–0.366s |
| 1,039,984 tokens | 1,050.138s | 99.9881% | 0.554s | 2.759s |

The full-window request used 1,039,984 input tokens (99.18% of the configured
1,048,576 window), completed without OOM, queueing, or container restart, and
returned the exact expected text. Its cold average was 990 tok/s. The appended
request contained 1,039,996 input tokens, hit 1,039,872 cached tokens, recomputed
124, and completed server-side in 2.998s.

An external-client control also separated network upload from engine time. A
250,037-token native Anthropic body was about 1.04 MB; on a relayed overlay path
the client wall time was 193.5s versus 140.9s server-side. Its appended turn hit
99.923% of prompt tokens, yet retransmitting the same JSON history still added
about 42.7s outside the engine. Prefix caching avoids GPU recomputation; it does
not stop stateless clients from uploading the complete history every turn.

## 2026-08-21 thinking/API compatibility gate

Overlay `0.1.9-stable-20260821` adds the merged vLLM Model Runner V2
thinking-budget implementation and a bounded Responses usage-accounting fix.
The live matrix used one arithmetic answer (`8887500053`) across all three APIs:

| Path | Control | Result |
| --- | --- | --- |
| Anthropic Messages | manual `budget_tokens=1024` | PASS in 4.141s; non-empty thinking, exact answer |
| Anthropic Messages | adaptive + `effort=max` | PASS in 4.821s; non-empty thinking, exact answer |
| OpenAI Chat | `thinking_token_budget=1024` | PASS in 2.899s; model naturally ended before the cap, exact answer |
| OpenAI Chat | `reasoning_effort=max` | PASS in 6.094s; non-empty reasoning, exact answer |
| OpenAI Responses | `reasoning.effort=max` | PASS in 5.818s; 263 output tokens, 257 reported as reasoning, exact answer |
| All three APIs | streaming | PASS; protocol-native reasoning and completion events |
| All three APIs | thinking disabled | PASS; no reasoning item/block/field |
| Chat and Responses | automatic structured tool call | PASS; exact tool and JSON arguments, no DSML/XML leakage |

The first numeric-budget request compiled the new Triton budget kernels once;
subsequent requests reused them. Three concurrent Chat requests with exact
32/64/128-token budgets produced 38/70/134 completion tokens respectively: each
budget plus the same six-token visible answer. The 32- and 64-token answers were
wrong while 128 was correct, showing both the hard boundary and the quality cost
of truncating mid-thought. The normal recommendation is protocol effort plus a
budget of at least 1,024 when a hard cap is needed. OpenAI Responses has no
separate standard numeric reasoning cap in this profile; `max_output_tokens`
covers reasoning plus visible output.

## 2026-08-21 `0.1.9` full-window and compaction gate

The final gate used the same reviewed image and the native Anthropic endpoint:

```bash
python3 probes/anthropic_compaction_round.py \
  --base-url http://HEAD-CLIENT-IP:8888/v1 \
  --model deepseek-v4-flash \
  --target-tokens 1040000 \
  --turns 96 \
  --effort max
```

| Phase | Input | Output | Server elapsed | Result |
| --- | ---: | ---: | ---: | --- |
| Full context | 1,040,105 | 85 | 1,127.003s | PASS; structured tool result, no marker leakage |
| Compaction | 1,040,118 | 381 | 11.293s | PASS; all four anchors retained |
| Post compact | 658 | 84 | 5.579s | PASS; structured tool call from summary |

The compaction request hit `1,039,872` prefix tokens, recomputed 246, and
therefore achieved a `99.976349%` prompt-cache hit. The cold full request's
end-to-end input rate was approximately 923 tok/s; this includes output decode
and is deliberately not mislabeled as a pure Prefill measurement. The three
input counts sum exactly to the post-run `prompt_tokens_total` delta, and the
cache-hit counter delta equals `1,039,872`. The server ended at running 0,
waiting 0, KV usage 0, and HTTP 200 health, with no ERROR, traceback, exception,
or NCCL failure in the run log.

## Current single-user profile

```text
KV_CACHE_DTYPE=nvfp4_ds_mla
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_BATCHED_TOKENS=16384
LONG_PREFILL_TOKEN_THRESHOLD=0
GPU_MEM=0.835
MTP_NUM_TOKENS=5
MAX_CUDAGRAPH=36              # derived: seqs * (MTP + 1)
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=(unset)
async scheduling=(disabled)
```

`LONG_PREFILL_TOKEN_THRESHOLD=0` is intentional for one active request. A human
operator launching concurrent subagents also creates concurrency; use the
2026-08-19 `8192 / 1024 / 0.78` fallback when decode fairness matters more than
single-request cold-prefill latency.
