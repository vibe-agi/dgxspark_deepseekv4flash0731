# Stable NVFP4 runtime overlay

This directory builds a reproducible thin image over Anemll `0.1.1`. It keeps
the model weights unchanged and installs exactly three narrowly scoped runtime
fixes:

1. vLLM PR #50686 history normalization, which merges adjacent assistant
   records before DeepSeek-V4 prompt encoding;
2. tolerant full-width/ASCII/abbreviated DSML parsing, which prevents valid
   tool syntax from leaking into assistant text;
3. MiaAI Issue #22, which routes padded `nvfp4_ds_mla` through the working fast
   MLA kernel path.

`nvfp4_ds_mla` in this runtime uses the same padded 584-byte DeepSeek-V4 MLA
layout as `fp8_ds_mla`. It is not a generic 4-bit KV format and does not halve
the KV footprint. Its value here is the validated kernel route and compatibility
with the current DSpark recipe.

The repository does not ship the other upstream candidate hotfixes. Applying
the complete upstream hotfix set at once caused a reproducible shared-memory broadcast stall on the third
sequential 8K cold request; the isolated Issue #22 image completed ten such
requests. See `upstream-hotfixes/SOURCE.md`.

## Build on both nodes

```bash
cd deploy/stable-runtime
./build.sh
```

The default tag is `deepseek-v4-flash:0.1.1-stable-nvfp4-20260819`. Both nodes
must build from the same base image and produce identical hashes for the patched
vLLM files. The Docker image IDs may differ because build metadata is local.

## Runtime profile

The tracked defaults select:

```text
max_model_len                  1048576
max_num_seqs                   6
max_num_batched_tokens         16384
long_prefill_token_threshold   0
gpu_memory_utilization         0.835
kv_cache_dtype                 nvfp4_ds_mla
MTP draft depth                5
max_cudagraph_capture_size     seqs * (MTP + 1) = 36
async scheduling               disabled
breakable CUDA Graph           disabled (regular graphs)
prefix cache retention env     disabled
```

This is the validated single-user/full-window lane: threshold 0 lets one active
prefill consume the 16K batch, while memory 0.835 retains enough KV capacity for
one 1M request. For concurrent subagents or a shared service, use the paired
fairness fallback `GPU_MEM=0.78 MAX_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024`. Do not set
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unless the matching Issue #26 hybrid-SWA
coordinator fix has also been installed and validated; the stable image
intentionally does neither.

## Reproducible protocol probe

After starting Worker and then Head, run the native Anthropic long-agent gate on
the Head:

```bash
python3 probes/anthropic_long_agent.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash \
  --target-tokens 240000
```

The probe constructs alternating agent history plus a historical
`tool_use`/`tool_result`, forces a final tool call, and rejects leaked DSML or
Anthropic XML markers.

## FP8 rollback

The launch scripts accept one-shot overrides without editing tracked files:

```bash
KV_CACHE_DTYPE=fp8 \
IMAGE=deepseek-v4-flash:0.1.1-stable-nvfp4-20260819 \
bash ../start-worker.sh

KV_CACHE_DTYPE=fp8 \
IMAGE=deepseek-v4-flash:0.1.1-stable-nvfp4-20260819 \
bash ../start-head.sh
```

Keep the last known-good stopped containers under explicit names until the new
profile has completed a soak period.
