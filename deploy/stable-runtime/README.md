# Stable runtime overlay

This is a reproducible thin image over the pinned Anemll `0.1.1` runtime.

It backports vLLM PR #50686 (commit `b68ecd479ca6c29ba36601d6cb00053c6b4fccfb`)
to merge consecutive assistant messages before DeepSeek-V4 prompt encoding. Without
this fix, agent gateways can replay one assistant turn as a content message followed
by a reasoning/tool-call message, producing malformed think/EOS boundaries that
accumulate over long sessions.

The deployment also runs with async scheduling disabled and `MAX_NUM_SEQS=1` by
default. Serial scheduling prevents a later request in the same scheduler step from
hitting a prefix block whose KV has not yet been written. Prefix caching and DSpark
remain enabled for long-context performance.

The patch is derived from vLLM code and remains under Apache-2.0. See
`THIRD_PARTY_NOTICES.md` and `LICENSE.Apache-2.0`. The rest of this repository uses
the repository-level MIT license.

Build on both nodes:

```bash
cd deploy/stable-runtime
./build.sh
```

Rollback does not delete anything:

```bash
IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 MAX_NUM_SEQS=1 ../start-worker.sh
IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1 MAX_NUM_SEQS=1 ../start-head.sh
```
