# Stable runtime overlay

This is a reproducible thin image over the pinned Anemll `0.1.1` runtime.

It backports vLLM PR #50686 (commit `b68ecd479ca6c29ba36601d6cb00053c6b4fccfb`)
to merge consecutive assistant messages before DeepSeek-V4 prompt encoding. It also
patches the DeepSeek-V4 parser to accept canonical full-width DSML, ASCII DSML, and
abbreviated closing tags. Without these fixes, malformed think/EOS boundaries can
accumulate over long sessions and valid tool calls can leak into assistant text as
`<|DSML|...>` or `</parameter>` markup.

The deployment also runs with async scheduling disabled and `MAX_NUM_SEQS=6` by
default. Synchronous scheduling avoids the known unstable scheduling path without
forcing strictly serial execution: vLLM may keep up to six sequences active. Prefix
caching and DSpark remain enabled for long-context performance. Set
`MAX_NUM_SEQS=1` only as a conservative isolation/diagnostic override.

The patches are derived from vLLM code and remain under Apache-2.0. See
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
