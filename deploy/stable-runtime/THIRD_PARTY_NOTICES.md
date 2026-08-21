# Third-party notice

The runtime patches modify vLLM, which is distributed under Apache License 2.0.
The copied/adapted patch content remains subject to that license; a copy is
included as `LICENSE.Apache-2.0`. This does not change the MIT license covering
this repository's original deployment scripts and documentation.

The overlay is fixed to upstream vLLM commit
[`752a3a504`](https://github.com/vllm-project/vllm/commit/752a3a504485790a2e8491cacbb35c137339ad34).
Patch provenance is:

- `0001`: vLLM [PR #50686](https://github.com/vllm-project/vllm/pull/50686),
  commit `b68ecd479ca6c29ba36601d6cb00053c6b4fccfb`;
- `0002`: vLLM [PR #52254](https://github.com/vllm-project/vllm/pull/52254),
  commit `d48c411ab361d8c89a91e0917eda34e45d0dfba2`, adapted to coexist
  with `0001`;
- `0003`: local compatibility patch for equivalent DeepSeek-V4 DSML marker
  variants observed in model output;
- `0004`: pinned-base NVIDIA adaptation of vLLM
  [PR #52941](https://github.com/vllm-project/vllm/pull/52941), whose reviewed
  head was `e9ceeb36605ef0ad9abed849f68b7416176a4bd8`; the newer broadcast
  symbol/path is intentionally omitted because it does not exist at the pinned
  vLLM commit;
- `0005`: conservative subset of vLLM
  [PR #52645](https://github.com/vllm-project/vllm/pull/52645) through commit
  `c848ab5aa41b8bd848fcdf169c43fea278f7e751`, manually merged with
  `0003`;
- `0006`: vLLM [PR #52865](https://github.com/vllm-project/vllm/pull/52865),
  commit `08cbee10aa35bbe8c7d46cd88049547d6ea1ed11`, manually merged with
  the preceding parser patches;
- `0007`, `0008`, and `0009`: local hardening derived from whole-series review
  and native-Anthropic regression samples. They generalize bounded recovery
  state, enforce declared-tool-name validation, and keep long emitted JSON
  fragments list-backed until one structural materialization;
- `0010`: local Anthropic/OpenAI thinking-control compatibility patch, tracked
  against vLLM [issue #29915](https://github.com/vllm-project/vllm/issues/29915)
  and the Model Runner V2 limitation in
  [issue #50473](https://github.com/vllm-project/vllm/issues/50473). It preserves
  effort semantics and translates Anthropic controls into the common request;
- `0011`: pinned-base adaptation of merged vLLM
  [PR #46727](https://github.com/vllm-project/vllm/pull/46727), commit
  `72c0d6765793e4c7242c3586274af3e1a8aca170`. The upstream rejection sampler
  was refactored after this pinned base, so the adaptation passes the existing
  `idx_mapping` through the older sampler flow while retaining the upstream
  thinking-budget state machine;
- `0012`: local bounded Responses usage-accounting fix for a reasoning span
  opened by the prompt template, tracked against vLLM
  [issue #49711](https://github.com/vllm-project/vllm/issues/49711) and proposed
  [PR #49743](https://github.com/vllm-project/vllm/pull/49743).

The Issue #22 hotfix under `upstream-hotfixes/` comes from
[`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
at commit `8997d41746e5e1441c15b5e188e6922d1d7bf233`. The executable change is
unchanged; this copy corrects only a comment typo. Provenance is recorded in
`upstream-hotfixes/SOURCE.md`. That repository distributes the script under the
MIT license; its copyright and license are included as `LICENSE.MiaAI-MIT`.

These patches are redistributed to make the exact runtime reproducible against
the pinned container image. Some cited vLLM pull requests were open/unmerged at
the time of this overlay; passing local gates is not a claim of upstream
maintainer approval. PR #46727 was merged upstream and is explicitly identified
as such above.
