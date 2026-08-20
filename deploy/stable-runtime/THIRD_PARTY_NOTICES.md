# Third-party notice

`patches/0001-merge-consecutive-assistant-messages.patch` is derived from the
DeepSeek-V4 tokenizer changes proposed to the vLLM project in
[vLLM PR #50686](https://github.com/vllm-project/vllm/pull/50686), commit
`b68ecd479ca6c29ba36601d6cb00053c6b4fccfb`.

`patches/0002-tolerate-deepseek-v4-dsml-drift.patch` modifies vLLM's
DeepSeek-V4 streaming parser so equivalent ASCII/full-width DSML markers and
abbreviated closing tags are parsed into the same tool events.

vLLM is distributed under the Apache License 2.0. The copied and modified patch
content in this directory remains subject to that license; a copy is included as
`LICENSE.Apache-2.0`. This notice does not change the MIT license that applies to
the repository's original deployment scripts and documentation.

The patches are redistributed here solely to make the exact runtime fixes
reproducible against the pinned Anemll container image.

The Issue #22 hotfix under `upstream-hotfixes/` is vendored from
[`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
at commit `8997d41746e5e1441c15b5e188e6922d1d7bf233`. The script is kept
byte-for-byte and its provenance is recorded in `upstream-hotfixes/SOURCE.md`.
That repository distributes the script under the MIT license; its copyright
and license text are included as `LICENSE.MiaAI-MIT`.
