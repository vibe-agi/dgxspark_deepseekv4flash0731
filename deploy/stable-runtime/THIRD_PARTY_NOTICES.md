# Third-party notice

`patches/0001-merge-consecutive-assistant-messages.patch` is derived from the
DeepSeek-V4 tokenizer changes proposed to the vLLM project in
[vLLM PR #50686](https://github.com/vllm-project/vllm/pull/50686), commit
`b68ecd479ca6c29ba36601d6cb00053c6b4fccfb`.

vLLM is distributed under the Apache License 2.0. The copied and modified patch
content in this directory remains subject to that license; a copy is included as
`LICENSE.Apache-2.0`. This notice does not change the MIT license that applies to
the repository's original deployment scripts and documentation.

The patch is redistributed here solely to make the exact runtime fix reproducible
against the pinned Anemll container image.
