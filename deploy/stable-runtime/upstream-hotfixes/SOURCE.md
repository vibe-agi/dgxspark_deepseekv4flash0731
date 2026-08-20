# Vendored DSpark hotfix provenance

The Issue #22 script is copied without local edits from:

- repository: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- commit: `8997d41746e5e1441c15b5e188e6922d1d7bf233`
- captured: 2026-08-19

The upstream candidate set was audited, but this repository deliberately
vendors and applies **only** `hotfix-nvfp4-ds-mla-issue22.sh`. That is the one runtime change required to
route padded `nvfp4_ds_mla` through the working fast MLA kernel in the pinned
Anemll `0.1.1` image.

The scheduler, hybrid-prefix-cache, spin-wait, tool-truncation and performance
scripts are not shipped. A
2026-08-19 dual-Spark A/B of the whole set stalled on the third sequential 8K
cold request and logged a shared-memory broadcast timeout. The isolated Issue
#22 image completed ten such requests, six-way mixed prefill/decode, and the
long-agent protocol gate. Do not batch-apply the other scripts without a fresh,
one-change-at-a-time validation.

When refreshing Issue #22, pin one upstream commit, rebuild both nodes, and
rerun the throughput, long-context, concurrency and tool-protocol gates.
