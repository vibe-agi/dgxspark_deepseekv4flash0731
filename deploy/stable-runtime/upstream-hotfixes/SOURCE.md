# Vendored DSpark hotfix provenance

The Issue #22 script originates from:

- repository: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`;
- commit: `8997d41746e5e1441c15b5e188e6922d1d7bf233`;
- captured: 2026-08-19.

The executable change remains the upstream Issue #22 change: route padded
`nvfp4_ds_mla` through the validated fast MLA kernel in the pinned Anemll base.
This repository corrected only the prose typo `The584-byte` to `The 584-byte`;
no command or replacement expression was altered.

The scheduler, hybrid-prefix-cache, spin-wait, tool-truncation, and broad
performance scripts are not included. A 2026-08-19 dual-Spark A/B of the whole
candidate set stalled on the third sequential 8K cold request and logged a
shared-memory broadcast timeout. The isolated Issue #22 route completed ten
such requests, a six-way mixed prefill/decode gate, and long-agent protocol
tests. Any additional upstream script must therefore be introduced and tested
one change at a time.

When refreshing Issue #22, pin a new upstream commit, rebuild both nodes, compare
the patched-file hashes, and rerun throughput, long-context, concurrency, and
tool-protocol gates.
