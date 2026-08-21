# Runtime patch-set quality review

Review date: 2026-08-21
Overlay: `0.1.9-stable-20260821`
Pinned vLLM: `752a3a504485790a2e8491cacbb35c137339ad34`
Pinned base digest: `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`

## Outcome

The series is suitable for this pinned two-DGX-Spark runtime after two review
passes, an API-semantics pass, and live native-Anthropic/OpenAI regression. It
is not presented as a generic patch set for arbitrary vLLM versions. Every hunk
applies with `--fuzz=0`, both
nodes build the same patched-file hashes, malformed recovery is bounded and
lossless on rejection, and tool-name repair is constrained to tools declared in
the current request.

Review was intentionally separate from “the container starts.” It covered:

- prompt/tokenizer semantics across replayed assistant turns;
- parser state transitions and commit/abort behavior;
- arbitrary streaming boundaries, including one-character closers;
- time complexity for large tool strings;
- memory growth under malformed/incomplete output;
- tool-name trust boundaries;
- pinned-source provenance and patch replay;
- dual-node image parity, startup warmup, rollback, and live API behavior.

## Findings fixed during review

1. One earlier compatibility patch depended on fuzzy relocation. It was
   regenerated against the exact base; the Docker build now forces
   `patch --fuzz=0` for the complete series.
2. A malformed marker made only whitespace progress and could grow a hold buffer
   without a bound. The prefix buffer is now capped at 4,096 characters.
3. Provisional wrapper recovery used repeated string concatenation. Raw recovery
   is now stored as parts and joined once, with an 8 MiB ceiling and lossless
   abort/replay.
4. A recovered invoke could leave a stale “outer close pending” flag and consume
   a later literal close after normal prose. Real prose now clears that flag.
5. The long-string fast path treated every `<` as a possible DSML boundary,
   regressing markup/code arguments toward repeated whole-buffer conversion. It
   now falls back only for a real pending DSML tag prefix.
6. The base image used a mutable tag. It is now pinned by digest in the
   Dockerfile, direct build script, and the `prepare.sh` configuration path so
   orchestration cannot silently override the pin.
7. Initial tests assumed a complete tool name/argument arrived in one event.
   Verification now reconstructs legal deltas and tests every character split.
8. A live compact smoke produced a canonical invoke whose name was contaminated
   by a truncated outer marker. Patch `0008` repairs it only if removing one or
   more bounded DSML prefixes yields an exact declared-tool match; arbitrary or
   undeclared names are suppressed.
9. The long-string fast path still appended each emitted JSON fragment to one
   immutable string. That preserved correctness but contradicted the claimed
   O(n) bound for very large arguments. Patch `0009` stores fragments in a list
   and joins once when the parser returns to structural conversion.
10. The first compaction probe changed the complete tool inventory between its
    full-context and summary requests. Since tools are rendered before messages,
    this invalidated the prompt prefix and measured two cold prefills. The probe
    now keeps one ordered inventory across all phases, matching a real agent.
11. A stale ignored `config.local.sh` can override a newly tracked image default.
    The rollout gate now verifies the image printed by each launcher; the example
    warns against locally pinning `IMAGE`, while intentional private tags remain
    supported through the same override mechanism.
12. The launcher requests a CUDA Graph ceiling of `6 * (5 + 1) = 36`, but the
    pinned vLLM normalizes its actual capture list to a maximum of `32`. This is
    sufficient for the single-sequence lane under review, but the documentation
    no longer claims that six-way MTP decode is fully captured. A ceiling of 40
    is deferred to a measured concurrency/memory A/B.
13. Anthropic's top-level `thinking` object was accepted as an unknown Pydantic
    field and discarded. Forwarding its numeric budget exposed a second issue:
    DSpark's mandatory Model Runner V2 rejected `thinking_token_budget` with
    HTTP 500. Patch `0010` parses manual/adaptive thinking, gives explicit effort
    precedence, and uses neutral `high` only to activate thinking when effort is
    absent. The rejected `0.1.5` candidate remains available for rollback.
14. The initial compatibility fix avoided the V2 500 by clearing the numeric
    cap, but that made manual budgets approximate. Patch `0011` is a pinned-base
    adaptation of merged vLLM PR #46727: its GPU sampler tracks each request's
    reasoning state and forces the configured transition at the exact budget in
    both normal and speculative paths. Live 32/64/128/1024-token tests confirmed
    hard boundaries without leaking marker text.
15. OpenAI Responses returned a reasoning item but reported zero
    `reasoning_tokens`. DeepSeek-V4 opens `<think>` in its prompt, so the generated
    suffix has a close marker without a generated start marker. Patch `0012`
    seeds counting from the parser's initial reasoning state; live Responses
    usage now reports a nonzero parser-backed split.
16. The first draft of `0012` applied with a one-line offset and GNU patch left
    `parser_engine.py.orig`. The hunk was regenerated at the exact pinned line,
    replayed from a clean worktree with zero offsets, and the image verifier now
    rejects any `.orig` or `.rej` artifact under `vllm`.
17. Review then found that unconditionally seeding a prompt-opened reasoning
    depth would double-open the span if a model also generated `<think>`, causing
    visible answer tokens after `</think>` to be misclassified. The final `0012`
    seeds only when the generated suffix has no start token, following the
    marker-presence rule in upstream PR #49743, and adds both repeated-start and
    truncated-no-end regression cases. This correction is versioned as `0.1.9`.

## Per-patch assessment

| Patch | Quality invariant | Residual scope/risk |
| --- | --- | --- |
| `0001` adjacent assistant merge | Preserves order; deep-copies input; end-of-turn flags come from the last record | Consecutive assistant records are treated as one logical turn, matching the target client replay shape |
| `0002` empty thinking | Opens a historical thinking block only when it has replayable reasoning; current generation still opens normally | Fixed to the pinned encoder templates |
| `0003` DSML variants | Equivalent full-width/ASCII/bare closers produce the same semantic events | Delimiter protocol remains inherently ambiguous if a literal argument contains its exact closing delimiter |
| `0004` mHC warmup | Uses only the four TileLang functions present on the pinned NVIDIA path; non-DSv4 exits early | Pinned NVIDIA adaptation, not a direct cherry-pick for newer vLLM/ROCm paths; adds about 50 s to cold startup |
| `0005` wrapper recovery | Holds events provisionally; validates tool name; commits only on a valid invoke close; rejects losslessly | Complete malformed calls up to the 8 MiB hold limit are recoverable |
| `0006` long arguments | Linear escaped streaming after schema-stable string detection; nullable/union fields remain buffered | Exact DSML delimiter inside raw content cannot be disambiguated locally |
| `0007` state hardening | 4K malformed-prefix cap, 8 MiB provisional cap, list-backed raw storage, stale-close clearing | Bounds parser-owned raw text; model/server output limits remain the outer resource control |
| `0008` name validation | Exact current-request allow-list; bounded prefix stripping; invalid names produce no tool delta | It repairs only DSML-prefix contamination, not arbitrary misspellings or semantic tool selection errors |
| `0009` JSON fragment accumulation | List-backed fragments are materialized once at structural fallback/final flush | Adds one list reference per streamed fragment until the current string parameter closes |
| `0010` thinking controls | Parses Anthropic manual/adaptive modes; explicit effort wins; Chat budget alone activates thinking | Protocol translation is pinned to this vLLM request schema |
| `0011` V2 thinking budget | Per-request GPU state; exact forced transition; natural close stops enforcement; normal and speculative samplers covered | Very small budgets can force an incoherent mid-thought transition; use `>=1024` for normal work |
| `0012` Responses reasoning usage | Counts only tokens inside parser-classified reasoning, including prompt-opened, truncated, and repeated-start spans | Local bounded adaptation pending resolution of the broader upstream accounting proposal |
| Issue #22 NVFP4 route | One isolated kernel-dispatch change; executable upstream replacement retained | Specific to the padded DSv4 MLA layout in this base image |

## Complexity and resource bounds

- Normal content and established long string arguments are processed in O(n)
  total time with respect to emitted characters.
- Both raw DSML input and emitted JSON deltas use list-backed accumulation on
  the long-string path; neither repeatedly copies the complete prefix.
- DSML conversion is allowed to rescan only while structure is unresolved; once
  a schema-stable string is open, each chunk is JSON-escaped once.
- Malformed outer-prefix buffering is capped at 4,096 characters.
- Provisional missing-wrapper recovery is capped at 8 MiB of raw characters,
  stored as a list and joined only on abort.
- Tool-name repair performs at most four passes over four short configured
  markers, then requires an exact declared-tool match.

## Verification matrix

| Gate | Result |
| --- | --- |
| Apply all patches from the pinned base with `--fuzz=0` | PASS on both nodes |
| Python compile + tokenizer/empty-thinking assertions | PASS on both image builds |
| Unicode, ASCII, and bare DSML variants | PASS |
| Missing wrapper: declared tool, unknown tool, incomplete call, wrong close, stale outer close | PASS |
| Every character split for canonical/recovered/unknown calls | PASS |
| 4K malformed-prefix and 8 MiB recovery-limit abort paths | PASS |
| 4 KiB/128 KiB/1 MiB/multiline/control-character/markup tool strings | PASS; 1 MiB parser gate under 10 s |
| Closing DSML markers delivered one character at a time | PASS |
| Nullable union schema does not stream prematurely | PASS |
| Contaminated declared name is repaired; arbitrary contaminated name is suppressed | PASS |
| Pinned NVIDIA mHC op discovery, shapes, and startup warmup | PASS; both ranks completed warmup |
| NVFP4 Issue #22 fast-route marker | PASS |
| Patched vLLM file hashes identical across Head and Worker | PASS |
| No `.orig`/`.rej` patch artifacts in either image | PASS; also enforced by `verify_patch.py` |
| Native-Anthropic small compact smoke | PASS after `0008` |
| Anthropic manual/adaptive, OpenAI Chat budget/effort, OpenAI Responses effort | PASS on live `0.1.9`; all five returned non-empty reasoning and exact answer `8887500053` |
| Anthropic/Chat/Responses streaming, disabled thinking, and auto tool calls | PASS; no DSML/XML marker leakage and exact tool arguments |
| OpenAI Responses reasoning-token usage | PASS; 257 of 263 output tokens classified as reasoning in the final max-effort arithmetic case |
| Live OpenAPI schema | PASS; Chat exposes `reasoning_effort`/`thinking_token_budget`, Responses exposes `reasoning`, Anthropic exposes `thinking`/`output_config` |
| Native-Anthropic ~1.04M tool → compact → post-compact tool | See “Full-window result” below |

The thinking-control matrix used `probes/thinking_compat.py`. Anthropic manual
budget, Anthropic adaptive/max, Chat budget, Chat max, and Responses max all
returned the exact answer. `/v1/messages/count_tokens` accepted adaptive/max
and returned the same 101-token prompt size reported by generation. All three
streaming paths, disabled thinking, and automatic tool calls also passed. Three
concurrent Chat boundary probes used 32/64/128-token budgets and returned
38/70/134 completion tokens: the requested reasoning budget plus the same
six-token visible answer. The first two deliberately truncated answers were
wrong; 128 was correct. The first request compiled the budget kernel once;
subsequent requests reused it.

## Full-window result

The final measured result is recorded here after running
`probes/anthropic_compaction_round.py --target-tokens 1040000 --turns 96` on the
same image used for this review.

<!-- FULL_WINDOW_RESULT_START -->
PASS on the reviewed `0.1.9` image. The native Anthropic request contained
`1,040,105` input tokens (99.1921% of the configured window), returned the
expected structured tool result, and completed in `1,127.003s` without OOM,
queueing, container restart, or protocol-marker leakage.

The compaction request replayed `1,040,118` input tokens, hit `1,039,872`
prefix-cache tokens, recomputed 246 (`99.976349%` hit), preserved all four
anchors, and completed in `11.293s`. Replacing the transcript with its summary
then produced a valid structured tool call from 658 input tokens in `5.579s`.
The three request token counts sum exactly to the post-run
`prompt_tokens_total` delta; the cache-hit metric delta also equals the value
above. Final engine state was running 0, waiting 0, KV usage 0, and `/health`
returned HTTP 200. No runtime ERROR, traceback, exception, or NCCL failure was
logged.
<!-- FULL_WINDOW_RESULT_END -->

## Remaining limitations and rejected changes

- A configured 1M limit is not six-way 1M capacity. Read the startup KV token
  count and maximum-concurrency line for each boot.
- Client-side compaction is not a vLLM feature. vLLM serves/counts Anthropic
  messages; Claude Code or another agent owns transcript replacement.
- Native Anthropic rendering/tokenization and network upload contribute to wall
  time outside GPU prefill/decode. Prefix caching avoids repeated compute, not
  retransmission of a stateless request body.
- The first request for an unseen W4A16 MoE or 1M prefill metadata shape can
  still trigger a one-time CuTeDSL/Triton JIT warning. A same-process repeat did
  not compile the shape again. This is a cold-tail latency gap, not a parser or
  correctness failure; broad speculative warmup changes were not added without
  a pinned provider contract.
- Model Runner V2 now enforces Anthropic `budget_tokens` and the vLLM Chat
  extension `thinking_token_budget`, but forcing a tiny budget can move an
  unfinished thought into visible output or exhaust the request. Protocol-native
  effort remains the recommended quality control; use numeric budgets as hard
  resource boundaries and normally keep them at `1024` or above.
- OpenAI Responses has no standard reasoning-only numeric budget in this pinned
  API. Use `reasoning.effort`; `max_output_tokens` is the combined reasoning and
  visible-output ceiling. Patch `0012` reports the reasoning split from parser
  state rather than estimating it from character counts.
- The open FilteredTopK/softmax changes were not included: this GB10 path does
  not enter the reported long-row kernel for the target 1M workload, and the
  proposed C++ changes lack a sufficient end-to-end quality gate here.
- The reasoning-stop guard was not included because the current native
  Anthropic workload did not provide a reproduced, request-level stop condition
  justifying another generation-path patch.
- Broad MiaAI scheduler/hybrid-cache/spin-wait bundles remain rejected after a
  reproducible third-request stall. Only Issue #22 is retained.

## Rollback contract

Keep the previous Head and Worker containers stopped under explicit names until
the full-window gate and a real agent soak pass. Rollback must use the matching
image on both ranks and preserve Worker-first startup order.
