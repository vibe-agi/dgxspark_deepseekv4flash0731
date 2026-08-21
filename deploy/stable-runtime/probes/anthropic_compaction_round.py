#!/usr/bin/env python3
"""Validate a native Anthropic 1M-context request and one compaction round.

The compaction step is deliberately client-side: vLLM serves the Anthropic
protocol, while an agent client owns when a long transcript is summarized and
replaced.  This probe asks the model to distill the full transcript into a
structured summary, then starts a small follow-up context from that summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


ECHO_TOOL = {
    "name": "echo_probe",
    "description": "Return the supplied text exactly",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}

COMPACT_TOOL = {
    "name": "save_compaction",
    "description": (
        "Save a concise replacement context that preserves the requested "
        "anchors and the current task state"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "anchors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "anchors"],
    },
}

# Keep the inventory and ordering identical across the full-context request and
# the compaction request. Real agent clients do this, and prompt-prefix caching
# cannot hit if the tool schema at the beginning of the prompt changes.
AGENT_TOOLS = [ECHO_TOOL, COMPACT_TOOL]

FORBIDDEN_MARKERS = ("<|DSML|", "<｜DSML｜", "<antml:", "</parameter>")


def post(url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:2000]}") from error


def build_base_messages(repeats: int, turns: int, anchors: list[str]) -> list[dict]:
    unit = "repository evidence datum remains relevant to the requested change. "
    filler = unit * repeats
    anchor_turns = {
        0: anchors[0],
        turns // 3: anchors[1],
        (2 * turns) // 3: anchors[2],
        turns - 1: anchors[3],
    }
    messages: list[dict] = []
    for turn in range(turns):
        anchor = anchor_turns.get(turn)
        anchor_text = f" Durable anchor: {anchor}." if anchor else ""
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Evidence batch {turn}.{anchor_text} {filler}",
                },
                {
                    "role": "assistant",
                    "content": f"Recorded evidence batch {turn}; continue.",
                },
            ]
        )
    return messages


def tool_input(response: dict, expected_name: str) -> dict:
    raw = json.dumps(response, ensure_ascii=False)
    assert not any(marker in raw for marker in FORBIDDEN_MARKERS), raw[:4000]
    uses = [block for block in response.get("content", []) if block.get("type") == "tool_use"]
    assert len(uses) == 1, raw[:4000]
    assert uses[0].get("name") == expected_name, raw[:4000]
    value = uses[0].get("input")
    assert isinstance(value, dict), raw[:4000]
    return value


def timed_post(url: str, body: dict, timeout: int) -> tuple[dict, float]:
    started = time.perf_counter()
    response = post(url, body, timeout)
    return response, time.perf_counter() - started


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--target-tokens", type=int, default=1_040_000)
    parser.add_argument("--turns", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="max")
    args = parser.parse_args()

    if args.target_tokens + 2048 > 1_048_576:
        raise ValueError("target tokens must leave at least 2048 output tokens")
    if args.turns < 6:
        raise ValueError("turns must be at least 6")

    anchors = [
        "CTX-ANCHOR-ALPHA-17",
        "CTX-ANCHOR-BRAVO-29",
        "CTX-ANCHOR-CHARLIE-43",
        "CTX-ANCHOR-DELTA-61",
    ]
    compact_instruction = {
        "role": "user",
        "content": (
            "Compact this transcript now. Call save_compaction exactly once. "
            "The summary must preserve the current evidence-recording task, "
            "and anchors must contain all four durable anchor strings exactly."
        ),
    }

    repeats = max(1, args.target_tokens // (args.turns * 12))
    count = 0
    count_rounds: list[dict] = []
    for round_index in range(5):
        base_messages = build_base_messages(repeats, args.turns, anchors)
        compact_messages = [*base_messages, compact_instruction]
        count_started = time.perf_counter()
        count = post(
            f"{args.base_url}/messages/count_tokens",
            {
                "model": args.model,
                "messages": compact_messages,
                "tools": AGENT_TOOLS,
            },
            args.timeout,
        )["input_tokens"]
        count_elapsed = time.perf_counter() - count_started
        count_rounds.append(
            {
                "round": round_index + 1,
                "input_tokens": count,
                "elapsed_s": round(count_elapsed, 3),
            }
        )
        print(
            f"count_tokens round={round_index + 1} input={count} "
            f"elapsed={count_elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        if abs(count - args.target_tokens) <= 512:
            break
        repeats = max(1, round(repeats * args.target_tokens / count))
    else:
        raise RuntimeError(
            f"could not converge on {args.target_tokens} input tokens; last={count}"
        )

    full_marker = "FULL-CONTEXT-TOOL-OK"
    full_messages = [
        *base_messages,
        {
            "role": "user",
            "content": (
                f"Call echo_probe with text {full_marker}. Do not answer in prose."
            ),
        },
    ]
    progress("phase=full_context start")
    full_response, full_elapsed = timed_post(
        f"{args.base_url}/messages",
        {
            "model": args.model,
            "max_tokens": 256,
            "temperature": 0,
            "output_config": {"effort": args.effort},
            "tools": AGENT_TOOLS,
            "tool_choice": {"type": "tool", "name": "echo_probe"},
            "messages": full_messages,
        },
        args.timeout,
    )
    progress(f"phase=full_context done elapsed={full_elapsed:.3f}s")
    full_input = tool_input(full_response, "echo_probe")
    assert full_input == {"text": full_marker}, full_input

    progress("phase=compaction start")
    compact_response, compact_elapsed = timed_post(
        f"{args.base_url}/messages",
        {
            "model": args.model,
            "max_tokens": 2048,
            "temperature": 0,
            "output_config": {"effort": args.effort},
            "tools": AGENT_TOOLS,
            "tool_choice": {"type": "tool", "name": "save_compaction"},
            "messages": compact_messages,
        },
        args.timeout,
    )
    progress(f"phase=compaction done elapsed={compact_elapsed:.3f}s")
    compacted = tool_input(compact_response, "save_compaction")
    summary = compacted.get("summary")
    retained = compacted.get("anchors")
    assert isinstance(summary, str) and summary.strip(), compacted
    assert isinstance(retained, list), compacted
    missing = [anchor for anchor in anchors if anchor not in retained]
    assert not missing, {"missing_anchors": missing, "compacted": compacted}

    post_marker = "POST-COMPACT-TOOL-OK"
    post_messages = [
        {
            "role": "user",
            "content": (
                "Continue from this compacted context:\n"
                + json.dumps(compacted, ensure_ascii=False)
            ),
        },
        {
            "role": "assistant",
            "content": "The compacted state is loaded; ready for the next task.",
        },
        {
            "role": "user",
            "content": (
                f"Call echo_probe with text {post_marker}. Do not answer in prose."
            ),
        },
    ]
    progress("phase=post_compact start")
    post_response, post_elapsed = timed_post(
        f"{args.base_url}/messages",
        {
            "model": args.model,
            "max_tokens": 256,
            "temperature": 0,
            "output_config": {"effort": args.effort},
            "tools": AGENT_TOOLS,
            "tool_choice": {"type": "tool", "name": "echo_probe"},
            "messages": post_messages,
        },
        args.timeout,
    )
    progress(f"phase=post_compact done elapsed={post_elapsed:.3f}s")
    post_input = tool_input(post_response, "echo_probe")
    assert post_input == {"text": post_marker}, post_input

    print(
        json.dumps(
            {
                "status": "PASS",
                "target_tokens": args.target_tokens,
                "counted_compact_input_tokens": count,
                "count_tokens": count_rounds,
                "full_context": {
                    "input_tokens": full_response["usage"]["input_tokens"],
                    "output_tokens": full_response["usage"]["output_tokens"],
                    "elapsed_s": round(full_elapsed, 3),
                },
                "compaction": {
                    "input_tokens": compact_response["usage"]["input_tokens"],
                    "output_tokens": compact_response["usage"]["output_tokens"],
                    "elapsed_s": round(compact_elapsed, 3),
                    "summary_chars": len(summary),
                    "anchors": retained,
                },
                "post_compact": {
                    "input_tokens": post_response["usage"]["input_tokens"],
                    "output_tokens": post_response["usage"]["output_tokens"],
                    "elapsed_s": round(post_elapsed, 3),
                },
                "effort": args.effort,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
