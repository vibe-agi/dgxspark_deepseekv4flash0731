#!/usr/bin/env python3
"""Correctness probe for the dual-DGX DeepSeek-V4 endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


BANNED_OUTPUT = (
    "<antml:",
    "<｜DSML｜",
    "<think>",
    "</think>",
)


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error
    return body, time.monotonic() - started


def clean_text(text: str, marker: str) -> None:
    if marker not in text:
        raise AssertionError(f"missing marker {marker!r}: {text[:500]!r}")
    leaked = [token for token in BANNED_OUTPUT if token in text]
    if leaked:
        raise AssertionError(f"protocol markup leaked: {leaked}: {text[:500]!r}")


def build_split_history(rounds: int, filler_repetitions: int, marker: str):
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Preserve the exact marker {marker}. Process the checks.",
        }
    ]
    for index in range(rounds):
        call_id = f"call_{index:03d}"
        filler = (f" evidence-{index:03d}" * filler_repetitions).strip()
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Starting check {index}.",
                },
                {
                    "role": "assistant",
                    "reasoning": f"Use record for check {index}.",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "record",
                                "arguments": json.dumps({"index": index}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"check {index} completed; {filler}",
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": f"Reply with exactly this marker and nothing else: {marker}",
        }
    )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--filler-repetitions", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    responses_marker = "RESPONSES_STABLE_OK_4C2A"
    response, elapsed = post_json(
        args.base_url,
        "/v1/responses",
        {
            "model": args.model,
            "input": f"Reply with exactly: {responses_marker}",
            "max_output_tokens": 64,
            "temperature": 0,
        },
        args.timeout,
    )
    response_text = response["output"][0]["content"][0]["text"]
    clean_text(response_text, responses_marker)
    print(
        json.dumps(
            {
                "case": "responses",
                "elapsed_seconds": round(elapsed, 3),
                "input_tokens": response["usage"]["input_tokens"],
                "output": response_text,
            },
            ensure_ascii=False,
        )
    )

    history_marker = "SPLIT_HISTORY_STABLE_OK_9F7C"
    payload = {
        "model": args.model,
        "messages": build_split_history(
            args.rounds, args.filler_repetitions, history_marker
        ),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "description": "Record a completed synthetic check.",
                    "parameters": {
                        "type": "object",
                        "properties": {"index": {"type": "integer"}},
                        "required": ["index"],
                    },
                },
            }
        ],
        "tool_choice": "none",
        "max_tokens": 64,
        "temperature": 0,
    }

    def run_chat(case: str):
        result, duration = post_json(
            args.base_url, "/v1/chat/completions", payload, args.timeout
        )
        text = result["choices"][0]["message"]["content"] or ""
        clean_text(text, history_marker)
        return {
            "case": case,
            "elapsed_seconds": round(duration, 3),
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "output": text,
        }

    for index in range(args.repeat):
        print(json.dumps(run_chat(f"split_history_serial_{index + 1}")))

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(run_chat, f"split_history_concurrent_{index + 1}")
                for index in range(args.concurrency)
            ]
            for future in futures:
                print(json.dumps(future.result()))


if __name__ == "__main__":
    main()
