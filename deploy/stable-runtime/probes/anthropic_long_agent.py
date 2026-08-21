#!/usr/bin/env python3
"""Exercise native Anthropic history + tool use at a large context size."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


TOOL = {
    "name": "echo_probe",
    "description": "Return the supplied text",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def build_messages(repeats: int, turns: int, marker: str) -> list[dict]:
    unit = "repository evidence datum remains relevant to the requested change. "
    filler = unit * repeats
    messages: list[dict] = []
    for turn in range(turns):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Evidence batch {turn}: {filler}",
                },
                {
                    "role": "assistant",
                    "content": f"Recorded evidence batch {turn}; continue.",
                },
            ]
        )

    messages.extend(
        [
            {"role": "user", "content": "Call echo_probe with historical-check."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_historical_probe",
                        "name": "echo_probe",
                        "input": {"text": "historical-check"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_historical_probe",
                        "content": "historical-check",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The historical tool result was recorded.",
            },
            {
                "role": "user",
                "content": (
                    f"Now call echo_probe with text {marker}. "
                    "Do not answer in prose."
                ),
            },
        ]
    )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--target-tokens", type=int, default=240_000)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--marker", default="LONG-AGENT-TOOL-OK")
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="max",
    )
    args = parser.parse_args()

    repeats = max(1, args.target_tokens // (args.turns * 12))
    for _ in range(6):
        messages = build_messages(repeats, args.turns, args.marker)
        count = post(
            f"{args.base_url}/messages/count_tokens",
            {"model": args.model, "messages": messages, "tools": [TOOL]},
        )["input_tokens"]
        if abs(count - args.target_tokens) <= 1000:
            break
        repeats = max(1, round(repeats * args.target_tokens / count))
    else:
        raise RuntimeError("could not converge on requested context length")

    body = {
        "model": args.model,
        "max_tokens": 128,
        "temperature": 0,
        "output_config": {"effort": args.effort},
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "echo_probe"},
        "messages": messages,
    }
    started = time.perf_counter()
    response = post(f"{args.base_url}/messages", body)
    elapsed = time.perf_counter() - started
    raw = json.dumps(response, ensure_ascii=False)

    forbidden = ("<|DSML|", "</parameter>", "<antml:", "<｜DSML｜")
    assert not any(token in raw for token in forbidden), raw
    tool_uses = [block for block in response.get("content", []) if block.get("type") == "tool_use"]
    assert len(tool_uses) == 1, raw
    assert tool_uses[0]["name"] == "echo_probe", raw
    assert tool_uses[0]["input"] == {"text": args.marker}, raw

    print(
        json.dumps(
            {
                "status": "PASS",
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": response["usage"]["output_tokens"],
                "elapsed_s": round(elapsed, 3),
                "effort": args.effort,
                "tool": tool_uses[0],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
