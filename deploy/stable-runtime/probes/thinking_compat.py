#!/usr/bin/env python3
"""Exercise native Anthropic and OpenAI-compatible thinking controls."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any


PROMPT = "Compute 99991 * 88883. Return only the integer in the final answer."
EXPECTED = "8887500053"


def post(base_url: str, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {exc.read().decode()}") from exc
    return payload, time.monotonic() - started


def post_stream(
    base_url: str,
    path: str,
    body: dict[str, Any],
    protocol: str,
) -> tuple[str, str, list[str], float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps({**body, "stream": True}).encode(),
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    started = time.monotonic()
    thinking: list[str] = []
    text: list[str] = []
    event_types: set[str] = set()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                event_type = event.get("type") or event.get("object")
                if event_type:
                    event_types.add(str(event_type))
                if event.get("type") == "error" or event.get("error") is not None:
                    raise RuntimeError(f"{path} returned an SSE error: {event}")

                if protocol == "anthropic":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "thinking_delta":
                        thinking.append(delta.get("thinking", ""))
                    elif delta.get("type") == "text_delta":
                        text.append(delta.get("text", ""))
                elif protocol == "chat":
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        thinking.append(
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                        text.append(delta.get("content") or "")
                elif protocol == "responses":
                    if event.get("type") == "response.reasoning_text.delta":
                        thinking.append(event.get("delta", ""))
                    elif event.get("type") == "response.output_text.delta":
                        text.append(event.get("delta", ""))
                else:
                    raise ValueError(f"unknown streaming protocol: {protocol}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{path} returned HTTP {exc.code}: {exc.read().decode()}"
        ) from exc

    return (
        "".join(thinking),
        "".join(text),
        sorted(event_types),
        time.monotonic() - started,
    )


def anthropic_parts(payload: dict[str, Any]) -> tuple[str, str]:
    thinking = "".join(
        block.get("thinking", "")
        for block in payload.get("content", [])
        if block.get("type") == "thinking"
    )
    text = "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )
    return thinking, text


def chat_parts(payload: dict[str, Any]) -> tuple[str, str]:
    message = payload["choices"][0]["message"]
    return message.get("reasoning") or message.get("reasoning_content") or "", message.get(
        "content"
    ) or ""


def responses_parts(payload: dict[str, Any]) -> tuple[str, str]:
    thinking: list[str] = []
    text: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") == "reasoning":
            for part in item.get("content") or []:
                thinking.append(part.get("text", ""))
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text.append(part.get("text", ""))
    return "".join(thinking), "".join(text)


def result(name: str, payload: dict[str, Any], elapsed: float, parts) -> dict[str, Any]:
    thinking, text = parts(payload)
    digits = "".join(re.findall(r"\d", text))
    return {
        "case": name,
        "elapsed_seconds": round(elapsed, 3),
        "thinking_chars": len(thinking),
        "text": text,
        "answer_ok": EXPECTED in digits,
        "usage": payload.get("usage"),
    }


def stream_result(
    name: str,
    thinking: str,
    text: str,
    event_types: list[str],
    elapsed: float,
) -> dict[str, Any]:
    digits = "".join(re.findall(r"\d", text))
    return {
        "case": name,
        "elapsed_seconds": round(elapsed, 3),
        "thinking_chars": len(thinking),
        "text": text,
        "answer_ok": EXPECTED in digits,
        "stream": True,
        "event_types": event_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    common_anthropic = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 2048,
        "temperature": 0,
    }
    common_chat = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": 2048,
        "temperature": 0,
    }

    cases: list[dict[str, Any]] = []

    payload, elapsed = post(
        args.base_url,
        "/v1/messages",
        {
            **common_anthropic,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )
    cases.append(result("anthropic-manual-budget", payload, elapsed, anthropic_parts))

    payload, elapsed = post(
        args.base_url,
        "/v1/messages",
        {
            **common_anthropic,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
    )
    cases.append(result("anthropic-adaptive-max", payload, elapsed, anthropic_parts))

    payload, elapsed = post(
        args.base_url,
        "/v1/chat/completions",
        {**common_chat, "thinking_token_budget": 1024},
    )
    cases.append(result("openai-chat-budget", payload, elapsed, chat_parts))

    payload, elapsed = post(
        args.base_url,
        "/v1/chat/completions",
        {**common_chat, "reasoning_effort": "max"},
    )
    cases.append(result("openai-chat-max", payload, elapsed, chat_parts))

    payload, elapsed = post(
        args.base_url,
        "/v1/responses",
        {
            "model": args.model,
            "input": PROMPT,
            "max_output_tokens": 2048,
            "temperature": 0,
            "reasoning": {"effort": "max"},
        },
    )
    cases.append(result("openai-responses-max", payload, elapsed, responses_parts))

    thinking, text, event_types, elapsed = post_stream(
        args.base_url,
        "/v1/messages",
        {
            **common_anthropic,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        "anthropic",
    )
    cases.append(
        stream_result(
            "anthropic-adaptive-max-stream",
            thinking,
            text,
            event_types,
            elapsed,
        )
    )

    thinking, text, event_types, elapsed = post_stream(
        args.base_url,
        "/v1/chat/completions",
        {**common_chat, "reasoning_effort": "max"},
        "chat",
    )
    cases.append(
        stream_result(
            "openai-chat-max-stream", thinking, text, event_types, elapsed
        )
    )

    thinking, text, event_types, elapsed = post_stream(
        args.base_url,
        "/v1/responses",
        {
            "model": args.model,
            "input": PROMPT,
            "max_output_tokens": 2048,
            "temperature": 0,
            "reasoning": {"effort": "max"},
        },
        "responses",
    )
    cases.append(
        stream_result(
            "openai-responses-max-stream",
            thinking,
            text,
            event_types,
            elapsed,
        )
    )

    count_payload, count_elapsed = post(
        args.base_url,
        "/v1/messages/count_tokens",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
    )

    report = {
        "cases": cases,
        "count_tokens": count_payload,
        "count_elapsed_seconds": round(count_elapsed, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = [case["case"] for case in cases if not case["answer_ok"]]
    no_thinking = [case["case"] for case in cases if case["thinking_chars"] == 0]
    responses_usage = next(
        case["usage"]
        for case in cases
        if case["case"] == "openai-responses-max"
    )
    reasoning_tokens = (
        (responses_usage or {}).get("output_tokens_details") or {}
    ).get("reasoning_tokens", 0)
    if failed or no_thinking or reasoning_tokens <= 0:
        raise SystemExit(
            "thinking compatibility gate failed: "
            f"bad_answer={failed}, no_thinking={no_thinking}, "
            f"responses_reasoning_tokens={reasoning_tokens}"
        )


if __name__ == "__main__":
    main()
