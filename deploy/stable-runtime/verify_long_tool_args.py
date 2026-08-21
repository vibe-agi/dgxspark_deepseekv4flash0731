#!/usr/bin/env python3
"""Validate linear, JSON-safe streaming of large DeepSeek V4 tool arguments."""

import json
import time
from types import SimpleNamespace

from openai.types.responses import FunctionTool

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.engine.parser_engine import ToolCallSlot


class DummyTokenizer:
    all_special_tokens = []
    all_special_ids = []

    @staticmethod
    def get_vocab():
        return {}


def stream_string_argument(
    body: str,
    *,
    tool_name: str = "emit_text",
    parameter_name: str = "text",
    parameter_schema: dict | None = None,
    chunk_size: int = 32,
    tail_chunk_size: int | None = None,
):
    schema = parameter_schema or {"type": "string"}
    tool = FunctionTool(
        type="function",
        name=tool_name,
        parameters={
            "type": "object",
            "properties": {parameter_name: schema},
            "required": [parameter_name],
        },
    )
    parser = DeepSeekV4Parser(DummyTokenizer(), tools=[tool])
    request = SimpleNamespace(tools=[tool], tool_choice="auto")
    header = (
        "<｜DSML｜tool_calls>\n"
        f'<｜DSML｜invoke name="{tool_name}">\n'
        f'<｜DSML｜parameter name="{parameter_name}" string="true">'
    )
    tail = "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    chunks = [header]
    chunks.extend(body[i : i + chunk_size] for i in range(0, len(body), chunk_size))
    if tail_chunk_size is None:
        chunks.append(tail)
    else:
        chunks.extend(
            tail[i : i + tail_chunk_size]
            for i in range(0, len(tail), tail_chunk_size)
        )

    argument_deltas: list[tuple[int, str]] = []
    previous = ""
    start = time.perf_counter()
    for index, delta_text in enumerate(chunks):
        current = previous + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous = current
        if delta:
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                if function and function.arguments is not None:
                    argument_deltas.append((index, function.arguments))
    finish = parser.finish_streaming()
    if finish:
        for tool_call in finish.tool_calls or []:
            function = tool_call.function
            if function and function.arguments is not None:
                argument_deltas.append((len(chunks), function.arguments))
    return argument_deltas, time.perf_counter() - start, len(chunks)


def stream_tool_name(raw_name: str):
    tool = FunctionTool(
        type="function",
        name="save_compaction",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )
    parser = DeepSeekV4Parser(DummyTokenizer(), tools=[tool])
    request = SimpleNamespace(tools=[tool], tool_choice="auto")
    text = (
        "<｜DSML｜tool_calls>"
        f'<｜DSML｜invoke name="{raw_name}">'
        '<｜DSML｜parameter name="summary" string="true">ok'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    )
    name_deltas: list[str] = []
    argument_deltas: list[str] = []
    previous = ""
    for offset in range(0, len(text), 3):
        delta_text = text[offset : offset + 3]
        current = previous + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous = current
        if delta:
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                if function and function.name:
                    name_deltas.append(function.name)
                if function and function.arguments is not None:
                    argument_deltas.append(function.arguments)
    finish = parser.finish_streaming()
    if finish:
        for tool_call in finish.tool_calls or []:
            function = tool_call.function
            if function and function.name:
                name_deltas.append(function.name)
            if function and function.arguments is not None:
                argument_deltas.append(function.arguments)
    return "".join(name_deltas), "".join(argument_deltas)


def main():
    # The fast path keeps emitted JSON fragments in a list. A complete prefix
    # is copied only when a structural edge requires the generic converter.
    slot = ToolCallSlot()
    slot.streamed_json = '{"text":"'
    for _ in range(1024):
        slot.append_streamed_json("x" * 1024)
    assert slot.streamed_json == '{"text":"'
    assert slot.streamed_json_parts is not None
    assert slot.materialize_streamed_json() == '{"text":"' + ("x" * 1024 * 1024)
    assert slot.streamed_json_parts is None

    body_4k = "A" * 4096
    deltas, _, chunk_count = stream_string_argument(body_4k)
    before_close = [value for index, value in deltas if index < chunk_count - 1 and value]
    assert len(before_close) > 10, len(before_close)
    assert json.loads("".join(value for _, value in deltas)) == {"text": body_4k}

    # Exercise the worst lexer boundary: every closing marker arrives one
    # character at a time after the long-string fast path has been entered.
    deltas, _, _ = stream_string_argument(body_4k, tail_chunk_size=1)
    assert json.loads("".join(value for _, value in deltas)) == {"text": body_4k}

    body_128k = "B" * (128 * 1024)
    deltas, elapsed, _ = stream_string_argument(body_128k)
    assert elapsed < 5.0, f"128 KiB stream took {elapsed:.3f}s"
    assert json.loads("".join(value for _, value in deltas)) == {"text": body_128k}

    body_1m = "C" * (1024 * 1024)
    deltas, elapsed, _ = stream_string_argument(body_1m, chunk_size=64)
    assert elapsed < 10.0, f"1 MiB stream took {elapsed:.3f}s"
    assert json.loads("".join(value for _, value in deltas)) == {"text": body_1m}

    line = 'def foo():\n    msg = "hello \\ world"\t# comment\x00\n'
    multiline = line * 2048
    deltas, elapsed, _ = stream_string_argument(
        multiline,
        tool_name="emit_code",
        parameter_name="code",
    )
    assert elapsed < 5.0, f"multiline stream took {elapsed:.3f}s"
    assert json.loads("".join(value for _, value in deltas)) == {"code": multiline}

    markup = '<div data-kind="sample">x < y</div>\n' * 4096
    deltas, elapsed, _ = stream_string_argument(
        markup,
        tool_name="emit_markup",
        parameter_name="html",
    )
    assert elapsed < 5.0, f"markup stream took {elapsed:.3f}s"
    assert json.loads("".join(value for _, value in deltas)) == {"html": markup}

    union_text = "not-null query text"
    deltas, _, chunk_count = stream_string_argument(
        union_text,
        tool_name="search",
        parameter_name="query",
        parameter_schema={"type": ["string", "null"]},
        chunk_size=4,
    )
    premature_values = [
        value for index, value in deltas if index < chunk_count - 1 and value
    ]
    assert "query text" not in "".join(premature_values)
    assert json.loads("".join(value for _, value in deltas)) == {
        "query": union_text
    }

    # A real long-context sample placed a strict prefix of the outer marker
    # inside the following invoke name. Only a declared suffix may be repaired;
    # arbitrary contaminated or undeclared names remain suppressed.
    name, arguments = stream_tool_name("<｜DSML｜tool\n\n\nsave_compaction")
    assert name == "save_compaction", name
    assert json.loads(arguments) == {"summary": "ok"}, arguments
    invalid_name, invalid_args = stream_tool_name("<garbage\n\n\nsave_compaction")
    assert invalid_name == "", invalid_name
    assert invalid_args == "", invalid_args

    print("DeepSeek V4 long tool argument verification passed")


if __name__ == "__main__":
    main()
