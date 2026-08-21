#!/usr/bin/env python3
"""Focused regression checks for the pinned DeepSeek V4 parser backport."""

from vllm.parser.deepseek_v4 import deepseek_v4_config
from vllm.parser.engine.events import EventType
import vllm.parser.engine.streaming_parser_engine as streaming_parser
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine


U_TOOL_START = "<｜DSML｜tool_calls>"
U_TOOL_END = "</｜DSML｜tool_calls>"
U_INVOKE = '<｜DSML｜invoke name="'
U_INVOKE_END = "</｜DSML｜invoke>"
U_PARAM = (
    '<｜DSML｜parameter name="command" string="true">'
    "echo ok</｜DSML｜parameter>"
)
A_TOOL_START = "<|DSML|tool_calls>"
A_TOOL_END = "</|DSML|tool_calls>"
A_INVOKE = '<|DSML|invoke name="'
A_INVOKE_END = "</|DSML|invoke>"
A_PARAM = (
    '<|DSML|parameter name="command" string="true">'
    "echo ok</|DSML|parameter>"
)
NAME_END = '">'


def parse(text: str, *, thinking: bool = False, valid=("shell",)):
    engine = StreamingParserEngine(deepseek_v4_config(thinking), None)
    engine.recovery_tool_name_validator = lambda name: name in valid
    return engine.parse_complete(text)


def parse_chunks(chunks, *, thinking: bool = False, valid=("shell",)):
    engine = StreamingParserEngine(deepseek_v4_config(thinking), None)
    engine.recovery_tool_name_validator = lambda name: name in valid
    events = []
    for chunk in chunks:
        events.extend(engine.feed(chunk, []))
    events.extend(engine.finish())
    return events


def signature(events):
    return [(event.type.name, event.value, event.tool_index) for event in events]


def assert_one_tool(events, name="shell", *, tail=None, allow_dsml_text=False):
    sig = signature(events)
    types = [item[0] for item in sig]
    assert types.count("TOOL_CALL_START") == 1, sig
    assert types.count("TOOL_CALL_END") == 1, sig
    streamed_name = "".join(v for t, v, _ in sig if t == "TOOL_NAME")
    assert streamed_name == name, sig
    streamed_args = "".join(v for t, v, _ in sig if t == "ARG_VALUE_CHUNK")
    assert "echo ok" in streamed_args, sig
    text = "".join(v for t, v, _ in sig if t == "TEXT_CHUNK")
    if not allow_dsml_text:
        assert "DSML" not in text, sig
    if tail is not None:
        assert text == tail, sig


def main():
    canonical = U_TOOL_START + U_INVOKE + "shell" + NAME_END + U_PARAM
    canonical += U_INVOKE_END + U_TOOL_END
    assert_one_tool(parse(canonical))

    recovered = U_INVOKE + "shell" + NAME_END + U_PARAM + U_INVOKE_END
    assert_one_tool(parse(recovered + U_TOOL_END + "tail"), tail="tail")

    # The wire stream may split a marker at any character boundary. Verify
    # both canonical and recovered calls without relying on favorable SSE
    # chunking from the model server.
    for sample, tail in (
        (canonical + "tail", "tail"),
        (recovered + U_TOOL_END + "tail", "tail"),
    ):
        for split in range(len(sample) + 1):
            assert_one_tool(parse_chunks((sample[:split], sample[split:])), tail=tail)

    # Once real prose appears, a later closer is no longer part of the
    # recovered invocation and must be preserved as text.
    late_close_events = parse(recovered + "tail" + U_TOOL_END)
    assert_one_tool(
        late_close_events,
        tail="tail" + U_TOOL_END,
        allow_dsml_text=True,
    )

    recovered_ascii = A_INVOKE + "shell" + NAME_END + A_PARAM + A_INVOKE_END
    assert_one_tool(parse(recovered_ascii + A_TOOL_END + "tail"), tail="tail")

    recovered_bare_end = U_INVOKE + "shell" + NAME_END + U_PARAM + "</invoke>"
    assert_one_tool(parse(recovered_bare_end + "tail"), tail="tail")

    reasoning_events = parse("analysis" + recovered, thinking=True)
    assert_one_tool(reasoning_events)
    reasoning_types = [event.type for event in reasoning_events]
    assert EventType.REASONING_CHUNK in reasoning_types, signature(reasoning_events)
    assert EventType.REASONING_END in reasoning_types, signature(reasoning_events)

    two = recovered + recovered_ascii + U_TOOL_END + "tail"
    two_events = parse(two)
    assert [e.type for e in two_events].count(EventType.TOOL_CALL_START) == 2
    assert [e.type for e in two_events].count(EventType.TOOL_CALL_END) == 2
    assert "".join(e.value for e in two_events if e.type == EventType.TEXT_CHUNK) == "tail"

    unknown = U_INVOKE + "not_declared" + NAME_END + U_PARAM + U_INVOKE_END
    unknown_events = parse(unknown)
    assert not any(e.type == EventType.TOOL_CALL_START for e in unknown_events)
    assert "".join(e.value for e in unknown_events) == unknown
    for split in range(len(unknown) + 1):
        split_events = parse_chunks((unknown[:split], unknown[split:]))
        assert not any(e.type == EventType.TOOL_CALL_START for e in split_events)
        assert "".join(e.value for e in split_events) == unknown

    incomplete = U_INVOKE + "shell" + NAME_END + U_PARAM
    incomplete_events = parse(incomplete)
    assert not any(e.type == EventType.TOOL_CALL_START for e in incomplete_events)
    assert "".join(e.value for e in incomplete_events) == incomplete

    wrong_close = U_INVOKE + "shell" + NAME_END + U_PARAM + U_TOOL_END
    wrong_events = parse(wrong_close)
    assert not any(e.type == EventType.TOOL_CALL_START for e in wrong_events)
    assert "".join(e.value for e in wrong_events) == wrong_close

    newline_name = U_INVOKE + "shell\nordinary prose"
    newline_events = parse(newline_name)
    assert not any(e.type == EventType.TOOL_CALL_START for e in newline_events)
    assert "".join(e.value for e in newline_events) == newline_name

    # Real model samples emitted several truncated wrapper lengths immediately
    # before valid calls. Every strict prefix of either marker is handled by
    # the same bounded state rather than by one literal special case.
    fragments = (
        "<｜DSML｜\n\n",
        "<｜DSML｜tool\n",
        "<｜DSML｜tool_c\n\n",
        "<｜DSML｜inv\n",
        "<|DSML|\n\n",
        "<|DSML|tool_c\n",
        "<|DSML|inv\n",
    )
    for fragment in fragments:
        fragment_events = parse(fragment + canonical)
        assert_one_tool(fragment_events)
        assert not any(fragment in e.value for e in fragment_events)

        # Do not eat prose just because it resembles the beginning of DSML.
        fragment_only = fragment + "ordinary text"
        fragment_only_events = parse(fragment_only)
        assert "".join(e.value for e in fragment_only_events) == fragment_only

    # Fragmented streaming deltas must behave the same as one complete delta.
    chunked_engine = StreamingParserEngine(deepseek_v4_config(False), None)
    chunked_engine.recovery_tool_name_validator = lambda name: name == "shell"
    chunked_events = []
    for chunk in ("<｜D", "SML｜", "\n", "\n", canonical):
        chunked_events.extend(chunked_engine.feed(chunk, []))
    chunked_events.extend(chunked_engine.finish())
    assert_one_tool(chunked_events)

    # An attacker-controlled whitespace stream cannot grow the hold buffer
    # without bound. Once the cap is crossed, the original text is preserved.
    oversized = "<｜DSML｜" + (" " * 5000)
    oversized_events = parse(oversized + canonical)
    assert_one_tool(oversized_events, allow_dsml_text=True)
    oversized_text = "".join(
        e.value for e in oversized_events if e.type == EventType.TEXT_CHUNK
    )
    assert oversized_text == oversized

    unrelated = "ordinary <｜DSML｜ discussion"
    assert "".join(e.value for e in parse(unrelated)) == unrelated

    # Provisional recovery is lossless and bounded. Use a tiny cap here to
    # exercise the abort path without allocating the production 8 MiB limit.
    original_cap = streaming_parser._MAX_RECOVERY_HOLD_CHARS
    streaming_parser._MAX_RECOVERY_HOLD_CHARS = 64
    try:
        over_cap = U_INVOKE + "shell" + NAME_END + ("x" * 80)
        over_cap_events = parse(over_cap)
    finally:
        streaming_parser._MAX_RECOVERY_HOLD_CHARS = original_cap
    assert not any(e.type == EventType.TOOL_CALL_START for e in over_cap_events)
    assert "".join(e.value for e in over_cap_events) == over_cap

    print("DeepSeek V4 DSML recovery verification passed")


if __name__ == "__main__":
    main()
