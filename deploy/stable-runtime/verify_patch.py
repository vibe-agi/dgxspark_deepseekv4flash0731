from pathlib import Path

from vllm.tokenizers.deepseek_v4_encoding import (
    encode_messages,
    merge_consecutive_assistant_messages,
)
from vllm.parser.deepseek_v4 import _dsml_arg_converter, deepseek_v4_config
from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine


patch_artifacts = [
    path
    for path in Path("vllm").rglob("*")
    if path.is_file() and path.suffix in {".orig", ".rej"}
]
assert not patch_artifacts, f"patch backup/reject artifacts found: {patch_artifacts}"


messages = [
    {"role": "user", "content": "check"},
    {"role": "assistant", "content": "I will check."},
    {
        "role": "assistant",
        "reasoning": "Need a tool.",
        "tool_calls": [{"id": "call_1", "type": "function"}],
        "wo_eos": True,
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
]

merged = merge_consecutive_assistant_messages(messages)
assert [message["role"] for message in merged] == ["user", "assistant", "tool"]
assistant = merged[1]
assert assistant["content"] == "I will check."
assert assistant["reasoning"] == "Need a tool."
assert assistant["tool_calls"] == [{"id": "call_1", "type": "function"}]
assert assistant["wo_eos"] is True


tool_schema = {
    "type": "function",
    "function": {
        "name": "echo_probe",
        "description": "Return a marker.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}
thinking_history = [
    {"role": "system", "tools": [tool_schema]},
    {"role": "user", "content": "Start."},
    {"role": "assistant", "content": "On it."},
    {"role": "user", "content": "Continue."},
]
encoded = encode_messages(
    thinking_history,
    thinking_mode="thinking",
    drop_thinking=True,
)
assert "<think></think>" not in encoded
assert "<｜Assistant｜></think>On it." in encoded
assert encoded.endswith("<｜Assistant｜><think>")

thinking_history[2]["reasoning"] = "Need to continue."
encoded_with_reasoning = encode_messages(
    thinking_history,
    thinking_mode="thinking",
    drop_thinking=True,
)
assert "<think>Need to continue.</think>On it." in encoded_with_reasoning


def parse_types(text: str) -> list[EventType]:
    engine = StreamingParserEngine(deepseek_v4_config(False), None)
    return [event.type for event in engine.parse_complete(text)]


ascii_dsml = (
    '<|DSML|tool_calls><|DSML|invoke name="Bash">'
    '<|DSML|parameter name="command" string="true">git status'
    '</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
)
bare_close_dsml = (
    '<｜DSML｜tool_calls><｜DSML｜invoke name="Bash">'
    '<｜DSML｜parameter name="command" string="true">git status'
    '</parameter></invoke></tool_calls>'
)

for sample in (ascii_dsml, bare_close_dsml):
    event_types = parse_types(sample)
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_END in event_types
    assert EventType.TEXT_CHUNK not in event_types

assert _dsml_arg_converter(
    '<|DSML|parameter name="command" string="true">git status</|DSML|parameter>',
    False,
) == '{"command": "git status"}'
assert _dsml_arg_converter(
    '<｜DSML｜parameter name="command" string="true">git status</parameter>',
    False,
) == '{"command": "git status"}'
