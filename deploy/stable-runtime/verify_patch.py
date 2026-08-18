from vllm.tokenizers.deepseek_v4_encoding import (
    merge_consecutive_assistant_messages,
)


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
