from inspect import getsource
from types import SimpleNamespace

from pydantic import ValidationError

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState


MESSAGE = [{"role": "user", "content": "Compute 99991 * 88883."}]


def chat_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="deepseek-v4-flash", messages=MESSAGE, **kwargs)


# Standard OpenAI reasoning_effort is authoritative.  The vLLM extension only
# supplies a neutral fallback when it is the sole thinking control.
positive = chat_request(thinking_token_budget=1024)
assert positive.reasoning_effort == "high"
assert positive.thinking_token_budget == 1024
assert positive.build_chat_params(None, "auto").chat_template_kwargs[
    "enable_thinking"
] is True

unlimited = chat_request(thinking_token_budget=-1)
assert unlimited.reasoning_effort == "high"
assert unlimited.thinking_token_budget is None

disabled = chat_request(thinking_token_budget=0)
assert disabled.reasoning_effort == "none"
assert disabled.thinking_token_budget == 0

explicit = chat_request(reasoning_effort="max", thinking_token_budget=1024)
assert explicit.reasoning_effort == "max"
assert explicit.thinking_token_budget == 1024


# Anthropic manual thinking forwards the exact budget and activates thinking.
manual = AnthropicMessagesRequest(
    model="deepseek-v4-flash",
    messages=MESSAGE,
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 1024},
)
manual_chat = AnthropicServingMessages._convert_anthropic_to_openai_request(manual)
assert manual_chat.reasoning_effort == "high"
assert manual_chat.thinking_token_budget == 1024

# Explicit Anthropic effort wins over the neutral manual/adaptive fallback.
manual_max = AnthropicMessagesRequest(
    model="deepseek-v4-flash",
    messages=MESSAGE,
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 1024},
    output_config={"effort": "max"},
)
manual_max_chat = AnthropicServingMessages._convert_anthropic_to_openai_request(
    manual_max
)
assert manual_max_chat.reasoning_effort == "max"
assert manual_max_chat.thinking_token_budget == 1024

adaptive_count = AnthropicCountTokensRequest(
    model="deepseek-v4-flash",
    messages=MESSAGE,
    thinking={"type": "adaptive"},
    output_config={"effort": "max"},
)
adaptive_count_chat = AnthropicServingMessages._convert_anthropic_to_openai_request(
    adaptive_count
)
assert adaptive_count_chat.reasoning_effort == "max"

anthropic_disabled = AnthropicMessagesRequest(
    model="deepseek-v4-flash",
    messages=MESSAGE,
    max_tokens=4096,
    thinking={"type": "disabled"},
)
anthropic_disabled_chat = (
    AnthropicServingMessages._convert_anthropic_to_openai_request(anthropic_disabled)
)
assert anthropic_disabled_chat.reasoning_effort == "none"
assert anthropic_disabled_chat.thinking_token_budget == 0

try:
    AnthropicMessagesRequest(
        model="deepseek-v4-flash",
        messages=MESSAGE,
        max_tokens=4096,
        thinking={"type": "enabled"},
    )
except ValidationError:
    pass
else:
    raise AssertionError("enabled Anthropic thinking must require budget_tokens")


# The pinned DSpark runtime requires Model Runner V2.  The adapted upstream
# implementation must enforce numeric budgets in the GPU sampler rather than
# rejecting or silently clearing them at the input boundary.
validate_source = getsource(InputProcessor._validate_params)
assert "thinking_token_budget is not yet supported" not in validate_source
assert "params.thinking_token_budget = None" not in validate_source

sampler_source = getsource(Sampler.apply_sampling_params)
assert "self.thinking_budget_state.apply" in sampler_source

budget_source = getsource(ThinkingBudgetState)
assert "natural_reasoning_end_token_ids" in budget_source
assert "reasoning_end_token_ids" in budget_source
assert "use_thinking_budget" in budget_source


# Responses usage accounting must start inside a prompt-opened reasoning span.
# Use the real method with a minimal object so this stays tokenizer-independent
# and runs during every image build.
prompt_opened = SimpleNamespace(
    _reasoning_start_token_id=10,
    _reasoning_end_token_id=11,
    parser_engine_config=SimpleNamespace(initial_state=ParserState.REASONING),
)
content_started = SimpleNamespace(
    _reasoning_start_token_id=10,
    _reasoning_end_token_id=11,
    parser_engine_config=SimpleNamespace(initial_state=ParserState.CONTENT),
)
assert ParserEngine.count_reasoning_tokens(prompt_opened, [1, 2, 11, 3]) == 2
assert ParserEngine.count_reasoning_tokens(prompt_opened, [1, 2]) == 2
assert ParserEngine.count_reasoning_tokens(prompt_opened, [10, 1, 2, 11, 3]) == 2
assert ParserEngine.count_reasoning_tokens(content_started, [10, 1, 2, 11, 3]) == 2
assert ParserEngine.count_reasoning_tokens(content_started, [1, 2, 11, 3]) == 0
