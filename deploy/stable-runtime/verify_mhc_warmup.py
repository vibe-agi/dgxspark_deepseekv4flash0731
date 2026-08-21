import torch

import vllm.model_executor.warmup.deepseek_v4_mhc_warmup as warmup


class _Norm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(4, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.variance_epsilon = 1e-5


class DeepseekV4DecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 4
        self.hc_mult = 2
        self.rms_norm_eps = 1e-5
        self.hc_eps = 2e-5
        self.hc_post_alpha = 2.0
        self.hc_sinkhorn_iters = 3
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_scale = torch.nn.Parameter(torch.empty(3))
        self.hc_attn_base = torch.nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_scale = torch.nn.Parameter(torch.empty(3))
        self.hc_ffn_base = torch.nn.Parameter(torch.empty(mix_hc))
        self.attn_norm = _Norm()
        self.ffn_norm = _Norm()


root = torch.nn.Module()
root.layer = DeepseekV4DecoderLayer()
assert warmup._find_first_mhc_layer(root) is root.layer

real_ops = warmup._get_tilelang_mhc_ops()
assert tuple(op.__name__ for op in real_ops) == (
    "mhc_pre_tilelang",
    "mhc_fused_post_pre_tilelang",
    "mhc_post_tilelang",
    "hc_head_fused_kernel_tilelang",
)

selected_sizes = warmup._select_mhc_warmup_token_sizes(
    max_tokens=16_384,
    cudagraph_capture_sizes=[24, 32],
)
assert 1 in selected_sizes
assert 24 in selected_sizes
assert 8160 in selected_sizes
assert 16_384 in selected_sizes

calls = []


def fake_pre(residual, *args, **kwargs):
    calls.append(("pre", residual.shape[0]))
    count = residual.shape[0]
    return (
        torch.empty(count, 2),
        torch.empty(count, 2, 2),
        torch.empty(count, 4, dtype=torch.bfloat16),
    )


def fake_fused(layer_input, residual, *args, **kwargs):
    calls.append(("fused", residual.shape[0]))
    count = residual.shape[0]
    return (
        torch.empty_like(residual),
        torch.empty(count, 2),
        torch.empty(count, 2, 2),
        torch.empty(count, 4, dtype=torch.bfloat16),
    )


def fake_post(layer_input, residual, *args):
    calls.append(("post", residual.shape[0]))
    return torch.empty_like(residual)


original_get_ops = warmup._get_tilelang_mhc_ops
try:
    warmup._get_tilelang_mhc_ops = lambda: (
        fake_pre,
        fake_fused,
        fake_post,
        lambda *args: None,
    )
    warmup._warmup_layer_mhc(root.layer, [1, 3])
finally:
    warmup._get_tilelang_mhc_ops = original_get_ops

assert calls == [
    ("pre", 1),
    ("fused", 1),
    ("post", 1),
    ("pre", 3),
    ("fused", 3),
    ("post", 3),
]
