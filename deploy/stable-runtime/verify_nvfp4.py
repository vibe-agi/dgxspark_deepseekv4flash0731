from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/attention/backends/mla/flashmla_sparse.py"
)
text = path.read_text(encoding="utf-8")
marker = 'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")'
assert marker in text, f"missing Issue #22 NVFP4 routing marker in {path}"

print("NVFP4 runtime marker verified")
