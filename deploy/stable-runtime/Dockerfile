ARG BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
FROM ${BASE_IMAGE}

COPY patches/ /tmp/dspark-patches/patches/
COPY upstream-hotfixes/hotfix-nvfp4-ds-mla-issue22.sh /tmp/dspark-patches/
COPY verify_patch.py /tmp/dspark-patches/
COPY verify_mhc_warmup.py /tmp/dspark-patches/
COPY verify_dsml_recovery.py /tmp/dspark-patches/
COPY verify_long_tool_args.py /tmp/dspark-patches/
COPY verify_thinking_compat.py /tmp/dspark-patches/
COPY verify_nvfp4.py /tmp/dspark-patches/

RUN cd /usr/local/lib/python3.12/dist-packages \
    && for patch_file in /tmp/dspark-patches/patches/*.patch; do \
        patch --batch --forward --fuzz=0 -p1 < "$patch_file"; \
    done \
    && bash /tmp/dspark-patches/hotfix-nvfp4-ds-mla-issue22.sh \
    && python3 -m py_compile \
        vllm/tokenizers/deepseek_v4_encoding.py \
        vllm/parser/deepseek_v4.py \
        vllm/parser/engine/parser_engine.py \
        vllm/parser/engine/parser_engine_config.py \
        vllm/parser/engine/streaming_parser_engine.py \
        vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py \
        vllm/v1/attention/backends/mla/flashmla_sparse.py \
        vllm/entrypoints/anthropic/protocol.py \
        vllm/entrypoints/anthropic/serving.py \
        vllm/entrypoints/openai/chat_completion/protocol.py \
        vllm/config/reasoning.py \
        vllm/config/vllm.py \
        vllm/v1/engine/input_processor.py \
        vllm/v1/worker/gpu/model_runner.py \
        vllm/v1/worker/gpu/sample/sampler.py \
        vllm/v1/worker/gpu/sample/thinking_budget.py \
        vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
    && python3 /tmp/dspark-patches/verify_patch.py \
    && python3 /tmp/dspark-patches/verify_mhc_warmup.py \
    && python3 /tmp/dspark-patches/verify_dsml_recovery.py \
    && python3 /tmp/dspark-patches/verify_long_tool_args.py \
    && python3 /tmp/dspark-patches/verify_thinking_compat.py \
    && python3 /tmp/dspark-patches/verify_nvfp4.py \
    && rm -rf /tmp/dspark-patches

LABEL org.opencontainers.image.title="DeepSeek V4 Flash 0731 stable runtime for dual DGX Spark" \
      org.opencontainers.image.version="0.1.9-stable-20260821" \
      org.opencontainers.image.description="Pinned Anemll runtime with native Anthropic/OpenAI thinking controls, long-agent DSML streaming, mHC warmup, and NVFP4 MLA fixes" \
      org.opencontainers.image.source="https://github.com/vibe-agi/dgxspark_deepseekv4flash0731"
