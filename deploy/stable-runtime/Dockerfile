ARG BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1
FROM ${BASE_IMAGE}

COPY patches/ /tmp/dspark-patches/patches/
COPY upstream-hotfixes/hotfix-nvfp4-ds-mla-issue22.sh /tmp/dspark-patches/
COPY verify_patch.py /tmp/dspark-patches/
COPY verify_nvfp4.py /tmp/dspark-patches/

RUN cd /usr/local/lib/python3.12/dist-packages \
    && for patch_file in /tmp/dspark-patches/patches/*.patch; do \
        patch --batch --forward -p1 < "$patch_file"; \
    done \
    && bash /tmp/dspark-patches/hotfix-nvfp4-ds-mla-issue22.sh \
    && python3 -m py_compile \
        vllm/tokenizers/deepseek_v4_encoding.py \
        vllm/parser/deepseek_v4.py \
        vllm/v1/attention/backends/mla/flashmla_sparse.py \
    && python3 /tmp/dspark-patches/verify_patch.py \
    && python3 /tmp/dspark-patches/verify_nvfp4.py \
    && rm -rf /tmp/dspark-patches

LABEL org.opencontainers.image.title="DeepSeek V4 Flash 0731 stable runtime for dual DGX Spark" \
      org.opencontainers.image.version="0.1.1-stable-nvfp4-20260819" \
      org.opencontainers.image.description="Anemll 0.1.1 plus proven long-agent parser fixes and the isolated NVFP4 MLA kernel fix" \
      org.opencontainers.image.source="https://github.com/vibe-agi/dgxspark_deepseekv4flash0731"
