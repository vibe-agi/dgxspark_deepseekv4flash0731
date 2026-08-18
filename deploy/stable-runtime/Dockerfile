ARG BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1
FROM ${BASE_IMAGE}

COPY patches/ /tmp/dspark-patches/patches/
COPY verify_patch.py /tmp/dspark-patches/

RUN cd /usr/local/lib/python3.12/dist-packages \
    && for patch_file in /tmp/dspark-patches/patches/*.patch; do \
        patch --batch --forward -p1 < "$patch_file"; \
    done \
    && python3 -m py_compile \
        vllm/tokenizers/deepseek_v4_encoding.py \
        vllm/parser/deepseek_v4.py \
    && python3 /tmp/dspark-patches/verify_patch.py \
    && rm -rf /tmp/dspark-patches

LABEL org.opencontainers.image.title="DeepSeek V4 Flash 0731 stable runtime for dual DGX Spark" \
      org.opencontainers.image.version="0.1.1-stable-20260818" \
      org.opencontainers.image.description="Anemll 0.1.1 plus DeepSeek-V4 long-history and tolerant DSML parser fixes" \
      org.opencontainers.image.source="https://github.com/vibe-agi/dgxspark_deepseekv4flash0731"
