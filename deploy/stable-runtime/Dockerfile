ARG BASE_IMAGE=ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1
FROM ${BASE_IMAGE}

COPY patches/0001-merge-consecutive-assistant-messages.patch /tmp/dspark-patches/
COPY verify_patch.py /tmp/dspark-patches/

RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch --batch --forward -p1 \
        < /tmp/dspark-patches/0001-merge-consecutive-assistant-messages.patch \
    && python3 -m py_compile \
        vllm/tokenizers/deepseek_v4_encoding.py \
    && python3 /tmp/dspark-patches/verify_patch.py \
    && rm -rf /tmp/dspark-patches

LABEL org.opencontainers.image.title="DeepSeek V4 Flash 0731 stable runtime for dual DGX Spark" \
      org.opencontainers.image.version="0.1.1-stable-pr50686" \
      org.opencontainers.image.description="Anemll 0.1.1 plus vLLM DeepSeek-V4 consecutive-assistant prompt fix" \
      org.opencontainers.image.source="https://github.com/vibe-agi/dgxspark_deepseekv4flash0731"
