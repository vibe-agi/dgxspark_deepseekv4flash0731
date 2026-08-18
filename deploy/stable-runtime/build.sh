#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1}"
IMAGE="${IMAGE:-deepseek-v4-flash:0.1.1-stable-20260818}"
CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    docker pull "$BASE_IMAGE"
fi
docker build \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --label "org.opencontainers.image.created=$CREATED" \
    --label "org.opencontainers.image.base.name=$BASE_IMAGE" \
    --label "org.opencontainers.image.revision=vllm-pr-50686-plus-dsml-drift" \
    --tag "$IMAGE" \
    "$SCRIPT_DIR"

docker run --rm --entrypoint python3 "$IMAGE" -c \
    'import json; from vllm.tokenizers.deepseek_v4_encoding import merge_consecutive_assistant_messages; from vllm.parser.deepseek_v4 import _dsml_arg_converter; assert len(merge_consecutive_assistant_messages([{"role":"assistant","content":"a"},{"role":"assistant","reasoning":"b"}])) == 1; assert _dsml_arg_converter("<|DSML|parameter name=\"x\" string=\"true\">ok</|DSML|parameter>", False) == json.dumps({"x":"ok"})'

docker image inspect "$IMAGE" --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.version"}}'
