#!/usr/bin/env bash
# Pier DeepSWE via already-running offload adapter (start with eval/offload/api.sh first).
# Sampling is applied inside the adapter; pier only needs drop_params/timeout.
set -euo pipefail

# Pier agent 在容器内，不能用 127.0.0.1；改成你的宿主机 IP / adapter 端口
export DOCKER_HOST=unix:///tmp/docker-rt.sock
export OPENAI_BASE_URL="http://10.244.7.247:18029/v1"
export OPENAI_API_KEY="dummy"
unset MSWEA_API_KEY || true

mkdir -p /workspace/work/mjy/eval/deep-swe-jobs

echo "Pier -> ${OPENAI_BASE_URL}  model=openai/offload-router"
echo "Results -> /workspace/work/mjy/eval/deep-swe-jobs/offload_api/"

pier run \
  -p /workspace/work/mjy/eval/deep-swe/tasks \
  -a mini-swe-agent \
  -m openai/offload-router \
  -n 4 \
  -l 4 \
  --sample-seed 0 \
  -o /workspace/work/mjy/eval/deep-swe-jobs \
  --job-name 0927_c19_keep_1013_n4 \
  --ak 'cost_limit=9999' \
  --ak 'model_class=litellm' \
  --ak 'model_kwargs={"drop_params":true,"timeout":3000}' \
  -y

