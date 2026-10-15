#!/usr/bin/env bash
# Pier DeepSWE：一次跑完 agent + verifier（自带评测）
# deepseek-v4-flash-0731 @ 208.64.254.189:8000
set -euo pipefail

export DOCKER_HOST=unix:///tmp/docker-rt.sock
export OPENAI_BASE_URL="http://10.244.7.191:8066/v1"
export OPENAI_API_KEY="dummy"
unset MSWEA_API_KEY || true

mkdir -p /workspace/work/mjy/eval/deep-swe-jobs

echo "Pier -> ${OPENAI_BASE_URL}  model=openai/qwen3.5_4b"
echo "Results -> /workspace/work/mjy/eval/deep-swe-jobs/qwen3.5_4b/"

pier run \
  -p /workspace/work/mjy/eval/deep-swe/tasks \
  -a mini-swe-agent \
  -m openai/qwen3.5_4b \
  -n 8 \
  -o /workspace/work/mjy/eval/deep-swe-jobs \
  --job-name qwen3.5_4b_1009 \
  --ak 'cost_limit=9999' \
  --ak 'model_class=litellm' \
  --ak 'model_kwargs={"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"max_tokens":8192,"drop_params":true,"timeout":3000,"extra_body":{"top_k":20,"min_p":0.0,"repetition_penalty":1.0}}' \
  -y
