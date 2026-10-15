#!/usr/bin/env bash
# Pier DeepSWE：一次跑完 agent + verifier（自带评测）
# deepseek-v4-flash-0731 @ 208.64.254.189:8000
set -euo pipefail

export DOCKER_HOST=unix:///tmp/docker-rt.sock
export OPENAI_BASE_URL="https://pre-token-plan-cn-east-1.pyromind-asia.cn/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
unset MSWEA_API_KEY || true

mkdir -p /workspace/work/mjy/eval/deep-swe-jobs

echo "Pier -> ${OPENAI_BASE_URL}  model=openai/deepseek-v4.1-flash"
echo "Results -> /workspace/work/mjy/eval/deep-swe-jobs/deepseek-v4.1-flash/"

pier run \
  -p /workspace/work/mjy/eval/deep-swe/tasks \
  -a mini-swe-agent \
  -m openai/deepseek-v4.1-flash \
  -n 4 \
  -o /workspace/work/mjy/eval/deep-swe-jobs \
  --job-name deepseek-v4.1-flash-1011_all \
  --ak 'cost_limit=9999' \
  --ak 'model_kwargs={"temperature":1.0,"top_p":1.0,"drop_params":true,"timeout":3000,"extra_body":{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"max"},"reasoning_effort":"max"}}' \
  -y

echo "Done. See:"
echo "  /workspace/work/mjy/eval/deep-swe-jobs/deepseek-v4.1-flash/"
echo "  /workspace/work/mjy/eval/deep-swe-jobs/deepseek-v4.1-flash/*/verifier/reward.json"
