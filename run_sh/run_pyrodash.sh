#!/usr/bin/env bash
# Pier DeepSWE：一次跑完 agent + verifier（自带评测）
# deepseek-v4-flash-0731 @ 208.64.254.189:8000
set -euo pipefail

export DOCKER_HOST=unix:///tmp/docker-rt.sock
export OPENAI_BASE_URL="https://pre-token-plan-cn-east-1.pyromind-asia.cn/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
unset MSWEA_API_KEY || true

mkdir -p /workspace/work/mjy/eval/deep-swe-jobs

echo "Pier -> ${OPENAI_BASE_URL}  model=openai/pyrodash"
echo "Results -> /workspace/work/mjy/eval/deep-swe-jobs/pyrodash/"

pier run \
  -p /workspace/work/mjy/eval/deep-swe/tasks \
  -a mini-swe-agent \
  -m openai/pyrodash \
  -n 4 \
  -l 4 \
  --sample-seed 0 \
  -o /workspace/work/mjy/eval/deep-swe-jobs \
  --job-name pyrodash-1010-2 \
  --ak 'cost_limit=9999' \
  -y

echo "Done. See:"
echo "  /workspace/work/mjy/eval/deep-swe-jobs/pyrodash/"
echo "  /workspace/work/mjy/eval/deep-swe-jobs/pyrodash/*/verifier/reward.json"