#!/usr/bin/env bash
# DeepSWE offload: start vLLM SLM → offload adapter → pier (agent+verifier).
# Edit MODEL_PATHS / LLM_* / GPU below; sampling is applied in the adapter.
set -euo pipefail

# shellcheck source=/dev/null
source /workspace/work/mjy/.venv/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tag offload API lives outside this project.
OFFLOAD_ROOT="$(cd "${ROOT}/../offload" && pwd)"
JOBS_DIR="/workspace/work/mjy/eval/deep-swe-jobs"

# =========================
# 只改这里
# =========================
MODEL_PATHS=(
 /workspace/work/spt/slime/runs/agent_offload_pyrodash4b_sft_entropy_docker_async_turn_20260922_125447/checkpoints/iter_0000199_hf
)

export CUDA_VISIBLE_DEVICES=1

# remote LLM (DeepSeek)
LLM_BASE_URL="https://pre-token-plan-cn-east-1.pyromind-asia.cn/v1"
LLM_API_KEY="${OPENAI_API_KEY}"
LLM_MODEL="deepseek-v4.1-flash"

# pier
N_CONCURRENT=8
JOB_NAME_PREFIX="deep-swe-offload"

# --- SLM sampling ---
SLM_TEMPERATURE=0.6
SLM_TOP_P=0.95
SLM_TOP_K=20
SLM_MIN_P=0.0
SLM_PRESENCE_PENALTY=0.0
SLM_REPETITION_PENALTY=1.0

# --- LLM sampling ---
LLM_TEMPERATURE=1.0
LLM_TOP_P=1.0

# shared per-turn SLM+LLM completion budget; 0 = unlimited
MAX_NEW_TOKENS=8192
# =========================

source "${OFFLOAD_ROOT}/serve/common.sh"
source "${OFFLOAD_ROOT}/serve/vllm.sh"
# ports must be after source vllm.sh (it defaults VLLM_PORT=8001)
VLLM_PORT=8007
ADAPTER_PORT=18027

detect_host_ip() {
  local ip=""
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "$ip" ]]; then
    ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
  fi
  echo "${ip:-127.0.0.1}"
}

HOST_IP="${HOST_IP:-$(detect_host_ip)}"
EXPOSED_MODEL="offload-router"
ADAPTER_PID=""
ADAPTER_LOG=""

cleanup_adapter() {
  if [[ -n "${ADAPTER_PID}" ]] && kill -0 "${ADAPTER_PID}" 2>/dev/null; then
    echo "[adapter] stopping pid=${ADAPTER_PID}"
    kill "${ADAPTER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${ADAPTER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${ADAPTER_PID}" 2>/dev/null; then
      kill -9 "${ADAPTER_PID}" 2>/dev/null || true
    fi
    wait "${ADAPTER_PID}" 2>/dev/null || true
  fi
  ADAPTER_PID=""
}

cleanup_all() {
  cleanup_adapter
  cleanup_vllm
}
trap cleanup_all EXIT INT TERM

wait_adapter_ready() {
  local url="http://127.0.0.1:${ADAPTER_PORT}/health"
  local deadline=$((SECONDS + 120))
  echo "[adapter] waiting for ${url}"
  while (( SECONDS < deadline )); do
    if ! kill -0 "${ADAPTER_PID}" 2>/dev/null; then
      echo "[adapter ERROR] exited early; see ${ADAPTER_LOG}" >&2
      return 1
    fi
    if curl -sf --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "[adapter] ready"
      return 0
    fi
    sleep 1
  done
  echo "[adapter ERROR] not ready; see ${ADAPTER_LOG}" >&2
  return 1
}

start_adapter() {
  local served_name="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}/adapter_stats"
  export TOKEN_USAGE_DIR="${ROOT}/token_usage/live/$(basename "${out_dir}")"
  mkdir -p "${TOKEN_USAGE_DIR}"
  ADAPTER_LOG="${out_dir}/adapter.log"
  echo "[adapter] token usage (live) -> ${TOKEN_USAGE_DIR}"
  echo "[adapter] SLM=${served_name} @ http://127.0.0.1:${VLLM_PORT}/v1"
  echo "[adapter] LLM=${LLM_MODEL} @ ${LLM_BASE_URL}"
  echo "[adapter] listen 0.0.0.0:${ADAPTER_PORT}  log=${ADAPTER_LOG}"
  /root/micromamba/envs/slime/bin/python \
    "${OFFLOAD_ROOT}/serve_adapter.py" \
    --host 0.0.0.0 \
    --port "${ADAPTER_PORT}" \
    --slm-base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --slm-api-key "dummy" \
    --slm-model "${served_name}" \
    --llm-base-url "${LLM_BASE_URL}" \
    --llm-api-key "${LLM_API_KEY}" \
    --llm-model "${LLM_MODEL}" \
    --use-llm true \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --slm-temperature "${SLM_TEMPERATURE}" \
    --slm-top-p "${SLM_TOP_P}" \
    --slm-top-k "${SLM_TOP_K}" \
    --slm-min-p "${SLM_MIN_P}" \
    --slm-presence-penalty "${SLM_PRESENCE_PENALTY}" \
    --slm-repetition-penalty "${SLM_REPETITION_PENALTY}" \
    --llm-temperature "${LLM_TEMPERATURE}" \
    --llm-top-p "${LLM_TOP_P}" \
    --stats-dir "${out_dir}/adapter_stats" \
    >"${ADAPTER_LOG}" 2>&1 &
  ADAPTER_PID=$!
  wait_adapter_ready
}

run_pier() {
  local job_name="$1"
  # Pier agent 在容器内，必须用宿主机 IP，不能用 127.0.0.1
  export DOCKER_HOST=unix:///tmp/docker-rt.sock
  export OPENAI_BASE_URL="http://${HOST_IP}:${ADAPTER_PORT}/v1"
  export OPENAI_API_KEY="dummy"
  unset MSWEA_API_KEY || true

  echo "Pier -> ${OPENAI_BASE_URL}  model=openai/${EXPOSED_MODEL}"
  echo "Results -> ${JOBS_DIR}/${job_name}/"
  pier run \
    -p "${ROOT}/tasks" \
    -a mini-swe-agent \
    -m "openai/${EXPOSED_MODEL}" \
    -n "${N_CONCURRENT}" \
    -o "${JOBS_DIR}" \
    --job-name "${job_name}" \
    --ak 'cost_limit=9999' \
    --ak 'model_kwargs={"drop_params":true,"timeout":3000}' \
    -y
}

mkdir -p "${JOBS_DIR}"

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "ERROR: MODEL_PATHS is empty; edit $0" >&2
  exit 1
fi

for model_path in "${MODEL_PATHS[@]}"; do
  [[ -n "${model_path}" && "${model_path}" != \#* ]] || continue
  [[ -d "${model_path}" ]] || { echo "model path missing: ${model_path}" >&2; exit 1; }

  model_name="$(basename "${model_path}")"
  experiment_name="$(experiment_name_from_model_path "${model_path}")"
  job_name="${JOB_NAME_PREFIX}_${experiment_name}_${model_name}"
  job_dir="${JOBS_DIR}/${job_name}"
  mkdir -p "${job_dir}"

  echo "[pipeline] model=${model_path}"
  echo "[pipeline] job=${job_name}"

  start_vllm "${model_path}" "${model_name}" "${job_dir}"
  start_adapter "${model_name}" "${job_dir}"
  run_pier "${job_name}"

  if [[ -f "${job_dir}/adapter_stats/summary.json" ]]; then
    echo "[pipeline] adapter stats -> ${job_dir}/adapter_stats/summary.json"
    /root/micromamba/envs/slime/bin/python - <<PY
import json
from pathlib import Path
s = json.loads(Path("${job_dir}/adapter_stats/summary.json").read_text())
t = s.get("totals") or {}
print(
    "[pipeline] totals:",
    f"sessions={t.get('sessions')} turns={t.get('turns')} offloads={t.get('offload_count')} "
    f"slm_p={t.get('slm_prompt_tokens')} slm_c={t.get('slm_completion_tokens')} "
    f"llm_p={t.get('llm_prompt_tokens')} llm_c={t.get('llm_completion_tokens')}",
)
PY
  fi

  cleanup_adapter
  cleanup_vllm
  echo "Done. See: ${job_dir}/"
  echo "  ${JOBS_DIR}/${job_name}/*/verifier/reward.json"
done

echo "[pipeline] all models finished"
