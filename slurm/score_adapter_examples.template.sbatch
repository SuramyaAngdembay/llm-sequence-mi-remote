#!/bin/bash
# Score example-level NLLs from a fixed adapted model over a JSONL pool.
# This is detector benchmarking only: no hidden-state delta extraction.
#SBATCH -A cis230270-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH -J score_examples
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
conda activate "$CONDA_ENV"

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
JSONL_PATH="${JSONL_PATH:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl/all.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
FLUSH_ROWS="${FLUSH_ROWS:-4096}"
USERS_FILE="${USERS_FILE:-}"
LOG_EVERY="${LOG_EVERY:-4096}"
GPU_POLL_SEC="${GPU_POLL_SEC:-0}"
SEPARATE_BASE_MODEL="${SEPARATE_BASE_MODEL:-0}"

mkdir -p "$OUTPUT_DIR"
GPU_LOG="${GPU_LOG:-${OUTPUT_DIR}/gpu_poll_${SLURM_JOB_ID}.csv}"
TIME_LOG="${TIME_LOG:-${OUTPUT_DIR}/time_${SLURM_JOB_ID}.txt}"
JOB_META="${JOB_META:-${OUTPUT_DIR}/job_${SLURM_JOB_ID}.json}"

EXTRA_ARGS=()
if [[ -n "$USERS_FILE" ]]; then
  EXTRA_ARGS+=(--users-file "$USERS_FILE")
fi
if [[ "$SEPARATE_BASE_MODEL" == "1" || "$SEPARATE_BASE_MODEL" == "true" ]]; then
  EXTRA_ARGS+=(--separate-base-model)
fi

cat > "$JOB_META" <<EOM
{
  "job_id": "${SLURM_JOB_ID}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-}",
  "node": "$(hostname)",
  "config": "${CONFIG}",
  "jsonl_path": "${JSONL_PATH}",
  "adapter_dir": "${ADAPTER_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "batch_size": ${BATCH_SIZE},
  "loss_batch_size": ${LOSS_BATCH_SIZE},
  "max_examples": ${MAX_EXAMPLES},
  "separate_base_model": "${SEPARATE_BASE_MODEL}"
}
EOM

MONITOR_PID=""
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if (( GPU_POLL_SEC > 0 )); then
  {
    echo "timestamp,index,name,memory.used MiB,memory.total MiB,utilization.gpu %"
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
      sleep "$GPU_POLL_SEC"
    done
  } > "$GPU_LOG" &
  MONITOR_PID=$!
fi

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-} batch_size=$BATCH_SIZE loss_batch_size=$LOSS_BATCH_SIZE max_examples=$MAX_EXAMPLES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

set +e
/usr/bin/time -v -o "$TIME_LOG" \
  python scripts/score_adapter_examples.py \
  --config "$CONFIG" \
  --jsonl-path "$JSONL_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --loss-batch-size "$LOSS_BATCH_SIZE" \
  --max-examples "$MAX_EXAMPLES" \
  --flush-rows "$FLUSH_ROWS" \
  --log-every "$LOG_EVERY" \
  "${EXTRA_ARGS[@]}"
RC=$?
set -e
exit "$RC"
