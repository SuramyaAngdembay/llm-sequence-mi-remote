#!/bin/bash
# Token-level causal patching on adapter deltas. This is the sharper successor to mean-pooled patching.
# Default target = layer 18, latent_mult 2, k 8.
#SBATCH --job-name=token_delta_causal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/qwen3b_qlora_session.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp/adapter}"
EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18}"
FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08}"

LAYER="${LAYER:-18}"
LATENT_MULT="${LATENT_MULT:-2}"
TOPK="${TOPK:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-0}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-0}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}"
TOP_SETS="${TOP_SETS:-top1,top3,top5}"
CONTROL_SET="${CONTROL_SET:-control3}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_RECEIVERS="${MAX_RECEIVERS:-0}"
MAX_CANDIDATE_DONORS="${MAX_CANDIDATE_DONORS:-16}"
EXCLUDE_SAME_USER_DONORS="${EXCLUDE_SAME_USER_DONORS:-0}"
GPU_POLL_SEC="${GPU_POLL_SEC:-0}"

mkdir -p "$OUTPUT_DIR"
GPU_LOG="${GPU_LOG:-${OUTPUT_DIR}/gpu_poll_${SLURM_JOB_ID}.csv}"
TIME_LOG="${TIME_LOG:-${OUTPUT_DIR}/time_${SLURM_JOB_ID}.txt}"
JOB_META="${JOB_META:-${OUTPUT_DIR}/job_${SLURM_JOB_ID}.json}"

cat > "$JOB_META" <<EOM
{
  "job_id": "${SLURM_JOB_ID}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-}",
  "node": "$(hostname)",
  "config": "${CONFIG}",
  "data_dir": "${DATA_DIR}",
  "adapter_dir": "${ADAPTER_DIR}",
  "extract_dir": "${EXTRACT_DIR}",
  "frontier_dir": "${FRONTIER_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "layer": ${LAYER},
  "latent_mult": ${LATENT_MULT},
  "topk": ${TOPK},
  "batch_size": ${BATCH_SIZE},
  "loss_batch_size": ${LOSS_BATCH_SIZE},
  "patch_chunk_size": ${PATCH_CHUNK_SIZE},
  "max_receivers": ${MAX_RECEIVERS},
  "max_candidate_donors": ${MAX_CANDIDATE_DONORS},
  "context_modes": "${CONTEXT_MODES}",
  "top_sets": "${TOP_SETS}",
  "control_set": "${CONTROL_SET}",
  "exclude_same_user_donors": "${EXCLUDE_SAME_USER_DONORS}"
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

EXTRA_ARGS=()
if [[ "$EXCLUDE_SAME_USER_DONORS" == "1" || "$EXCLUDE_SAME_USER_DONORS" == "true" ]]; then
  EXTRA_ARGS+=(--exclude-same-user-donors)
fi

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-} batch_size=$BATCH_SIZE loss_batch_size=$LOSS_BATCH_SIZE patch_chunk_size=$PATCH_CHUNK_SIZE max_receivers=$MAX_RECEIVERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

set +e
/usr/bin/time -v -o "$TIME_LOG" \
  python -u scripts/eval_token_delta_sae_causal.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --extract-dir "$EXTRACT_DIR" \
  --frontier-dir "$FRONTIER_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --layer "$LAYER" \
  --latent-mult "$LATENT_MULT" \
  --k "$TOPK" \
  --batch-size "$BATCH_SIZE" \
  --loss-batch-size "$LOSS_BATCH_SIZE" \
  --sae-batch-size "$SAE_BATCH_SIZE" \
  --patch-chunk-size "$PATCH_CHUNK_SIZE" \
  --token-delta-dtype "$TOKEN_DELTA_DTYPE" \
  --context-modes "$CONTEXT_MODES" \
  --top-sets "$TOP_SETS" \
  --control-set "$CONTROL_SET" \
  --active-control-min-frac "$ACTIVE_CONTROL_MIN_FRAC" \
  --alphas "$ALPHAS" \
  --max-receivers "$MAX_RECEIVERS" \
  --max-candidate-donors "$MAX_CANDIDATE_DONORS" \
  "${EXTRA_ARGS[@]}"
RC=$?
set -e

cleanup
exit "$RC"
