#!/bin/bash
# Token-level necessity ablation on adapter deltas. This is the negative-intervention
# counterpart to token causal patching.
#SBATCH --job-name=token_delta_necessity
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
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3b/l18_m02_k08}"

LAYER="${LAYER:-18}"
LATENT_MULT="${LATENT_MULT:-2}"
TOPK="${TOPK:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-0}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-0}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
FULL_LOGITS_MAX_GIB="${FULL_LOGITS_MAX_GIB:-28}"
MAX_LOGIT_ELEMENTS="${MAX_LOGIT_ELEMENTS:-536870912}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
GPU_POLL_SEC="${GPU_POLL_SEC:-0}"
EXCLUDE_SAME_USER_MATCHES="${EXCLUDE_SAME_USER_MATCHES:-0}"

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
  "max_pairs": ${MAX_PAIRS},
  "context_modes": "${CONTEXT_MODES}",
  "top_sets": "${TOP_SETS}",
  "control_set": "${CONTROL_SET}"
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

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-} batch_size=$BATCH_SIZE loss_batch_size=$LOSS_BATCH_SIZE patch_chunk_size=$PATCH_CHUNK_SIZE max_pairs=$MAX_PAIRS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

set +e
EXTRA_ARGS=()
if [[ "$EXCLUDE_SAME_USER_MATCHES" == "1" || "$EXCLUDE_SAME_USER_MATCHES" == "true" ]]; then
  EXTRA_ARGS+=(--exclude-same-user-matches)
fi
/usr/bin/time -v -o "$TIME_LOG" \
  python -u scripts/eval_token_delta_sae_necessity.py \
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
  --full-logits-max-gib "$FULL_LOGITS_MAX_GIB" \
  --max-logit-elements "$MAX_LOGIT_ELEMENTS" \
  --token-delta-dtype "$TOKEN_DELTA_DTYPE" \
  --context-modes "$CONTEXT_MODES" \
  --top-sets "$TOP_SETS" \
  --control-set "$CONTROL_SET" \
  --active-control-min-frac "$ACTIVE_CONTROL_MIN_FRAC" \
  --alphas "$ALPHAS" \
  --max-pairs "$MAX_PAIRS" \
  "${EXTRA_ARGS[@]}"
RC=$?
set -e

cleanup
exit "$RC"
