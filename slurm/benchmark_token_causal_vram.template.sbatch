#!/bin/bash
# Short Qwen3-8B token-causal forward-hook VRAM/RSS probe.
# Submit with partition/account overrides from the launcher.
#SBATCH --job-name=tok_causal_vram
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

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
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_r42_causal_vram_probe/${SLURM_JOB_PARTITION:-unknown}}"
BATCH_SIZES="${BATCH_SIZES:-8,12,16,20,24}"
LAYER="${LAYER:-26}"
MAX_EXAMPLES="${MAX_EXAMPLES:-4096}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
BENCH_STEPS="${BENCH_STEPS:-2}"
SAMPLE_MODE="${SAMPLE_MODE:-longest}"
ATTN_IMPL="${ATTN_IMPL:-auto}"
TARGET_GIB="${TARGET_GIB:-0}"
TARGET_FRACTION="${TARGET_FRACTION:-0.87}"
TARGET_TOLERANCE_GIB="${TARGET_TOLERANCE_GIB:-2}"
SEED="${SEED:-42}"
GPU_POLL_SEC="${GPU_POLL_SEC:-2}"

mkdir -p "$OUT_DIR"
GPU_LOG="${OUT_DIR}/gpu_poll_${SLURM_JOB_ID}.csv"
TIME_LOG="${OUT_DIR}/time_${SLURM_JOB_ID}.txt"
JOB_META="${OUT_DIR}/job_${SLURM_JOB_ID}.json"

cat > "$JOB_META" <<EOM
{
  "job_id": "${SLURM_JOB_ID}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-}",
  "node": "$(hostname)",
  "config": "${CONFIG}",
  "adapter_dir": "${ADAPTER_DIR}",
  "data_dir": "${DATA_DIR}",
  "out_dir": "${OUT_DIR}",
  "batch_sizes": "${BATCH_SIZES}",
  "layer": ${LAYER},
  "sample_mode": "${SAMPLE_MODE}",
  "attn_impl": "${ATTN_IMPL}"
}
EOM

{
  echo "timestamp,index,name,memory.used MiB,memory.total MiB,utilization.gpu %"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep "$GPU_POLL_SEC"
  done
} > "$GPU_LOG" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-} batch_sizes=$BATCH_SIZES layer=$LAYER"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

set +e
/usr/bin/time -v -o "$TIME_LOG" \
  python scripts/benchmark_token_causal_vram.py \
    --config "$CONFIG" \
    --adapter-dir "$ADAPTER_DIR" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --batch-sizes "$BATCH_SIZES" \
    --layer "$LAYER" \
    --max-examples "$MAX_EXAMPLES" \
    --warmup-steps "$WARMUP_STEPS" \
    --bench-steps "$BENCH_STEPS" \
    --sample-mode "$SAMPLE_MODE" \
    --attn-impl "$ATTN_IMPL" \
    --target-gib "$TARGET_GIB" \
    --target-fraction "$TARGET_FRACTION" \
    --target-tolerance-gib "$TARGET_TOLERANCE_GIB" \
    --seed "$SEED"
RC=$?
set -e

cleanup
exit "$RC"
