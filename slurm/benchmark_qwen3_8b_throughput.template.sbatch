#!/bin/bash
# One-H100 real-data Qwen3-8B QLoRA throughput and VRAM benchmark.
# This runs short training loops over session JSONL and writes CSV/JSON rows.
# It is a benchmark-only job and does not save adapters or checkpoints.
#SBATCH --job-name=qwen3_thr_bench
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

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
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_r42_throughput_benchmark}"
MICRO_BATCHES="${MICRO_BATCHES:-12,13,14}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-4096}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
BENCH_STEPS="${BENCH_STEPS:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-longest}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GC_MODE="${GC_MODE:-on}"
DL_WORKERS="${DL_WORKERS:-4}"
TARGET_GIB="${TARGET_GIB:-70}"
TARGET_TOLERANCE_GIB="${TARGET_TOLERANCE_GIB:-2}"

EXTRA_ARGS=()
if [[ "$GC_MODE" == "on" || "$GC_MODE" == "config" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
elif [[ "$GC_MODE" == "off" ]]; then
  EXTRA_ARGS+=(--no-gradient-checkpointing)
else
  echo "Invalid GC_MODE=$GC_MODE (expected off|on|config)" >&2
  exit 2
fi

echo "job=$SLURM_JOB_ID node=$(hostname) config=$CONFIG data_dir=$DATA_DIR micro_batches=$MICRO_BATCHES sample_mode=$SAMPLE_MODE gc_mode=$GC_MODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/benchmark_qlora_throughput.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --micro-batches "$MICRO_BATCHES" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --warmup-steps "$WARMUP_STEPS" \
  --bench-steps "$BENCH_STEPS" \
  --sample-mode "$SAMPLE_MODE" \
  --attn-impl "$ATTN_IMPL" \
  --dataloader-workers "$DL_WORKERS" \
  --target-gib "$TARGET_GIB" \
  --target-tolerance-gib "$TARGET_TOLERANCE_GIB" \
  "${EXTRA_ARGS[@]}"
