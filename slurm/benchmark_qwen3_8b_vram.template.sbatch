#!/bin/bash
# One-H100 Qwen3-8B QLoRA VRAM stress benchmark. Measures per-GPU memory for
# synthetic full-length batches so we can pick a training microbatch rationally.
#SBATCH --job-name=qwen3_vram_bench
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
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
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_vram_benchmark}"
MICRO_BATCHES="${MICRO_BATCHES:-4,8,12,16}"
SEQ_LEN="${SEQ_LEN:-2048}"
BENCH_STEPS="${BENCH_STEPS:-2}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GC_MODE="${GC_MODE:-off}"

EXTRA_ARGS=()
if [[ "$GC_MODE" == "on" || "$GC_MODE" == "config" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
elif [[ "$GC_MODE" != "off" ]]; then
  echo "Invalid GC_MODE=$GC_MODE (expected off|on|config)" >&2
  exit 2
fi

echo "job=$SLURM_JOB_ID node=$(hostname) config=$CONFIG micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN gc_mode=$GC_MODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/benchmark_qlora_vram.py \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --micro-batches "$MICRO_BATCHES" \
  --seq-len "$SEQ_LEN" \
  --steps "$BENCH_STEPS" \
  --attn-impl "$ATTN_IMPL" \
  "${EXTRA_ARGS[@]}"
