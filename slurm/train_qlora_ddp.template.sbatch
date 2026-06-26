#!/bin/bash
# 4x H100 DDP QLoRA launcher (Anvil `ai` partition). Shared by the `Qwen 3B`
# pilot reruns and targeted larger-Qwen scale-ups. Training uses ALL 4 GPUs on
# one H100 node via torchrun data-parallel.
#SBATCH --job-name=qwen_qlora_ddp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # torchrun spawns the per-GPU workers; Slurm sees 1 task
#SBATCH --cpus-per-task=64          # ~16 cores/GPU x 4
#SBATCH --mem=480G
#SBATCH --gres=gpu:4               # whole node's GPUs
#SBATCH --time=12:00:00
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

# Compute nodes have no internet: weights must already be in HF_HOME (download on login node).
export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16          # ~cpus-per-task / nproc (torchrun forces 1 if unset)
# NOTE: do NOT set NCCL_P2P_DISABLE on H100 (NVLink) — leave P2P on.

CONFIG="${CONFIG:-configs/qwen3b_qlora_session.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp}"
NPROC="${NPROC:-4}"            # GPUs to use (=4 for the whole node)
MICRO_BS="${MICRO_BS:-8}"      # 3B-friendly default. Larger Qwen targeted runs override this.
GRAD_ACCUM="${GRAD_ACCUM:-1}"  # Larger Qwen targeted runs usually keep effective batch near 32.
DL_WORKERS="${DL_WORKERS:-16}"
MAX_TRAIN="${MAX_TRAIN:-0}"    # 0 = full dataset; set >0 for a capped smoke
MAX_VAL="${MAX_VAL:-0}"
ATTN_IMPL="${ATTN_IMPL:-}"     # e.g. sdpa or flash_attention_2 if the env supports it
GC_MODE="${GC_MODE:-config}"   # config | on | off
# LR note: config LR=2e-4 was tuned for effective batch 16. For eff batch 32, the sqrt rule
#   suggests ~2.8e-4 (or keep 2e-4 + longer warmup) -- experiment-design call, left to config owner.
# Throughput phase-2: true packing (DataCollatorWithFlattening + flash_attention_2) ~1.4x on short examples.

echo "job=$SLURM_JOB_ID node=$(hostname) | CONDA_ENV=$CONDA_ENV CONFIG=$CONFIG NPROC=$NPROC MICRO_BS=$MICRO_BS GRAD_ACCUM=$GRAD_ACCUM"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [[ -n "$ATTN_IMPL" ]]; then
  EXTRA_ARGS+=(--attn-impl "$ATTN_IMPL")
fi
case "$GC_MODE" in
  config)
    ;;
  on)
    EXTRA_ARGS+=(--gradient-checkpointing)
    ;;
  off)
    EXTRA_ARGS+=(--no-gradient-checkpointing)
    ;;
  *)
    echo "Invalid GC_MODE=$GC_MODE (expected: config|on|off)" >&2
    exit 2
    ;;
esac

# Single-node multi-GPU DDP. HF Trainer auto-detects WORLD_SIZE/RANK from torchrun
# and wraps the model in DistributedDataParallel.
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_qlora.py \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --micro-batch-size "$MICRO_BS" \
    --grad-accum "$GRAD_ACCUM" \
    --dataloader-workers "$DL_WORKERS" \
    --max-train-examples "$MAX_TRAIN" \
    --max-val-examples "$MAX_VAL" \
    "${EXTRA_ARGS[@]}"
