#!/bin/bash
#SBATCH --job-name=delta_extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # ~1/4 of the 96-core H100 node
#SBATCH --mem=240G                  # ~1/4 of ~1 TB host RAM
#SBATCH --time=24:00:00

# Anvil (Purdue RCAC). H100 = 'ai' partition; A100 alt: --account=cis230270-gpu --partition=gpu
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Write large delta shards to $SCRATCH (fast, regenerable); copy keepers to $PROJECT afterwards.
# Keep shards few-and-large: $SCRATCH is near its 1M-file limit.
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
# NOTE: default points at the 4-GPU DDP adapter (qwen3b_session_qlora_ddp), which is what we trained.
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/delta_cache/qwen3b_session_mean_deltas}"
SPLIT="${SPLIT:-eval}"
BATCH_SIZE="${BATCH_SIZE:-16}"        # 3B 4-bit x2 models + hidden states fits easily on 80GB (script default 2 is slow)
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"     # 0 = all examples in the split

python scripts/extract_adapter_deltas.py \
  --config configs/qwen3b_qlora_session.yaml \
  --data-dir "$DATA_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --max-examples "$MAX_EXAMPLES" \
  --pool-unit mean
