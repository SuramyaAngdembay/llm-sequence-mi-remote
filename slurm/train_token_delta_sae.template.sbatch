#!/bin/bash
# Token-level delta-SAE frontier. Start with layer 18 only and a capped token budget.
#SBATCH --job-name=token_delta_sae
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
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b}"
LAYERS="${LAYERS:-18}"
MAX_ROWS="${MAX_ROWS:-2000000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LATENT_MULTIPLIERS="${LATENT_MULTIPLIERS:-2,4}"
TOPK_VALUES="${TOPK_VALUES:-4,8}"

python scripts/train_delta_sae_frontier.py \
  --config configs/token_delta_sae_frontier.yaml \
  --extract-dir "$EXTRACT_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --device cuda \
  --layers "$LAYERS" \
  --max-rows "$MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --latent-multipliers "$LATENT_MULTIPLIERS" \
  --topk "$TOPK_VALUES"
