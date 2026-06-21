#!/bin/bash
#SBATCH --job-name=delta_sae
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # ~1/4 of the 96-core H100 node
#SBATCH --mem=240G                  # ~1/4 of ~1 TB host RAM
#SBATCH --time=24:00:00

# Anvil (Purdue RCAC). H100 = 'ai' partition. The SAE sweep is light — an A100 is often enough:
#   --account=cis230270-gpu  --partition=gpu
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/delta_cache/qwen3b_session_mean_deltas}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/delta_sae_frontier_qwen3b}"

python scripts/train_delta_sae_frontier.py \
  --config configs/delta_sae_frontier.yaml \
  --extract-dir "$EXTRACT_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --device cuda
