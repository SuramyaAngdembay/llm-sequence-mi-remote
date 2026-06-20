#!/bin/bash
#SBATCH --job-name=qwen3b_qlora
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # ~1/4 of the 96-core H100 node
#SBATCH --mem=240G                  # ~1/4 of ~1 TB host RAM
#SBATCH --time=24:00:00

# Anvil (Purdue RCAC). H100 = 'ai' partition; for an A100 smoke test use:
#   --account=cis230270-gpu  --partition=gpu-debug
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

# Compute nodes have no internet: model weights must already be in HF_HOME (download on login node).
export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_qlora.py \
  --config configs/qwen3b_qlora_session.yaml
