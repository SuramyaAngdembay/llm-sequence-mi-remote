#!/bin/bash
#SBATCH --job-name=delta_extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION

set -euo pipefail

cd /path/to/llm-sequence-mi-remote

python scripts/extract_adapter_deltas.py \
  --config configs/qwen3b_qlora_session.yaml

