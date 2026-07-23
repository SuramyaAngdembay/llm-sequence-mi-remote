#!/bin/bash
# Malicious-population subsampling test of the population-structure hypothesis (r4.2).
#SBATCH -A cis230270-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:15:00
#SBATCH -J subsample_r42
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI
module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
export HF_HOME=$PROJ/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_DIR"
python scripts/population_subsample_attribution.py \
  --extract-dir $PROJ/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on \
  --data-dir $PROJ/outputs/session_jsonl_r42 \
  --frontier-dir $PROJ/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on \
  --adapter-dir $PROJ/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter \
  --layer 26 --latent-mult 2 --k 4 \
  --out-dir $PROJ/outputs/population_subsample_r42 \
  --sizes 4,10,20,30,60 --reps 10 --benign-sample-prob 0.10 \
  --device cuda --seed 42
echo DONE
