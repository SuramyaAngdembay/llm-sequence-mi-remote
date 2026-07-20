#!/bin/bash
# Re-rank r6.2 token-SAE features per LOUO fold (all 4 folds in one job:
# the 956GB cache is read once, benign rows sampled at 5%).
#SBATCH -A cis230270-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH -J reselect_r62
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI
module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
cd "$REPO_DIR"
python scripts/reselect_token_sae_features.py \
  --extract-dir "$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2" \
  --data-dir "$PROJ/outputs/session_jsonl" \
  --frontier-dir "$PROJ/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream" \
  --out-frontier-dir "$PROJ/outputs/token_delta_sae_frontier_r62_louo" \
  --layer 18 --latent-mult 4 --k 8 \
  --louo-splits-dir "$PROJ/outputs/user_splits_r62" \
  --benign-sample-prob "${BENIGN_SAMPLE_PROB:-0.05}" \
  --batch-size 8192 \
  --device cuda \
  --seed 42
echo DONE
