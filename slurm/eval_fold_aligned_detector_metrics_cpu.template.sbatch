#!/bin/bash
# Fold-aligned detector evaluation over full-population remote example scores.
# This reconstructs the local leave-one-malicious-user-out test folds.
#SBATCH -A cis230270
#SBATCH -p shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J det_fold_eval
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
SCORES_PARQUET="${SCORES_PARQUET:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores/example_scores.parquet}"
RUN_NAME="${RUN_NAME:-qwen3_8b_r62_fold_aligned}"
SEED="${SEED:-42}"
BENIGN_TEST_USERS="${BENIGN_TEST_USERS:-800}"
FOLD_LIMIT="${FOLD_LIMIT:-0}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/qwen3_8b_token_causal/detector_metrics_fold_aligned}"

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
conda activate "$CONDA_ENV"

cd "$REPO_DIR"
python scripts/eval_fold_aligned_detector_metrics.py \
  --scores-parquet "$SCORES_PARQUET" \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --benign-test-users "$BENIGN_TEST_USERS" \
  --fold-limit "$FOLD_LIMIT" \
  --out-dir "$OUT_DIR"
