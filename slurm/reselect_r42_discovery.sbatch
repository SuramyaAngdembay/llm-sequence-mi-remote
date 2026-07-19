#!/bin/bash
# Re-rank r4.2 token-SAE features using discovery-user positives only.
# Short GPU job (SAE encoding passes); output frontier dir feeds the
# confirmation-user causal/necessity evals.
#SBATCH -A cis230270-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH -J reselect_r42
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI

module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"

cd "$REPO_DIR"
python scripts/reselect_token_sae_features.py \
  --extract-dir "$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on" \
  --data-dir "$PROJ/outputs/session_jsonl_r42" \
  --frontier-dir "$PROJ/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on" \
  --out-frontier-dir "$PROJ/outputs/token_delta_sae_frontier_r42_discovery_split" \
  --layer 26 --latent-mult 2 --k 4 \
  --discovery-user-file "$PROJ/outputs/user_splits_r42/discovery_users.txt" \
  --benign-sample-prob "${BENIGN_SAMPLE_PROB:-0.25}" \
  --batch-size 8192 \
  --device cuda \
  --seed 42

echo DONE
