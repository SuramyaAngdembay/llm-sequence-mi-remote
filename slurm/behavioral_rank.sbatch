#!/bin/bash
# SES-restricted behavioral feature ranking. DATASET=r62 (LOUO) | r42 (split).
#SBATCH -A cis230270-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
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
if [ "$DATASET" = "r62" ]; then
  python scripts/rank_behavioral_features.py \
    --extract-dir $PROJ/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2 \
    --data-dir $PROJ/outputs/session_jsonl \
    --frontier-dir $PROJ/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream \
    --adapter-dir $PROJ/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter \
    --out-frontier-dir $PROJ/outputs/token_delta_sae_frontier_r62_behavioral_louo \
    --layer 18 --latent-mult 4 --k 8 \
    --louo-splits-dir $PROJ/outputs/user_splits_r62 \
    --benign-sample-prob 0.05 --device cuda --seed 42
else
  python scripts/rank_behavioral_features.py \
    --extract-dir $PROJ/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on \
    --data-dir $PROJ/outputs/session_jsonl_r42 \
    --frontier-dir $PROJ/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on \
    --adapter-dir $PROJ/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter \
    --out-frontier-dir $PROJ/outputs/token_delta_sae_frontier_r42_behavioral_split \
    --layer 26 --latent-mult 2 --k 4 \
    --discovery-user-file $PROJ/outputs/user_splits_r42/discovery_users.txt \
    --benign-sample-prob 0.25 --device cuda --seed 42
fi
echo DONE
