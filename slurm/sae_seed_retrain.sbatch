#!/bin/bash
# Retrain the headline token-SAE config with a fresh seed on cached deltas.
# Parameterized: DATASET (r62|r42), SAE_SEED. SAE weights only; adapter and
# extraction untouched. Used for the seed-stability / non-identifiability
# control.
#SBATCH -A cis230270-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI
module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
cd "$REPO_DIR"
if [ "$DATASET" = "r62" ]; then
  EXTRACT=$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2
  LAYERS=${LAYERS:-18}; LM=${LM:-4}; KV=${KV:-8}
else
  EXTRACT=$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on
  LAYERS=${LAYERS:-26}; LM=${LM:-2}; KV=${KV:-4}
fi
python scripts/train_delta_sae_frontier.py \
  --config configs/token_delta_sae_frontier.yaml \
  --extract-dir "$EXTRACT" \
  --out-dir "$PROJ/outputs/token_delta_sae_frontier_${DATASET}_seed${SAE_SEED}" \
  --device cuda \
  --batch-size 4096 \
  --epochs 20 \
  --layers "$LAYERS" \
  --max-rows 2000000 \
  --latent-multipliers "$LM" \
  --topk "$KV" \
  --seed "$SAE_SEED"
echo DONE
