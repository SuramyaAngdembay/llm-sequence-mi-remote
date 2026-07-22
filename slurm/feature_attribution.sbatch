#!/bin/bash
# Feature-token attribution: do top causal features fire on behavioral SES
# tokens or profile DAY/PSY tokens? Parameterized by DATASET (r62|r42).
#SBATCH -A cis230270-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
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
  EXTRACT=$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2
  DATA=$PROJ/outputs/session_jsonl
  FRONTIER=$PROJ/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream
  ADAPTER=$PROJ/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter
  LAYER=18; LM=4; KV=8
else
  EXTRACT=$PROJ/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on
  DATA=$PROJ/outputs/session_jsonl_r42
  FRONTIER=$PROJ/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on
  ADAPTER=$PROJ/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter
  LAYER=26; LM=2; KV=4
fi
python scripts/feature_token_attribution.py \
  --extract-dir "$EXTRACT" \
  --data-dir "$DATA" \
  --frontier-dir "$FRONTIER" \
  --adapter-dir "$ADAPTER" \
  --layer $LAYER --latent-mult $LM --k $KV \
  --out-dir "$PROJ/outputs/feature_attribution_${DATASET}" \
  --device cuda --seed 42
echo DONE
