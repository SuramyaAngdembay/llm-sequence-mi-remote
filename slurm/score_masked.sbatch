#!/bin/bash
# Score a profile-masked eval JSONL with the frozen adapted model.
# Env: DATASET (r62|r42), VARIANT (no_psy|no_profile)
#SBATCH -A cis230270-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI
module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
export HF_HOME=$PROJ/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_DIR"
if [ "$DATASET" = "r62" ]; then
  ADAPTER=$PROJ/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter
  MASKDIR=$PROJ/outputs/masked_jsonl_r62
else
  ADAPTER=$PROJ/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter
  MASKDIR=$PROJ/outputs/masked_jsonl_r42
fi
python scripts/score_adapter_examples.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --jsonl-path "$MASKDIR/eval_${VARIANT}.jsonl" \
  --adapter-dir "$ADAPTER" \
  --output-dir "$PROJ/outputs/masked_scores_${DATASET}_${VARIANT}" \
  --batch-size 12 --loss-batch-size 4
echo DONE
