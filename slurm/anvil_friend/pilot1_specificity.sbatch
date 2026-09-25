#!/bin/bash
# EXPLORATORY Pilot 1: intervention specificity and implementation decomposition (H1, H2).
# docs/HYPOTHESIS_LEDGER_2026-09-25.md; budget 0.5 GPU-h (1 A100, 30 min).
# Collaborator allocation only. Inputs frozen in prep/ before submission.
#SBATCH --job-name=pilot1_specificity
#SBATCH --output=/anvil/scratch/x-bbhusal1/pilots_2026_09_25/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/pilots_2026_09_25/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=96G
#SBATCH --time=00:30:00
#SBATCH -A cis260991-gpu -p gpu-debug
set -euo pipefail
P=/anvil/scratch/x-bbhusal1/pilots_2026_09_25
S=/anvil/scratch/x-sangdembay/pkg4_share
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$P/repo"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$P/prep/pilot_populations.json" "$P/prep/pilot_deltas_l26.npz"
python scripts/pilot_intervention_specificity.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir "$S/session_jsonl_r42" --adapter-dir "$S/adapter" \
  --extract-dir "$S/token_deltas" --delta-cache "$P/prep/pilot_deltas_l26.npz" \
  --frontier-dir "$S/frontier" --populations "$P/prep/pilot_populations.json" \
  --out-dir "$P/pilot1" --batch-size 12 --loss-batch-size 4
echo PILOT1_DONE
