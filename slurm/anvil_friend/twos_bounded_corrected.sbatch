#!/bin/bash
# Bounded corrected TWOS intervention (docs/TWOS_CORRECTED_INTERVENTION_PLAN.md).
# Held-out confirmation users; features ranked on discovery users only; alpha 0
# is the reconstruction-only control. Collaborator allocation only.
#SBATCH --job-name=twos_corr_s42
#SBATCH --output=/anvil/scratch/x-bbhusal1/twos_corrected_2026_09_24/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/twos_corrected_2026_09_24/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=96G
#SBATCH --time=00:30:00
#SBATCH -A cis260991-gpu -p gpu-debug
set -euo pipefail
W=/anvil/scratch/x-bbhusal1/twos_corrected_2026_09_24
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$W/repo"
nvidia-smi --query-gpu=name --format=csv,noheader
OUT="$W/out/twos_s42_confirmation_bounded"
mkdir -p "$OUT"
python scripts/eval_token_delta_sae_causal.py \
  --config "$W/inputs/config/qwen3b_twos.yaml" \
  --data-dir "$W/inputs/audit_pool_v3" \
  --adapter-dir "$W/inputs/adapter_s42_fixed" \
  --extract-dir "$W/inputs/v3_deltas_s42" \
  --frontier-dir "$W/inputs/frontier/twos_s42_reselect_discovery" \
  --out-dir "$OUT" \
  --layer 24 --latent-mult 4 --k 8 \
  --batch-size 4 --loss-batch-size 2 --patch-chunk-size 2 --sae-batch-size 1024 \
  --context-modes team --top-sets top5 --control-set control5_active \
  --alphas 0.0,1.0 --max-receivers 0 --max-candidate-donors 16 \
  --receiver-user-file "$W/inputs/split/twos_confirmation_users.txt" \
  --exclude-same-user-donors --token-class-schema cert --seed 42
python scripts/analyze_intervention_endpoints.py "$OUT" --alpha 1.0 --json "$OUT/endpoints.json"
echo TWOS_BOUNDED_DONE
