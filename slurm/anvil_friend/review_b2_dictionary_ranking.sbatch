#!/bin/bash
# Stage B.C (docs/review/STAGE_B_PROTOCOL_2026-09-28.md sections 3-5): refit the r4.2 layer-26 token SAE
# (8,192 features, k=4, 2M-row cap, seed 42) with positive preservation limited to corrected-positive days of
# discovery users, then rank features on discovery positives vs held-out benign negatives and choose
# top5 / control5_active. Cap 1.25 GPU-h (one A100, 75 min wall). Collaborator allocation only.
#SBATCH --job-name=review_b2
#SBATCH --output=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=180G
#SBATCH --time=01:15:00
#SBATCH -A cis260991-gpu -p gpu
set -euo pipefail
W=/anvil/scratch/x-bbhusal1/cert_review_20260928
mkdir "$W/lock_b2" 2>/dev/null || { echo "B2 already ran or is running (lock present); exiting"; exit 0; }
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export PYTHONUNBUFFERED=1
cd "$W/repo"                                  # review branch snapshot (commit recorded below)
cat REVIEW_COMMIT
E=/anvil/scratch/x-sangdembay/pkg4_share/token_deltas
L="$W/labels_v1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum scripts/train_delta_sae_frontier.py scripts/reselect_token_sae_features.py scripts/sae_core.py \
  "$L"/*.parquet "$L"/*.txt "$E/example_scores.parquet" "$E/chunk_manifest.csv"

echo "=== [1] dictionary refit ==="
/usr/bin/time -v python scripts/train_delta_sae_frontier.py --config configs/token_delta_sae_frontier.yaml \
  --extract-dir "$E" --out-dir "$W/b2_out/frontier_calendar_v1" --device cuda --batch-size 4096 --epochs 20 \
  --layers 26 --max-rows 2000000 --latent-multipliers 2 --topk 4 --seed 42 \
  --labels-override "$L/sampling_labels_v1.parquet"

echo "=== [2] ranking: discovery positives vs held-out benign negatives ==="
/usr/bin/time -v python scripts/reselect_token_sae_features.py --extract-dir "$E" \
  --data-dir /anvil/scratch/x-sangdembay/pkg4_share/session_jsonl_r42 \
  --frontier-dir "$W/b2_out/frontier_calendar_v1" --out-frontier-dir "$W/b2_out/frontier_calendar_v1_selected" \
  --layer 26 --latent-mult 2 --k 4 --discovery-user-file "$L/discovery_users_v1.txt" \
  --negative-user-file "$L/negative_users_v1.txt" --labels-override "$L/ranking_labels_v1.parquet" \
  --benign-sample-prob 0.25 --batch-size 8192 --device cuda --seed 42 --active-control-min-frac 0.002
sha256sum "$W"/b2_out/frontier_calendar_v1*/layer_26/m02_k04/*
echo REVIEW_B2_DONE
