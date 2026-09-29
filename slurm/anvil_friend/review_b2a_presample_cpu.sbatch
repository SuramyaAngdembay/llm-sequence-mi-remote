#!/bin/bash
# Stage B.C, CPU part (docs/review/STAGE_B_PROTOCOL_2026-09-28.md sections 3 and 6): presample the dictionary's
# 2M training rows with the trainer's own functions, and copy the pilot population's layer-26 deltas.
# CPU only; collaborator CPU allocation. No GPU.
#SBATCH --job-name=review_b2a
#SBATCH --output=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=200G
#SBATCH --time=01:30:00
#SBATCH -A tra250034 -p debug
set -euo pipefail
W=/anvil/scratch/x-bbhusal1/cert_review_20260928
mkdir "$W/lock_b2a" 2>/dev/null || { echo "B2a already ran or is running (lock present); exiting"; exit 0; }
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
cd "$W/repo"; cat REVIEW_COMMIT
E=/anvil/scratch/x-sangdembay/pkg4_share/token_deltas
sha256sum scripts/review/presample_sae_rows.py scripts/train_delta_sae_frontier.py scripts/pilot_cache_deltas.py \
  "$W/labels_v1/sampling_labels_v1.parquet" "$W/pilot_populations_v1.json"
echo "=== [1] presample dictionary training rows ==="
/usr/bin/time -v python scripts/review/presample_sae_rows.py --extract-dir "$E" \
  --labels-override "$W/labels_v1/sampling_labels_v1.parquet" --layer 26 --max-rows 2000000 --seed 42 \
  --out-dir "$W/b2a_out/presampled_l26"
echo "=== [2] pilot population deltas ==="
/usr/bin/time -v python scripts/pilot_cache_deltas.py --extract-dir "$E" --layer 26 \
  --populations "$W/pilot_populations_v1.json" --out "$W/b2a_out/pilot_deltas_l26_v1.npz"
sha256sum "$W/b2a_out/presampled_l26/layer_26/chunk_00000.pt" "$W/b2a_out/pilot_deltas_l26_v1.npz"
echo REVIEW_B2A_DONE
