#!/bin/bash
# Stage B.C + C on Aquaman (protocol amendment A1). Runs after the two extraction shards finish.
# Each step is timed (/usr/bin/time -v); a failing step or gate stops the chain.
set -euo pipefail
W=/data/suramya/cert_review_20260928
cd "$W/repo"; echo "commit $(cat REVIEW_COMMIT)"
source ~/cert-venv/bin/activate
export HF_HOME=/data/suramya/insider_mi/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0}
L="$W/labels_v1"; X="$W/aquaman_l26_v1/merged"; F="$W/stage_c/frontier_calendar_v1"; FS="$W/stage_c/frontier_calendar_v1_selected"
mkdir -p "$W/stage_c"
step() { echo "=== [$(date -Is)] $1"; }
T="/usr/bin/time -v"

step "merge shards"
[ -d "$X" ] || $T python scripts/review/merge_delta_shards.py --shards "$W/aquaman_l26_v1/shard_a" "$W/aquaman_l26_v1/shard_b" --out-dir "$X"

step "presample dictionary rows"
$T python scripts/review/presample_sae_rows.py --extract-dir "$X" --labels-override "$L/sampling_labels_v1.parquet" \
  --layer 26 --max-rows 2000000 --seed 42 --out-dir "$W/stage_c/presampled_l26"

step "dictionary fit (8,192 features, k=4, 20 epochs, seed 42)"
$T python scripts/train_delta_sae_frontier.py --config configs/token_delta_sae_frontier.yaml \
  --extract-dir "$W/stage_c/presampled_l26" --out-dir "$F" --device cuda --batch-size 4096 --epochs 20 \
  --layers 26 --max-rows 2000000 --latent-multipliers 2 --topk 4 --seed 42 --torch-seed 42

step "ranking: discovery positives vs held-out benign negatives"
$T python scripts/reselect_token_sae_features.py --extract-dir "$X" --data-dir "$W/data_r42" \
  --frontier-dir "$F" --out-frontier-dir "$FS" --layer 26 --latent-mult 2 --k 4 \
  --discovery-user-file "$L/discovery_users_v1.txt" --negative-user-file "$L/negative_users_v1.txt" \
  --labels-override "$L/ranking_labels_v1.parquet" --benign-sample-prob 0.25 --batch-size 8192 --device cuda \
  --seed 42 --active-control-min-frac 0.002
SEL=$(python -c "import json;d=json.load(open('$FS/layer_26/m02_k04/reselect_summary.json'))['feature_sets'];print(','.join(map(str,d['top5'])),','.join(map(str,d['control5_active'])))")
read TOP5 CTRL5 <<< "$SEL"; echo "selected $TOP5 control $CTRL5"

step "pilot population deltas"
$T python scripts/pilot_cache_deltas.py --extract-dir "$X" --layer 26 --populations "$W/pilot_populations_v1.json" \
  --out "$W/stage_c/pilot_deltas_l26_v1.npz"

step "pilot scoring (batch 1: no padding; every condition in the same batches)"
$T python scripts/pilot_intervention_specificity.py --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir "$W/data_r42" --adapter-dir /home/suramya/cert-data/r42_adapter --extract-dir "$X" \
  --delta-cache "$W/stage_c/pilot_deltas_l26_v1.npz" --frontier-dir "$FS" --populations "$W/pilot_populations_v1.json" \
  --out-dir "$W/stage_c/pilot" --selected "$TOP5" --control "$CTRL5" --exclude-users "" \
  --labels "$L/ranking_labels_v1.parquet" --insider-file "$W/official_insiders_r42.txt" \
  --conditions zero,rpU_S,randI_S,randD_S,rpU_C,randI_C,randD_C,recon_S,recon_C,orig_S,orig_C,rpO_S,rpO_C,rpU_S@0.5,rpU_C@0.5 \
  --batch-size 1 --loss-batch-size 1 --seed 20260928

step "pre-specified analysis"
python scripts/review/analyze_corrected_pilot.py --run-dir "$W/stage_c/pilot" --answers "$W/answers.tar.bz2" \
  --out "$W/stage_c/pilot_analysis.json"
sha256sum "$F"/layer_26/m02_k04/* "$FS"/layer_26/m02_k04/* "$W/stage_c/pilot/pilot1_rows.csv"
echo STAGE_C_DONE
