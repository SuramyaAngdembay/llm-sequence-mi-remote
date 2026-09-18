#!/bin/bash
set -e
SCALE=$1; MODE=${2:-full}
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd $HOME/llm-sequence-mi-remote
if [ "$SCALE" = "8b" ]; then P=$SCRATCH/p1_8b; CFG=configs/qwen3_8b_qlora_session_targeted.yaml
else P=$SCRATCH/p1; CFG=configs/qwen3b_qlora_session.yaml; fi
ADP=$P/adapter_$MODE/adapter; POOL=$P/pool_$MODE; REF=$P/deltas_$MODE/example_scores.parquet
OUT=$SCRATCH/decomp/$SCALE/$MODE; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
# batch size 1 = the convention the cached scores were produced under; combined
# with matched GPU architecture this reproduces them to ~1e-8 (verified at 3B).

echo "=== [1] BOUNDED PILOT: 256 examples, batch 1, matched hardware ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $POOL \
  --adapter-dir $ADP --output $OUT/pilot_b1/class_scores.parquet --schema cert \
  --max-examples 256 --batch-size 1 --reference-scores $REF 2>&1 | grep -v "Loading weights"

echo "=== [2] GATE ==="
python - << PY
import json, sys
m=json.load(open("$OUT/pilot_b1/class_scores_manifest.json")); ck=m["checks"]; r=ck.get("reference",{})
hard=[]
if ck["partition_loss_max_abs_err"]>1e-6: hard.append("partition sum")
if ck["partition_count_mismatches"]: hard.append("partition counts")
if ck["targets_equal_tokens_minus_one"]: hard.append("n_targets != n_tokens-1")
if ck["mean_nll_recon_max_abs_err"]>1e-6: hard.append("mean reconstruction")
if r.get("n_token_mismatches",1): hard.append("token counts vs cache")
# numerical-agreement bound (not a structural check): identical code on matched
# hardware reproduced the cache to ~1e-8 at 3B; cross-architecture drift showed
# up as ~3e-3 mean. Anything beyond 1e-2 mean means something other than kernel
# nondeterminism is going on.
soft=[]
if r.get("nll_mean_abs_diff",9)>1e-2: soft.append(f"mean|dNLL|={r.get('nll_mean_abs_diff'):.4f}")
if r.get("spearman_like_rank_corr",0)<0.98: soft.append(f"rank corr={r.get('spearman_like_rank_corr'):.4f}")
print(json.dumps({"checks":ck,"throughput_ex_per_s":m["examples_per_sec"],"peak_gpu_gb":m["peak_gpu_gb"],"gpu":m["gpu"]},indent=2))
if hard: print("GATE_FAILED (structural): "+"; ".join(hard)); sys.exit(1)
if soft: print("GATE_FAILED (numerical): "+"; ".join(soft)); sys.exit(1)
print(f"GATE_PASSED  reproduction: mean|dNLL|={r['nll_mean_abs_diff']:.3e} max={r['nll_max_abs_diff']:.3e} rankcorr={r['spearman_like_rank_corr']:.10f}")
print(f"est_full_pool_hours={159064/max(m['examples_per_sec'],1e-9)/3600:.2f}")
PY

echo "=== [3] FULL POOL (159,064 examples, batch 1) ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $POOL \
  --adapter-dir $ADP --output $OUT/class_scores.parquet --schema cert \
  --batch-size 1 --reference-scores $REF --progress-every 20000 2>&1 | grep -v "Loading weights"

echo "=== [4] EVALUATE VIEWS (recomputed full AND the published cached score) ==="
python scripts/eval_score_views.py --class-scores $OUT/class_scores.parquet \
  --run-name ${SCALE}_${MODE} --schema cert --out-dir $OUT/views --reference-scores $REF
echo "DECOMP_${SCALE}_${MODE}_DONE"
