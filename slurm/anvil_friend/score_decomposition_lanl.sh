#!/bin/bash
#SBATCH -p ai
#SBATCH -A tra250034-ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -J dclanl
set -e
MODE=${1:-full}
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd $HOME/llm-sequence-mi-remote
L=$SCRATCH/lanl; CFG=configs/qwen3b_qlora_session.yaml
ADP=$L/adapter_$MODE/adapter; DATA=$L/data/$MODE; REF=$L/deltas_$MODE/example_scores.parquet
OUT=$SCRATCH/decomp/lanl/$MODE; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
# LANL windows average ~1256 tokens (max 1503) vs CERT's 293, so this is ~4x the
# work per example at 1/3 the example count. batch 1 = the cached convention.

echo "=== [1] BOUNDED PILOT: 256 windows, batch 1, LANL field schema ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $DATA \
  --adapter-dir $ADP --output $OUT/pilot_b1/class_scores.parquet --schema lanl \
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
if r.get("nll_mean_abs_diff",9)>1e-2: hard.append(f"mean|dNLL|={r.get('nll_mean_abs_diff'):.4f}")
if r.get("spearman_like_rank_corr",0)<0.98: hard.append(f"rankcorr={r.get('spearman_like_rank_corr'):.4f}")
# LANL-specific: identity and behaviour classes must both be populated, and the
# " | " separators land in OTHER by design - so OTHER is expected NON-zero here,
# unlike CERT.
share=ck["class_target_share"]
for c in ("ID_USER","ID_HOST","BEHAV","HOUR"):
    if share.get(c,0)<=0: hard.append(f"class {c} empty")
print(json.dumps({"checks":ck,"throughput_ex_per_s":m["examples_per_sec"],"peak_gpu_gb":m["peak_gpu_gb"],"gpu":m["gpu"]},indent=2))
if hard: print("GATE_FAILED: "+"; ".join(hard)); sys.exit(1)
print(f"GATE_PASSED  reproduction: mean|dNLL|={r['nll_mean_abs_diff']:.3e} rankcorr={r['spearman_like_rank_corr']:.10f}")
print(f"est_full_pool_hours={52973/max(m['examples_per_sec'],1e-9)/3600:.2f}")
PY

echo "=== [3] FULL POOL (52,973 windows, batch 1) ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $DATA \
  --adapter-dir $ADP --output $OUT/class_scores.parquet --schema lanl \
  --batch-size 1 --reference-scores $REF --progress-every 10000 2>&1 | grep -v "Loading weights"

echo "=== [4] EVALUATE under LANL's own seen/unseen protocol ==="
python scripts/eval_score_views_lanl.py --class-scores $OUT/class_scores.parquet \
  --data-dir $DATA --run-name lanl_${MODE} --out-dir $OUT/views \
  --reference-scores $REF --published-ap-json $L/results/ap_${MODE}.json
echo "DECOMP_LANL_${MODE}_DONE"
