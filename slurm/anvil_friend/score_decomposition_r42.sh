#!/bin/bash
#SBATCH -p gpu
#SBATCH -A cis260991-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -J dcr42
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd $HOME/llm-sequence-mi-remote
D=/anvil/scratch/x-sangdembay/r42_portability
CFG=configs/qwen3_8b_qlora_session_targeted.yaml
OUT=$SCRATCH/decomp/r42_headline; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
# NOTE: the cached r4.2 scores came from a batch-56 FORWARD pass
# (score_adapter_examples.py chunks only the cross-entropy, not the forward),
# so exact reproduction is not available here the way it was for the r6.2
# factorial. Expect ~1e-3 drift, which the gate tolerates; the scientific
# comparison (full vs behaviour-only) is within one pass, so drift cancels.

echo "=== [1] BOUNDED PILOT: 256 examples, batch 1 ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $D \
  --adapter-dir $D/adapter --output $OUT/pilot_b1/class_scores.parquet --schema cert \
  --max-examples 256 --batch-size 1 --reference-scores $D/reference_scores.parquet 2>&1 | grep -v "Loading weights"

echo "=== [2] GATE ==="
python - << PY
import json, sys
m=json.load(open("$OUT/pilot_b1/class_scores_manifest.json")); ck=m["checks"]; r=ck.get("reference",{})
hard=[]
if ck["partition_loss_max_abs_err"]>1e-6: hard.append("partition sum")
if ck["partition_count_mismatches"]: hard.append("partition counts")
if ck["targets_equal_tokens_minus_one"]: hard.append("n_targets != n_tokens-1")
if ck["mean_nll_recon_max_abs_err"]>1e-6: hard.append("mean reconstruction")
if r.get("n_token_mismatches",1): hard.append(f"token counts vs cache ({r.get('n_token_mismatches')})")
if r.get("nll_mean_abs_diff",9)>1e-2: hard.append(f"mean|dNLL|={r.get('nll_mean_abs_diff'):.4f}")
if r.get("spearman_like_rank_corr",0)<0.98: hard.append(f"rankcorr={r.get('spearman_like_rank_corr'):.4f}")
share=ck["class_target_share"]
for c in ("DAY","PSY","SES"):
    if share.get(c,0)<=0: hard.append(f"class {c} empty")
print(json.dumps({"checks":ck,"throughput_ex_per_s":m["examples_per_sec"],"gpu":m["gpu"]},indent=2))
if hard: print("GATE_FAILED: "+"; ".join(hard)); sys.exit(1)
print(f"GATE_PASSED  join={ck.get('reference_join_key')} mean|dNLL|={r['nll_mean_abs_diff']:.3e} rankcorr={r['spearman_like_rank_corr']:.8f}")
print(f"est_full_pool_hours={40519/max(m['examples_per_sec'],1e-9)/3600:.2f}")
PY

echo "=== [3] FULL POOL (40,519 rows, batch 1) ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $D \
  --adapter-dir $D/adapter --output $OUT/class_scores.parquet --schema cert \
  --batch-size 1 --reference-scores $D/reference_scores.parquet --progress-every 10000 2>&1 | grep -v "Loading weights"

echo "=== [4a] POOLED user-disjoint (anchors the published r4.2 numbers) ==="
python - << PY
import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0,"scripts")
from token_class_decomposition import CERT_VIEWS
d=pd.read_parquet("$OUT/class_scores.parquet")
ref=pd.read_parquet("$D/reference_scores.parquet")[["example_id","adapted_nll"]].rename(columns={"adapted_nll":"cached"})
d=d.merge(ref,on="example_id",how="left")
rows=[]
for name,cls in list(CERT_VIEWS.items())+[("full_cached",None)]:
    if name=="full_cached": s=d["cached"].to_numpy(float)
    else:
        loss=sum(d[f"loss_sum_{c}"] for c in cls if f"loss_sum_{c}" in d); n=sum(d[f"n_{c}"] for c in cls if f"n_{c}" in d)
        s=(loss/n.replace(0,np.nan)).to_numpy(float)
    y=d["y"].to_numpy(int); m=np.isfinite(s)
    u=pd.DataFrame({"u":d["user_id"],"y":y,"s":s}).groupby("u").agg(y=("y","max"),s=("s","max"))
    rows.append({"view":name,"day_roc":roc_auc_score(y[m],s[m]),"day_ap":average_precision_score(y[m],s[m]),
                 "user_roc":roc_auc_score(u.y,u.s),"user_ap":average_precision_score(u.y,u.s)})
r=pd.DataFrame(rows); r.to_csv("$OUT/pooled_user_disjoint.csv",index=False)
print(r.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\npublished r4.2 user-disjoint: day ROC 0.668, user ROC 0.565")
PY

echo "=== [4b] fold-aligned over the 60 malicious users (60 bootstrap clusters) ==="
python scripts/eval_score_views.py --class-scores $OUT/class_scores.parquet \
  --run-name r42_headline --schema cert --out-dir $OUT/views \
  --reference-scores $D/reference_scores.parquet
echo "DECOMP_R42_DONE"
