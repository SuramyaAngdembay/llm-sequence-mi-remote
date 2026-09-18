#!/bin/bash
#SBATCH -p gpu
#SBATCH -A cis260991-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 00:25:00
#SBATCH -J r42bt
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd $HOME/llm-sequence-mi-remote
D=/anvil/scratch/x-sangdembay/r42_portability; CFG=configs/qwen3_8b_qlora_session_targeted.yaml
OUT=$SCRATCH/decomp/r42_batchtest; mkdir -p $OUT
# CONTROLLED TEST of the V17 explanation. If the batch-56 FORWARD pass is why
# our batch-1 scores differ from the cached r4.2 scores, then rescoring the same
# examples at batch 56 should move them back toward the cache. This demonstrates
# the cause instead of inferring it from a length correlation.
for BS in 56 8; do
  python scripts/score_token_class_decomposition.py --config $CFG --data-dir $D \
    --adapter-dir $D/adapter --output $OUT/b$BS/class_scores.parquet --schema cert \
    --max-examples 256 --batch-size $BS --reference-scores $D/reference_scores.parquet 2>&1 | grep -v "Loading weights" | tail -3
done
python - << 'PY'
import json, os, pandas as pd, numpy as np
OUT=os.environ["SCRATCH"]+"/decomp/r42_batchtest"
P1=os.environ["SCRATCH"]+"/decomp/r42_headline/pilot_b1/class_scores_manifest.json"
print("\n=== agreement with the cached (batch-56 forward) scores ===")
rows=[]
for tag,path in [("batch 1",P1),("batch 8",f"{OUT}/b8/class_scores_manifest.json"),("batch 56",f"{OUT}/b56/class_scores_manifest.json")]:
    r=json.load(open(path))["checks"]["reference"]
    rows.append({"scoring batch":tag,"mean|dNLL|":r["nll_mean_abs_diff"],"max|dNLL|":r["nll_max_abs_diff"],"rank corr":r["spearman_like_rank_corr"]})
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f"{x:.3e}"))
print("\nIf padding in the cached run is the cause, agreement should IMPROVE monotonically toward batch 56.")
PY
echo "R42_BATCHTEST_DONE"
