#!/bin/bash
#SBATCH -p gpu
#SBATCH -A tra250034-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 08:00:00
set -e
MODE=$1
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$SCRATCH/lanl/patchlib:$PYTHONPATH
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/lanl; D=$P/data/$MODE
echo "[lanl extract $MODE]"
python scripts/extract_adapter_deltas.py --config configs/qwen3b_qlora_session.yaml --data-dir $D --adapter-dir $P/adapter_$MODE/adapter --output-dir $P/deltas_$MODE --split eval --pool-unit token --layers 12,18,24 --batch-size 1 --chunk-examples 512
echo "[lanl AP score $MODE]"
python - << PY
import pandas as pd, numpy as np, hashlib, json, os
df = pd.read_parquet("$P/deltas_$MODE/example_scores.parquet")
def fold(u): return int(hashlib.md5(str(u).encode()).hexdigest(),16)%5
df["fold"]=[fold(u) for u in df.user_id]
def ap(d):
    y=d["y"].to_numpy(); s=d["adapted_nll"].to_numpy()
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    o=np.argsort(-s); y=y[o]; tp=np.cumsum(y); prec=tp/np.arange(1,len(y)+1)
    return float((prec*y).sum()/y.sum())
def auc(d):
    y=d["y"].to_numpy(); s=d["adapted_nll"].to_numpy(); a,b=int(y.sum()),int((1-y).sum())
    if a==0 or b==0: return float("nan")
    r=pd.Series(s).rank(method="average").to_numpy(); return float((r[y==1].sum()-a*(a+1)/2)/(a*b))
seen=df[df.fold<4]; unseen=df[df.fold==4]
res={"mode":"$MODE","seen_ap":round(ap(seen),4),"seen_auc":round(auc(seen),4),"seen_pos":int(seen.y.sum()),"seen_n":len(seen),"unseen_ap":round(ap(unseen),4),"unseen_auc":round(auc(unseen),4),"unseen_pos":int(unseen.y.sum()),"unseen_n":len(unseen)}
os.makedirs("$P/results",exist_ok=True); json.dump(res,open("$P/results/ap_$MODE.json","w"),indent=2); print(res)
PY
echo "LANL_EXTRACT_${MODE}_DONE"
