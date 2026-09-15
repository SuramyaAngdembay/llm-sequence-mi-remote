#!/bin/bash
#SBATCH -p ai
#SBATCH -A tra250034-ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 24:00:00
set -e
MODE=$1
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export PYTHONPATH=$SCRATCH/lanl/patchlib:$PYTHONPATH
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/lanl
D=$P/data/$MODE
mkdir -p $P/data $P/results
# stage data from x-sangdembay scratch (read-only) into own scratch once
if [ ! -f $D/train.jsonl ]; then
  mkdir -p $D
  cp /anvil/scratch/x-sangdembay/lanl2015/lanl_conditions/$MODE/{train,val,eval}.jsonl $D/
fi
RESUME=$(ls -dt $P/adapter_$MODE/checkpoint-* 2>/dev/null | head -1)
RESARG=""
if [ -n "$RESUME" ]; then RESARG="--resume-from-checkpoint $RESUME --skip-rng-state-resume"; echo "resuming from $RESUME"; fi
echo "[lanl train $MODE]"
python scripts/train_qlora.py --config configs/qwen3b_qlora_session.yaml --data-dir $D --output-dir $P/adapter_$MODE --seed 42 --max-train-examples 300000 --max-val-examples 2000 --dataloader-workers 2 $RESARG
echo "[lanl extract $MODE]"
python scripts/extract_adapter_deltas.py --config configs/qwen3b_qlora_session.yaml --data-dir $D --adapter-dir $P/adapter_$MODE/adapter --output-dir $P/deltas_$MODE --split eval --pool-unit token --layers 12,18,24 --batch-size 1
echo "[lanl AP score $MODE]"
python - << PY
import pandas as pd, numpy as np, hashlib, json
df = pd.read_parquet("$P/deltas_$MODE/example_scores.parquet")
def fold(u): return int(hashlib.md5(str(u).encode()).hexdigest(),16)%5
df["fold"]=[fold(u) for u in df.user_id]
def ap(d):
    y=d["y"].to_numpy(); s=d["adapted_nll"].to_numpy()
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    o=np.argsort(-s); y=y[o]
    tp=np.cumsum(y); prec=tp/np.arange(1,len(y)+1)
    return float((prec*y).sum()/y.sum())
def auc(d):
    y=d["y"].to_numpy(); s=d["adapted_nll"].to_numpy()
    np_,nn=int(y.sum()),int((1-y).sum())
    if np_==0 or nn==0: return float("nan")
    r=pd.Series(s).rank(method="average").to_numpy()
    return float((r[y==1].sum()-np_*(np_+1)/2)/(np_*nn))
seen=df[df.fold<4]; unseen=df[df.fold==4]
res={"mode":"$MODE",
     "seen_ap":round(ap(seen),4),"seen_auc":round(auc(seen),4),
     "seen_pos":int(seen.y.sum()),"seen_n":len(seen),
     "unseen_ap":round(ap(unseen),4),"unseen_auc":round(auc(unseen),4),
     "unseen_pos":int(unseen.y.sum()),"unseen_n":len(unseen)}
json.dump(res,open("$P/results/ap_$MODE.json","w"),indent=2); print(res)
PY
echo "LANL_${MODE}_DONE"
