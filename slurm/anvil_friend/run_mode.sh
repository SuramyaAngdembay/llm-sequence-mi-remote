#!/bin/bash
#SBATCH -p gpu
#SBATCH -A tra250034-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 24:00:00
set -e
MODE=$1
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1; D=$P/jsonl_r62_$MODE; CFG=configs/qwen3b_qlora_session.yaml
echo "[train $MODE]"
python scripts/train_qlora.py --config $CFG --data-dir $D --output-dir $P/adapter_$MODE --seed 42 --max-train-examples 300000 --max-val-examples 2000 --dataloader-workers 2
# audit pool: all eval (positive users, all days) + 12% val benign (unseen)
python - << PY
import json, random, os
rng=random.Random(42); out=[]
for line in open("$D/eval.jsonl"): out.append(line)
for line in open("$D/val.jsonl"):
    if rng.random()<0.12: out.append(line)
os.makedirs("$P/pool_$MODE", exist_ok=True)
open("$P/pool_$MODE/eval.jsonl","w").writelines(out)
print("pool:", len(out))
PY
echo "[extract $MODE]"
python scripts/extract_adapter_deltas.py --config $CFG --data-dir $P/pool_$MODE --adapter-dir $P/adapter_$MODE/adapter --output-dir $P/deltas_$MODE --split eval --pool-unit token --layers 12,18,24 --batch-size 1
# ROC readouts
python - << PY
import pandas as pd, numpy as np, json
df=pd.read_parquet("$P/deltas_$MODE/example_scores.parquet")
def roc(d):
    y=d["y"].to_numpy(); s=d["adapted_nll"].to_numpy()
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    r=pd.DataFrame({"s":s,"r":r}).groupby("s")["r"].transform("mean").to_numpy()
    np_,nn=int((y==1).sum()),int((y==0).sum())
    return float((r[y==1].sum()-np_*(np_+1)/2)/(np_*nn)) if np_ and nn else float("nan")
ud=pd.concat([df[df.y==1], df[(df.y==0)&(df.split=="val")]])
wu=[roc(df[df.user_id==u]) for u in sorted(set(df[df.y==1].user_id)) if df[df.user_id==u].y.nunique()==2]
res={"mode":"$MODE","user_disjoint_roc":round(roc(ud),4),"within_user_roc":round(float(np.mean(wu)),4),"n_pos":int((df.y==1).sum())}
json.dump(res, open("$P/results/roc_$MODE.json","w"), indent=2); print(res)
PY
# attribution only where profile tokens exist
if [ "$MODE" = "full" ] || [ "$MODE" = "shuffle_profile" ]; then
  echo "[sae+attribution $MODE]"
  python scripts/train_delta_sae_frontier.py --config configs/token_delta_sae_frontier.yaml --extract-dir $P/deltas_$MODE --out-dir $P/sae_$MODE --device cuda --layers 12,18,24 --latent-multipliers 4 --topk 8 --benign-only --seed 42
  python scripts/audit3b_select_attribute.py --extract-dir $P/deltas_$MODE --data-dir $P/pool_$MODE --frontier-dir $P/sae_$MODE --adapter-dir $P/adapter_$MODE/adapter --out-dir $P/audit_$MODE --latent-mult 4 --k 8
fi
echo "P1_${MODE}_DONE"
