#!/bin/bash
#SBATCH -p ai
#SBATCH -A tra250034-ai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
set -e
MODE=$1
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
export PYTHONPATH=$SCRATCH/lanl/patchlib:$PYTHONPATH   # CVE-2025-32434 torch.load guard patch (trusted ckpts)
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1_8b; D=$SCRATCH/p1/jsonl_r62_$MODE; CFG=configs/qwen3_8b_qlora_session_targeted.yaml
OUT=$P/adapter_$MODE
# auto-resume from latest checkpoint if present
RES=$(ls -dt $OUT/checkpoint-* 2>/dev/null | head -1); RESARG=""
[ -n "$RES" ] && RESARG="--resume-from-checkpoint $RES --skip-rng-state-resume" && echo resuming from $RES
echo "[8b train $MODE] eff-batch = 2 x 2 x 4gpu = 16 (matches P1 3B)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_qlora.py   --config $CFG --data-dir $D --output-dir $OUT --seed 42   --micro-batch-size 2 --grad-accum 2 --max-train-examples 300000 --max-val-examples 2000   --dataloader-workers 16 $RESARG
echo "[8b pool $MODE]"
python - << PY
import json,random,os
rng=random.Random(42); out=[]
for line in open("$D/eval.jsonl"): out.append(line)
for line in open("$D/val.jsonl"):
    if rng.random()<0.12: out.append(line)
os.makedirs("$P/pool_$MODE",exist_ok=True)
open("$P/pool_$MODE/eval.jsonl","w").writelines(out); print("pool",len(out))
PY
echo "[8b extract $MODE] single-GPU"
CUDA_VISIBLE_DEVICES=0 python scripts/extract_adapter_deltas.py --config $CFG --data-dir $P/pool_$MODE --adapter-dir $OUT/adapter --output-dir $P/deltas_$MODE --split eval --pool-unit token --layers 12,18,24 --batch-size 1
echo "[8b roc $MODE]"
python - << PY
import pandas as pd,numpy as np,json
df=pd.read_parquet("$P/deltas_$MODE/example_scores.parquet")
def roc(d):
    y=d["y"].to_numpy();s=d["adapted_nll"].to_numpy();o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(1,len(s)+1)
    r=pd.DataFrame({"s":s,"r":r}).groupby("s")["r"].transform("mean").to_numpy()
    a,b=int((y==1).sum()),int((y==0).sum());return float((r[y==1].sum()-a*(a+1)/2)/(a*b)) if a and b else float("nan")
ud=pd.concat([df[df.y==1],df[(df.y==0)&(df.split=="val")]])
wu=[roc(df[df.user_id==u]) for u in sorted(set(df[df.y==1].user_id)) if df[df.user_id==u].y.nunique()==2]
res={"mode":"$MODE","model":"qwen3-8b","user_disjoint_roc":round(roc(ud),4),"within_user_roc":round(float(np.mean(wu)),4),"n_pos":int((df.y==1).sum())}
os.makedirs("$P/results",exist_ok=True) if False else None
import os; os.makedirs("$P/results",exist_ok=True); json.dump(res,open("$P/results/roc8b_$MODE.json","w"),indent=2); print(res)
PY
if [ "$MODE" = full ] || [ "$MODE" = shuffle_profile ]; then
  echo "[8b sae+attribution $MODE]"
  CUDA_VISIBLE_DEVICES=0 python scripts/train_delta_sae_frontier.py --config configs/token_delta_sae_frontier.yaml --extract-dir $P/deltas_$MODE --out-dir $P/sae_$MODE --device cuda --layers 12,18,24 --latent-multipliers 4 --topk 8 --benign-only --seed 42 --max-rows 2000000 --batch-size 4096
  CUDA_VISIBLE_DEVICES=0 python scripts/audit3b_select_attribute.py --extract-dir $P/deltas_$MODE --data-dir $P/pool_$MODE --frontier-dir $P/sae_$MODE --adapter-dir $OUT/adapter --out-dir $P/audit_$MODE --latent-mult 4 --k 8
fi
echo "8B_${MODE}_DONE"
