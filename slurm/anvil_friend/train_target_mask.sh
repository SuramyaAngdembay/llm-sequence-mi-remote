#!/bin/bash
# Phase C, cells C and D: train one adapter whose profile TARGETS are excluded
# from the loss (the profile stays in the input context), then score the same
# audit pool with the class decomposition so both scoring views come out of one
# job. See docs/SCORE_DECOMPOSITION_PHASE_C_SPEC.md.
#
#   sbatch train_target_mask.sh 3b     # ~3 SU  (1xA100, ~3 h)
#   sbatch train_target_mask.sh 8b     # ~52 SU (4xH100, ~13 h)
#
# Everything except the loss mask is held identical to the `full` factorial arm:
# same data dir, same 300k/2k caps and example order, seed 42, effective batch
# 16, same LoRA/quantization config, same max_seq_len, same audit-pool recipe.
#
# NOT to be launched without an explicit go-ahead on the collaborator's
# allocation.
set -e
SCALE=${1:?usage: train_target_mask.sh {3b|8b}}
MODE=full                       # profile context present; only the loss changes
TAG=targetmask

module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
cd $HOME/llm-sequence-mi-remote

D=$SCRATCH/p1/jsonl_r62_$MODE
if [ "$SCALE" = "8b" ]; then
  P=$SCRATCH/p1_8b; CFG=configs/qwen3_8b_qlora_session_targeted.yaml
  TRAIN="torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_qlora.py --micro-batch-size 2 --grad-accum 2"
  SBS=8
else
  P=$SCRATCH/p1;    CFG=configs/qwen3b_qlora_session.yaml
  TRAIN="python scripts/train_qlora.py"
  SBS=16
fi
OUT=$P/adapter_${TAG}
RES=$(ls -dt $OUT/checkpoint-* 2>/dev/null | head -1); RESARG=""
[ -n "$RES" ] && RESARG="--resume-from-checkpoint $RES --skip-rng-state-resume" && echo "resuming from $RES"

echo "=== [1] train: profile TARGETS masked, profile CONTEXT retained ==="
$TRAIN --config $CFG --data-dir $D --output-dir $OUT --seed 42 \
  --max-train-examples 300000 --max-val-examples 2000 --dataloader-workers 16 \
  --target-loss-mask profile --loss-denominator all_targets $RESARG
python -c "import json;s=json.load(open('$OUT/train_summary.json'));print('masked_target_frac =',s.get('masked_target_frac'),'| denom =',s.get('loss_denominator'))"

echo "=== [2] audit pool (identical recipe to the factorial arms) ==="
python - << PY
import json, random, os
rng = random.Random(42); out = []
for line in open("$D/eval.jsonl"): out.append(line)
for line in open("$D/val.jsonl"):
    if rng.random() < 0.12: out.append(line)
os.makedirs("$P/pool_${TAG}", exist_ok=True)
open("$P/pool_${TAG}/eval.jsonl", "w").writelines(out)
print("pool", len(out))
PY

echo "=== [3] score with the class decomposition (cells C and D) ==="
python scripts/score_token_class_decomposition.py --config $CFG \
  --data-dir $P/pool_${TAG} --adapter-dir $OUT/adapter \
  --output $SCRATCH/decomp/$SCALE/${TAG}/class_scores.parquet \
  --schema cert --batch-size $SBS --length-sorted --progress-every 20000

echo "=== [4] evaluate views on the matched population ==="
python scripts/eval_score_views.py \
  --class-scores $SCRATCH/decomp/$SCALE/${TAG}/class_scores.parquet \
  --run-name ${SCALE}_${TAG} --schema cert \
  --out-dir $SCRATCH/decomp/$SCALE/${TAG}/views
echo "TARGETMASK_${SCALE}_DONE"
