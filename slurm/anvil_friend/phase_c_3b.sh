#!/bin/bash
#SBATCH -p gpu
#SBATCH -A cis260991-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -J pc3b
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1; D=$P/jsonl_r62_full; CFG=configs/qwen3b_qlora_session.yaml
V=$SCRATCH/phase_c_validate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- VALIDATION
# 200 steps (3200 examples / eff. batch 16) on the SAME data and seed, no mask,
# under the default loss path and under the custom fixed-denominator path.
# With nothing masked the two are arithmetically identical, so any divergence
# is an artifact of the custom compute_loss (e.g. transformers v5 normalizing
# a second time) and must abort before the real run.
for DEN in scored_targets all_targets; do
  echo "=== [validate] no mask, denominator=$DEN, 200 steps ==="
  rm -rf $V/$DEN
  python scripts/train_qlora.py --config $CFG --data-dir $D --output-dir $V/$DEN \
    --seed 42 --max-train-examples 3200 --max-val-examples 256 --dataloader-workers 4 \
    --eval-strategy no --save-steps 100000 --target-loss-mask none --loss-denominator $DEN \
    2>&1 | grep -E "'loss':|train_runtime|Error|Traceback" | tail -12
done
echo "=== [validate] GATE: do the two loss paths agree? ==="
python - << 'PY'
import json, re, sys, os, glob
V = os.environ["SCRATCH"] + "/phase_c_validate"
def losses(d):
    f = glob.glob(f"{V}/{d}/**/trainer_state.json", recursive=True)
    if not f:
        return None
    st = json.load(open(sorted(f)[-1]))
    return [(h["step"], h["loss"]) for h in st["log_history"] if "loss" in h]
a, b = losses("scored_targets"), losses("all_targets")
if not a or not b:
    print("GATE_FAILED: no trainer_state.json to compare"); sys.exit(1)
n = min(len(a), len(b))
if n == 0:
    print("GATE_FAILED: no logged losses"); sys.exit(1)
worst = max(abs(x[1] - y[1]) for x, y in zip(a[:n], b[:n]))
rel = worst / max(abs(a[0][1]), 1e-9)
print("steps compared:", n)
print("default path :", [f"{s}:{l:.4f}" for s, l in a[:n][:6]])
print("custom  path :", [f"{s}:{l:.4f}" for s, l in b[:n][:6]])
print(f"max |diff| = {worst:.6f}  (relative {rel:.4%})")
if rel > 0.01:
    print("GATE_FAILED: the custom fixed-denominator path does not reproduce the "
          "default loss when nothing is masked -> normalization differs; do NOT "
          "spend training SU on cells C/D until this is fixed.")
    sys.exit(1)
print("GATE_PASSED")
PY

# ---------------------------------------------------------------- CELLS C/D
echo "=== [train] 3B, profile TARGETS masked, profile CONTEXT retained ==="
OUT=$P/adapter_targetmask
RES=$(ls -dt $OUT/checkpoint-* 2>/dev/null | head -1); RESARG=""
[ -n "$RES" ] && RESARG="--resume-from-checkpoint $RES --skip-rng-state-resume" && echo "resuming from $RES"
python scripts/train_qlora.py --config $CFG --data-dir $D --output-dir $OUT --seed 42 \
  --max-train-examples 300000 --max-val-examples 2000 --dataloader-workers 8 \
  --target-loss-mask profile --loss-denominator all_targets $RESARG
python -c "import json;s=json.load(open('$OUT/train_summary.json'));print('masked_target_frac =',s.get('masked_target_frac'),'| denom =',s.get('loss_denominator'),'| mask =',s.get('target_loss_mask'))"

echo "=== [pool] identical recipe to the factorial arms ==="
python - << PY
import json, random, os
rng = random.Random(42); out = []
for line in open("$D/eval.jsonl"): out.append(line)
for line in open("$D/val.jsonl"):
    if rng.random() < 0.12: out.append(line)
os.makedirs("$P/pool_targetmask", exist_ok=True)
open("$P/pool_targetmask/eval.jsonl", "w").writelines(out); print("pool", len(out))
PY

echo "=== [score] cells C and D from one forward pass ==="
python scripts/score_token_class_decomposition.py --config $CFG \
  --data-dir $P/pool_targetmask --adapter-dir $OUT/adapter \
  --output $SCRATCH/decomp/3b/targetmask/class_scores.parquet \
  --schema cert --batch-size 16 --length-sorted --progress-every 20000

echo "=== [eval] views on the matched population ==="
python scripts/eval_score_views.py \
  --class-scores $SCRATCH/decomp/3b/targetmask/class_scores.parquet \
  --run-name 3b_targetmask --schema cert --out-dir $SCRATCH/decomp/3b/targetmask/views
echo "PHASE_C_3B_DONE"
