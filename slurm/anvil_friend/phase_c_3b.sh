#!/bin/bash
#SBATCH -p gpu
#SBATCH -A cis260991-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 16:00:00
#SBATCH -J pc3b2
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1; D=$P/jsonl_r62_full; CFG=configs/qwen3b_qlora_session.yaml
V=$SCRATCH/phase_c_validate
nvidia-smi --query-gpu=name --format=csv,noheader

# --------------------------------------------------------------- GATE
# 200 steps, nothing masked, under the HF-default loss path and under the
# fixed-denominator path. With nothing masked they are arithmetically
# identical, so any divergence is an artifact of compute_loss. transformers
# 5.16.1 skips `loss / grad_accum` whenever num_items_in_batch is supplied
# (verified in Trainer.training_step), so a per-micro-batch mean comes out
# grad_accum times too large - which is exactly what job 20810414 measured
# (4.0x at grad_accum=4).
for DEN in scored_targets all_targets; do
  echo "=== [gate] no mask, denominator=$DEN, 200 steps ==="
  rm -rf $V/$DEN
  python scripts/train_qlora.py --config $CFG --data-dir $D --output-dir $V/$DEN \
    --seed 42 --max-train-examples 3200 --max-val-examples 256 --dataloader-workers 4 \
    --eval-strategy no --save-steps 100000 --target-loss-mask none --loss-denominator $DEN \
    2>&1 | grep -E "'loss':|train_runtime|Error|Traceback" | tail -12
done
python - << 'PY'
import json, sys, os, glob
V = os.environ["SCRATCH"] + "/phase_c_validate"
def losses(d):
    f = glob.glob(f"{V}/{d}/**/trainer_state.json", recursive=True)
    if not f: return None
    return [(h["step"], h["loss"]) for h in json.load(open(sorted(f)[-1]))["log_history"] if "loss" in h]
a, b = losses("scored_targets"), losses("all_targets")
if not a or not b: print("GATE_FAILED: no logged losses"); sys.exit(1)
n = min(len(a), len(b))
worst = max(abs(x[1]-y[1]) for x, y in zip(a[:n], b[:n]))
rel = worst / max(abs(a[0][1]), 1e-9)
print("default:", [f"{s}:{l:.4f}" for s, l in a[:n][:5]])
print("custom :", [f"{s}:{l:.4f}" for s, l in b[:n][:5]])
print(f"max |diff| = {worst:.6f}  (relative {rel:.4%})")
if rel > 0.01:
    print("GATE_FAILED: fixed-denominator path still does not reproduce the default"); sys.exit(1)
print("GATE_PASSED")
PY

# --------------------------------------------------------------- TRAIN C/D
echo "=== [train] 3B, profile TARGETS excluded from the loss, CONTEXT retained ==="
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
echo "PHASE_C_3B_TRAIN_DONE  (scoring runs as a separate batch-1 job)"
