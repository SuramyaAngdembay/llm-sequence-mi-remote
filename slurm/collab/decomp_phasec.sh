#!/bin/bash
#SBATCH --job-name=pc3b_score
#SBATCH --output=/anvil/scratch/x-bbhusal1/phase_c_3b_score_%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/phase_c_3b_score_%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00
#SBATCH -A cis260991-gpu -p gpu
# Phase C, 3B: score the adapter trained with the PROFILE-MASKED loss, on the
# identical 159,064-example pool and identical batch-1 recipe used for the
# factorial arms (decomp_run2.sh), so its views are directly comparable to
# decomp/3b/full. This adapter has no cached score, so the numerical
# reproduction gate does not apply; the structural gate does.
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1; CFG=configs/qwen3b_qlora_session.yaml
ADP=$P/adapter_targetmask/adapter; POOL=$P/pool_full
OUT=$SCRATCH/decomp/3b/targetmask; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== adapter provenance ==="; python - <<PY
import json; c=json.load(open("$ADP/adapter_config.json"))
print("base", c["base_model_name_or_path"], "r", c["r"], "alpha", c["lora_alpha"])
PY
echo "=== [1] BOUNDED PILOT: 256 examples, batch 1 ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $POOL \
  --adapter-dir $ADP --output $OUT/pilot_b1/class_scores.parquet --schema cert \
  --max-examples 256 --batch-size 1 2>&1 | grep -v "Loading weights"
echo "=== [2] STRUCTURAL GATE (no cache exists for this adapter, so no reproduction check) ==="
python - << PY
import json, sys
m=json.load(open("$OUT/pilot_b1/class_scores_manifest.json")); ck=m["checks"]
hard=[]
if ck["partition_loss_max_abs_err"]>1e-6: hard.append("partition sum")
if ck["partition_count_mismatches"]: hard.append("partition counts")
if ck["targets_equal_tokens_minus_one"]: hard.append("n_targets != n_tokens-1")
if ck["mean_nll_recon_max_abs_err"]>1e-6: hard.append("mean reconstruction")
print(json.dumps({"checks":ck,"throughput_ex_per_s":m["examples_per_sec"],"peak_gpu_gb":m["peak_gpu_gb"],"gpu":m["gpu"]},indent=2))
if hard: print("GATE_FAILED (structural): "+"; ".join(hard)); sys.exit(1)
print("GATE_PASSED (structural)")
print(f"est_full_pool_hours={159064/max(m['examples_per_sec'],1e-9)/3600:.2f}")
PY
echo "=== [3] FULL POOL (159,064 examples, batch 1) ==="
python scripts/score_token_class_decomposition.py --config $CFG --data-dir $POOL \
  --adapter-dir $ADP --output $OUT/class_scores.parquet --schema cert \
  --batch-size 1 --progress-every 20000 2>&1 | grep -v "Loading weights"
echo "=== [4] EVALUATE VIEWS ==="
python scripts/eval_score_views.py --class-scores $OUT/class_scores.parquet \
  --run-name 3b_targetmask --schema cert --out-dir $OUT/views
echo "DECOMP_3b_targetmask_DONE"
