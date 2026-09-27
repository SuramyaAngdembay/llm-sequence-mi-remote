#!/bin/bash
# EXPLORATORY: bounded behavioural pilot of declared feasible interventions (only after a passed precheck).
# Protocol: research/imperfect_interventions_2026-09-27/h6_states/PRECHECK_PROTOCOL.md, Part 3b. Atomic mkdir lock.
#SBATCH --job-name=h6_pilot
#SBATCH --output=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=96G
#SBATCH --time=00:25:00
set -euo pipefail
P=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27
S=/anvil/scratch/x-sangdembay/pkg4_share
test -f "$P/PRECHECK_PASSED" || { echo "no PRECHECK_PASSED marker; exiting without work"; exit 0; }
if ! mkdir "$P/pilot_lock" 2>/dev/null; then echo "another pilot copy holds the lock; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(hostname) $(date -Is)" > "$P/pilot_lock/owner"
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$P/repo"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum scripts/h6_intervention_pilot.py "$P/projection/projected_states.npz" "$P/projection/projection_records.json"
/usr/bin/time -v python -u scripts/h6_intervention_pilot.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir "$S/session_jsonl_r42" --adapter-dir "$S/adapter" --extract-dir "$S/token_deltas" \
  --frontier-dir "$S/frontier" --split-dir "$S/user_splits_r42" \
  --h6-dir /anvil/scratch/x-bbhusal1/h6_2026_09_27/out --capture "$P/out" --projection "$P/projection" \
  --out-dir "$P/pilot_out" --batch-size 6 --smoke-pairs 2
date -Is > "$P/PILOT_DONE"
echo H6_PILOT_JOB_DONE
