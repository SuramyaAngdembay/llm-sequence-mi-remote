#!/bin/bash
# EXPLORATORY: capture the actual H6 layer-26 states (no behavioural outcome). Protocol: research/imperfect_interventions_2026-09-27/h6_states/.
# Collaborator allocation. Copies may be queued on several partitions; an atomic mkdir lock lets exactly one run.
#SBATCH --job-name=h6_capture
#SBATCH --output=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=96G
#SBATCH --time=00:25:00
set -euo pipefail
P=/anvil/scratch/x-bbhusal1/h6_states_2026_09_27
S=/anvil/scratch/x-sangdembay/pkg4_share
if ! mkdir "$P/lock" 2>/dev/null; then echo "another copy holds $P/lock; exiting without work"; exit 0; fi
echo "$SLURM_JOB_ID $(hostname) $(date -Is)" > "$P/lock/owner"
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$P/repo"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum scripts/h6_state_capture.py scripts/h6_profile_patching.py
/usr/bin/time -v python -u scripts/h6_state_capture.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir "$S/session_jsonl_r42" --adapter-dir "$S/adapter" --extract-dir "$S/token_deltas" \
  --frontier-dir "$S/frontier" --split-dir "$S/user_splits_r42" \
  --h6-dir /anvil/scratch/x-bbhusal1/h6_2026_09_27/out --out-dir "$P/out" \
  --batch-size 6 --smoke-pairs 2
date -Is > "$P/DONE"
echo H6_CAPTURE_JOB_DONE
