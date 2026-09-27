#!/bin/bash
# EXPLORATORY H6: where profile-dependent behaviour prediction arises (activation, component and SAE-mediation patching).
# docs/HYPOTHESIS_LEDGER_2026-09-25.md; budget 0.5 GPU-h (1 A100, 30 min); follow-up cap 1 GPU-h.
# Collaborator allocation only. Inputs frozen in prep/ before submission.
#SBATCH --job-name=h6_profile_gpu
#SBATCH --output=/anvil/scratch/x-bbhusal1/h6_2026_09_27/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/h6_2026_09_27/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=96G
#SBATCH --time=00:30:00
#SBATCH -A cis260991-gpu -p gpu
set -euo pipefail
P=/anvil/scratch/x-bbhusal1/h6_2026_09_27
S=/anvil/scratch/x-sangdembay/pkg4_share
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$P/repo"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$S/adapter/adapter_model.safetensors" "$S/frontier/layer_26/m02_k04/delta_sae_model.pt" "$S/session_jsonl_r42/eval.jsonl"
python -c "import transformers, peft, torch, bitsandbytes; print('versions', transformers.__version__, peft.__version__, torch.__version__, bitsandbytes.__version__)"
python -u scripts/h6_profile_patching.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir "$S/session_jsonl_r42" --adapter-dir "$S/adapter" --extract-dir "$S/token_deltas" \
  --frontier-dir "$S/frontier" --split-dir "$S/user_splits_r42" --out-dir "$P/out_gpu" \
  --batch-size 6 --smoke-pairs 2 --wall-budget-s 1500
echo H6_JOB_DONE
