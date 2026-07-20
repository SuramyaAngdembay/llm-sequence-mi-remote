#!/bin/bash
# gpu-debug VRAM probe for the r6.2 LOUO per-fold evals (layer-18 branch).
#SBATCH -A cis230270-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -J vram_probe_r62
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
PROJ=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI
module load conda/2026.03
conda activate "${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
export HF_HOME=$PROJ/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_DIR"
python scripts/benchmark_token_causal_vram.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --adapter-dir "$PROJ/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter" \
  --data-dir "$PROJ/outputs/session_jsonl" \
  --out-dir "$PROJ/outputs/vram_probe_r62_louo" \
  --layer 18 \
  --batch-sizes "8,12,16" \
  --sample-mode longest \
  --target-fraction 0.87 \
  --seed 42
echo DONE
