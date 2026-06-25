#!/bin/bash
# Token-level adapter-delta extraction. This is the decisive escalation after mean-pooled patching failed.
# Write bulky token deltas to scratch-like storage; keep summaries in project space.
#SBATCH --job-name=token_delta_extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --account=cis230270-ai
#SBATCH --partition=ai

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18}"
SPLIT="${SPLIT:-eval}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
CHUNK_EXAMPLES="${CHUNK_EXAMPLES:-1024}"
LAYERS="${LAYERS:-18}"

python scripts/extract_adapter_deltas.py \
  --config configs/qwen3b_qlora_session.yaml \
  --data-dir "$DATA_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --max-examples "$MAX_EXAMPLES" \
  --chunk-examples "$CHUNK_EXAMPLES" \
  --layers "$LAYERS" \
  --pool-unit token
