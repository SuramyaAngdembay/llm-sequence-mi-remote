#!/bin/bash
# Score example-level NLLs from a fixed adapted model over a JSONL pool.
# This is detector benchmarking only: no hidden-state delta extraction.
#SBATCH -A cis230270-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH -J score_examples
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
conda activate "$CONDA_ENV"

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
JSONL_PATH="${JSONL_PATH:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl/all.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
FLUSH_ROWS="${FLUSH_ROWS:-4096}"
USERS_FILE="${USERS_FILE:-}"
LOG_EVERY="${LOG_EVERY:-4096}"

EXTRA_ARGS=()
if [[ -n "$USERS_FILE" ]]; then
  EXTRA_ARGS+=(--users-file "$USERS_FILE")
fi

python scripts/score_adapter_examples.py \
  --config "$CONFIG" \
  --jsonl-path "$JSONL_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-examples "$MAX_EXAMPLES" \
  --flush-rows "$FLUSH_ROWS" \
  --log-every "$LOG_EVERY" \
  "${EXTRA_ARGS[@]}"
