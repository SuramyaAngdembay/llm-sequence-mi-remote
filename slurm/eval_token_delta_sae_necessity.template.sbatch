#!/bin/bash
# Token-level necessity ablation on adapter deltas. This is the negative-intervention
# counterpart to token causal patching.
#SBATCH --job-name=token_delta_necessity
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
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/qwen3b_qlora_session.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp/adapter}"
EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18}"
FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3b/l18_m02_k08}"

LAYER="${LAYER:-18}"
LATENT_MULT="${LATENT_MULT:-2}"
TOPK="${TOPK:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-0}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-0}"

python scripts/eval_token_delta_sae_necessity.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --extract-dir "$EXTRACT_DIR" \
  --frontier-dir "$FRONTIER_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --layer "$LAYER" \
  --latent-mult "$LATENT_MULT" \
  --k "$TOPK" \
  --batch-size "$BATCH_SIZE" \
  --sae-batch-size "$SAE_BATCH_SIZE" \
  --patch-chunk-size "$PATCH_CHUNK_SIZE" \
  --token-delta-dtype "$TOKEN_DELTA_DTYPE" \
  --context-modes "$CONTEXT_MODES" \
  --top-sets "$TOP_SETS" \
  --control-set "$CONTROL_SET" \
  --active-control-min-frac "$ACTIVE_CONTROL_MIN_FRAC" \
  --alphas "$ALPHAS" \
  --max-pairs "$MAX_PAIRS"
