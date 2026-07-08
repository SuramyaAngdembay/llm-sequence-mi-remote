#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_v1}"
COMMON_TOP_SETS="${COMMON_TOP_SETS:-top5}"
COMMON_CONTROL_SET="${COMMON_CONTROL_SET:-control5_active}"
COMMON_ACTIVE_CONTROL_MIN_FRAC="${COMMON_ACTIVE_CONTROL_MIN_FRAC:-0.002}"
COMMON_CONTEXT_MODES="${COMMON_CONTEXT_MODES:-team,role,dept,dept_role}"
COMMON_N_BOOTSTRAP="${COMMON_N_BOOTSTRAP:-4000}"
COMMON_BATCH_SIZE="${COMMON_BATCH_SIZE:-24}"
COMMON_PATCH_CHUNK_SIZE="${COMMON_PATCH_CHUNK_SIZE:-24}"
COMMON_CAUSAL_MEM="${COMMON_CAUSAL_MEM:-480G}"
COMMON_TOKEN_DELTA_DTYPE="${COMMON_TOKEN_DELTA_DTYPE:-float32}"

TAG="${TAG:-l26_m02_k04_top5_control5_active}"

echo "Submitting Qwen3-8B r4.2 native active-control audit..."
echo "tag=${TAG}"
echo "top_sets=${COMMON_TOP_SETS}"
echo "control_set=${COMMON_CONTROL_SET}"
echo "active_control_min_frac=${COMMON_ACTIVE_CONTROL_MIN_FRAC}"
echo "batch_size=${COMMON_BATCH_SIZE}"
echo "patch_chunk_size=${COMMON_PATCH_CHUNK_SIZE}"
echo "causal_mem=${COMMON_CAUSAL_MEM}"
echo

TOP_SETS="$COMMON_TOP_SETS" \
CONTROL_SET="$COMMON_CONTROL_SET" \
ACTIVE_CONTROL_MIN_FRAC="$COMMON_ACTIVE_CONTROL_MIN_FRAC" \
CONTEXT_MODES="$COMMON_CONTEXT_MODES" \
N_BOOTSTRAP="$COMMON_N_BOOTSTRAP" \
BATCH_SIZE="$COMMON_BATCH_SIZE" \
PATCH_CHUNK_SIZE="$COMMON_PATCH_CHUNK_SIZE" \
CAUSAL_MEM="$COMMON_CAUSAL_MEM" \
TOKEN_DELTA_DTYPE="$COMMON_TOKEN_DELTA_DTYPE" \
LAYER=26 \
LATENT_MULT=2 \
TOPK=4 \
TAG="$TAG" \
OUTPUT_DIR="${COMMON_OUTPUT_ROOT}/${TAG}" \
BOOTSTRAP_DIR="${COMMON_OUTPUT_ROOT}/${TAG}/bootstrap" \
bash scripts/submit_qwen3_8b_r42_token_eval_bundle_anvil.sh

cat <<'EOM'
R4.2 native active-control audit submitted.
Expected outputs:
- token_delta_sae_causal_summary.csv
- token_delta_sae_causal_best_rows.csv
- token_delta_sae_causal_selected_sets.csv
- TOKEN_DELTA_SAE_CAUSAL_REPORT.md
- bootstrap/token_delta_sae_bootstrap_summary.csv
- bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md
EOM
