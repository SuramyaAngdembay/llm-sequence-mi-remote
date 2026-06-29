#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh}"
COMMON_TOP_SETS="${COMMON_TOP_SETS:-top1,top3,top5}"
COMMON_CONTROL_SET="${COMMON_CONTROL_SET:-control3}"
COMMON_CONTEXT_MODES="${COMMON_CONTEXT_MODES:-team,role,project_role,dept_role}"
COMMON_N_BOOTSTRAP="${COMMON_N_BOOTSTRAP:-4000}"

run_bundle() {
  local layer="$1"
  local latent_mult="$2"
  local topk="$3"
  local tag="l${layer}_m$(printf '%02d' "$latent_mult")_k$(printf '%02d' "$topk")"

  TOP_SETS="$COMMON_TOP_SETS" \
  CONTROL_SET="$COMMON_CONTROL_SET" \
  CONTEXT_MODES="$COMMON_CONTEXT_MODES" \
  N_BOOTSTRAP="$COMMON_N_BOOTSTRAP" \
  LAYER="$layer" \
  LATENT_MULT="$latent_mult" \
  TOPK="$topk" \
  TAG="$tag" \
  OUTPUT_DIR="${COMMON_OUTPUT_ROOT}/${tag}" \
  BOOTSTRAP_DIR="${COMMON_OUTPUT_ROOT}/${tag}/bootstrap" \
  bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
}

echo "Submitting recommended Qwen3-8B causal suite..."
echo

echo "[1/3] layer=18 latent_mult=4 k=4"
run_bundle 18 4 4
echo

echo "[2/3] layer=18 latent_mult=2 k=4"
run_bundle 18 2 4
echo

echo "[3/3] layer=18 latent_mult=4 k=8"
run_bundle 18 4 8
echo

cat <<'EOM'
Recommended suite submitted.
Expected outputs per config:
- token_delta_sae_causal_summary.csv
- token_delta_sae_causal_best_rows.csv
- token_delta_sae_causal_selected_sets.csv
- bootstrap/token_delta_sae_bootstrap_summary.csv
- bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md
EOM
