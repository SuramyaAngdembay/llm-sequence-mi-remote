#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on}"
COMMON_CONTEXT_MODES="${COMMON_CONTEXT_MODES:-team,role,project_role,dept_role}"
COMMON_TOP_SETS="${COMMON_TOP_SETS:-top1,top3,top5}"
COMMON_CONTROL_SET="${COMMON_CONTROL_SET:-control3}"
COMMON_ACTIVE_CONTROL_MIN_FRAC="${COMMON_ACTIVE_CONTROL_MIN_FRAC:-0.002}"
COMMON_N_BOOTSTRAP="${COMMON_N_BOOTSTRAP:-4000}"

run_bundle() {
  local layer="$1"
  local latent_mult="$2"
  local topk="$3"
  local tag="l${layer}_m$(printf '%02d' "$latent_mult")_k$(printf '%02d' "$topk")"

  CONTEXT_MODES="$COMMON_CONTEXT_MODES" \
  TOP_SETS="$COMMON_TOP_SETS" \
  CONTROL_SET="$COMMON_CONTROL_SET" \
  ACTIVE_CONTROL_MIN_FRAC="$COMMON_ACTIVE_CONTROL_MIN_FRAC" \
  N_BOOTSTRAP="$COMMON_N_BOOTSTRAP" \
  LAYER="$layer" \
  LATENT_MULT="$latent_mult" \
  TOPK="$topk" \
  TAG="$tag" \
  OUTPUT_DIR="${COMMON_OUTPUT_ROOT}/${tag}" \
  BOOTSTRAP_DIR="${COMMON_OUTPUT_ROOT}/${tag}/bootstrap" \
  bash scripts/submit_qwen3_8b_r42_token_eval_bundle_anvil.sh
}

echo "Submitting Qwen3-8B r4.2 recommended causal suite..."
echo "context_modes=${COMMON_CONTEXT_MODES}"
echo "top_sets=${COMMON_TOP_SETS}"
echo "control_set=${COMMON_CONTROL_SET}"
echo "active_control_min_frac=${COMMON_ACTIVE_CONTROL_MIN_FRAC}"
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
Qwen3-8B r4.2 recommended causal suite submitted.
Expected outputs per config:
- token_delta_sae_causal_summary.csv
- token_delta_sae_causal_best_rows.csv
- token_delta_sae_causal_selected_sets.csv
- bootstrap/token_delta_sae_bootstrap_summary.csv
- bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md
EOM
