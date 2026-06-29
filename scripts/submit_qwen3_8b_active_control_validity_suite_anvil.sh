#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_active_control_v1}"
COMMON_TOP_SETS="${COMMON_TOP_SETS:-top5}"
COMMON_CONTROL_SET="${COMMON_CONTROL_SET:-control5_active}"
COMMON_ACTIVE_CONTROL_MIN_FRAC="${COMMON_ACTIVE_CONTROL_MIN_FRAC:-0.002}"
COMMON_CONTEXT_MODES="${COMMON_CONTEXT_MODES:-team,role,project_role,dept_role}"
COMMON_N_BOOTSTRAP="${COMMON_N_BOOTSTRAP:-4000}"

run_bundle() {
  local layer="$1"
  local latent_mult="$2"
  local topk="$3"
  local tag="l${layer}_m$(printf '%02d' "$latent_mult")_k$(printf '%02d' "$topk")_top5_control5_active"

  TOP_SETS="$COMMON_TOP_SETS" \
  CONTROL_SET="$COMMON_CONTROL_SET" \
  ACTIVE_CONTROL_MIN_FRAC="$COMMON_ACTIVE_CONTROL_MIN_FRAC" \
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

echo "Submitting Qwen3-8B active-control validity suite..."
echo "top_sets=${COMMON_TOP_SETS}"
echo "control_set=${COMMON_CONTROL_SET}"
echo "active_control_min_frac=${COMMON_ACTIVE_CONTROL_MIN_FRAC}"
echo

echo "[1/2] layer=18 latent_mult=4 k=4"
run_bundle 18 4 4
echo

echo "[2/2] layer=18 latent_mult=4 k=8"
run_bundle 18 4 8
echo

cat <<'EOM'
Active-control validity suite submitted.
Expected outputs per config:
- token_delta_sae_causal_summary.csv
- token_delta_sae_causal_best_rows.csv
- token_delta_sae_causal_selected_sets.csv
- bootstrap/token_delta_sae_bootstrap_summary.csv
- bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md
EOM
