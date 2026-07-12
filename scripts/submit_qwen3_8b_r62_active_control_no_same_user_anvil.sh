#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_no_same_user}"
TAG="${TAG:-l18_m04_k08_top5_control5_active_no_same_user}"

TOP_SETS="${TOP_SETS:-top5}" CONTROL_SET="${CONTROL_SET:-control5_active}" ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}" CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}" N_BOOTSTRAP="${N_BOOTSTRAP:-4000}" LAYER=18 LATENT_MULT=4 TOPK=8 EXCLUDE_SAME_USER_DONORS=1 TAG="$TAG" OUTPUT_DIR="${COMMON_OUTPUT_ROOT}/${TAG}" BOOTSTRAP_DIR="${COMMON_OUTPUT_ROOT}/${TAG}/bootstrap"   bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
