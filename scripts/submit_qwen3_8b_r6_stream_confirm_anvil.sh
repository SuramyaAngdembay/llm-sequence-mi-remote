#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

export CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
export DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
export ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter}"
export EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2}"
export FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream}"

export LAYER="${LAYER:-18}"
export LATENT_MULT="${LATENT_MULT:-4}"
export TOPK="${TOPK:-8}"
export TAG="${TAG:-l18_m04_k08_stream_confirm_v1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_stream_confirm/${TAG}}"
export BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${OUTPUT_DIR}/bootstrap}"

export CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}"
export TOP_SETS="${TOP_SETS:-top1,top3,top5}"
export CONTROL_SET="${CONTROL_SET:-control3}"
export ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
export ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
export MAX_RECEIVERS="${MAX_RECEIVERS:-0}"
export MAX_CANDIDATE_DONORS="${MAX_CANDIDATE_DONORS:-16}"
export CAUSAL_MEM="${CAUSAL_MEM:-256G}"
export PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-0}"
export TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"

bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
