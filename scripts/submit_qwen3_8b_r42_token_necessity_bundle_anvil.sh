#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on}"
FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on}"

LAYER="${LAYER:-26}"
LATENT_MULT="${LATENT_MULT:-2}"
TOPK="${TOPK:-4}"
TAG="${TAG:-l${LAYER}_m$(printf '%02d' "$LATENT_MULT")_k$(printf '%02d' "$TOPK")_top5_control5_active_necessity}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42/${TAG}}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${OUTPUT_DIR}/bootstrap}"

BATCH_SIZE="${BATCH_SIZE:-12}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-4}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-12}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
FULL_LOGITS_MAX_GIB="${FULL_LOGITS_MAX_GIB:-28}"
MAX_LOGIT_ELEMENTS="${MAX_LOGIT_ELEMENTS:-536870912}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,dept,dept_role}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
EXCLUDE_SAME_USER_MATCHES="${EXCLUDE_SAME_USER_MATCHES:-0}"
GPU_POLL_SEC="${GPU_POLL_SEC:-0}"
CAUSAL_PARTITION="${CAUSAL_PARTITION:-}"
CAUSAL_ACCOUNT="${CAUSAL_ACCOUNT:-}"
CAUSAL_MEM="${CAUSAL_MEM:-}"
CAUSAL_TIME="${CAUSAL_TIME:-}"
CAUSAL_CPUS="${CAUSAL_CPUS:-}"
N_BOOTSTRAP="${N_BOOTSTRAP:-4000}"
SEED="${SEED:-42}"

export REPO_DIR CONDA_ENV CONFIG DATA_DIR ADAPTER_DIR EXTRACT_DIR FRONTIER_DIR
export OUTPUT_DIR LAYER LATENT_MULT TOPK BATCH_SIZE LOSS_BATCH_SIZE SAE_BATCH_SIZE PATCH_CHUNK_SIZE TOKEN_DELTA_DTYPE
export FULL_LOGITS_MAX_GIB MAX_LOGIT_ELEMENTS
export CONTEXT_MODES TOP_SETS CONTROL_SET ACTIVE_CONTROL_MIN_FRAC ALPHAS MAX_PAIRS EXCLUDE_SAME_USER_MATCHES GPU_POLL_SEC

SBATCH_ARGS=(--parsable --export=ALL)
[[ -n "$CAUSAL_PARTITION" ]] && SBATCH_ARGS+=(--partition="$CAUSAL_PARTITION")
[[ -n "$CAUSAL_ACCOUNT" ]] && SBATCH_ARGS+=(--account="$CAUSAL_ACCOUNT")
[[ -n "$CAUSAL_MEM" ]] && SBATCH_ARGS+=(--mem="$CAUSAL_MEM")
[[ -n "$CAUSAL_TIME" ]] && SBATCH_ARGS+=(--time="$CAUSAL_TIME")
[[ -n "$CAUSAL_CPUS" ]] && SBATCH_ARGS+=(--cpus-per-task="$CAUSAL_CPUS")

CAUSAL_JOB=$(
  sbatch "${SBATCH_ARGS[@]}" \
    slurm/eval_token_delta_sae_necessity.template.sbatch
)

BEST_ROWS_CSV="${OUTPUT_DIR}/token_delta_sae_necessity_best_rows.csv"
OUT_DIR="$BOOTSTRAP_DIR"
export BEST_ROWS_CSV OUT_DIR N_BOOTSTRAP SEED

BOOTSTRAP_JOB=$(
  sbatch --parsable --dependency=afterok:${CAUSAL_JOB} \
    --export=ALL \
    slurm/bootstrap_token_delta_sae_necessity_cpu.template.sbatch
)

cat <<EOM
submitted_qwen3_8b_r42_token_necessity_bundle=1
causal_job=${CAUSAL_JOB}
bootstrap_job=${BOOTSTRAP_JOB}
conda_env=${CONDA_ENV}
tag=${TAG}
output_dir=${OUTPUT_DIR}
bootstrap_dir=${BOOTSTRAP_DIR}
layer=${LAYER}
latent_mult=${LATENT_MULT}
topk=${TOPK}
top_sets=${TOP_SETS}
control_set=${CONTROL_SET}
active_control_min_frac=${ACTIVE_CONTROL_MIN_FRAC}
batch_size=${BATCH_SIZE}
loss_batch_size=${LOSS_BATCH_SIZE}
patch_chunk_size=${PATCH_CHUNK_SIZE}
full_logits_max_gib=${FULL_LOGITS_MAX_GIB}
max_logit_elements=${MAX_LOGIT_ELEMENTS}
causal_partition=${CAUSAL_PARTITION}
causal_account=${CAUSAL_ACCOUNT}
EOM
