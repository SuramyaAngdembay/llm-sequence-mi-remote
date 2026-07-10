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
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-0}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-12}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,dept,dept_role}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
N_BOOTSTRAP="${N_BOOTSTRAP:-4000}"
SEED="${SEED:-42}"

export REPO_DIR CONDA_ENV CONFIG DATA_DIR ADAPTER_DIR EXTRACT_DIR FRONTIER_DIR
export OUTPUT_DIR LAYER LATENT_MULT TOPK BATCH_SIZE LOSS_BATCH_SIZE SAE_BATCH_SIZE PATCH_CHUNK_SIZE TOKEN_DELTA_DTYPE
export CONTEXT_MODES TOP_SETS CONTROL_SET ACTIVE_CONTROL_MIN_FRAC ALPHAS MAX_PAIRS

CAUSAL_JOB=$(
  sbatch --parsable --export=ALL \
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
EOM
