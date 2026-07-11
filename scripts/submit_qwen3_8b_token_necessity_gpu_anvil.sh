#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_ACCOUNT="${GPU_ACCOUNT:-cis230270-gpu}"
GPU_CAUSAL_MEM="${GPU_CAUSAL_MEM:-240G}"
GPU_CAUSAL_TIME="${GPU_CAUSAL_TIME:-24:00:00}"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter}"
EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2}"
FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream}"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_gpu}"
LAYER="${LAYER:-18}"
LATENT_MULT="${LATENT_MULT:-4}"
TOPK="${TOPK:-8}"
TAG="${TAG:-l${LAYER}_m$(printf '%02d' "$LATENT_MULT")_k$(printf '%02d' "$TOPK")_top5_control5_active_necessity_gpu_bs24_loss4}"
OUTPUT_DIR="${OUTPUT_DIR:-${COMMON_OUTPUT_ROOT}/${TAG}}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${OUTPUT_DIR}/bootstrap}"

TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,project_role,dept_role}"
N_BOOTSTRAP="${N_BOOTSTRAP:-4000}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-24}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-4}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-24}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
GPU_POLL_SEC="${GPU_POLL_SEC:-10}"

mkdir -p "$COMMON_OUTPUT_ROOT" "$BOOTSTRAP_DIR"

export REPO_DIR CONDA_ENV CONFIG DATA_DIR ADAPTER_DIR EXTRACT_DIR FRONTIER_DIR
export OUTPUT_DIR LAYER LATENT_MULT TOPK BATCH_SIZE LOSS_BATCH_SIZE SAE_BATCH_SIZE PATCH_CHUNK_SIZE TOKEN_DELTA_DTYPE
export CONTEXT_MODES TOP_SETS CONTROL_SET ACTIVE_CONTROL_MIN_FRAC ALPHAS MAX_PAIRS GPU_POLL_SEC

echo "Submitting Qwen3-8B r6.2 token necessity on A100..."
echo "partition=${GPU_PARTITION}"
echo "account=${GPU_ACCOUNT}"
echo "time=${GPU_CAUSAL_TIME}"
echo "mem=${GPU_CAUSAL_MEM}"
echo "tag=${TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "layer=${LAYER}"
echo "latent_mult=${LATENT_MULT}"
echo "topk=${TOPK}"
echo "top_sets=${TOP_SETS}"
echo "control_set=${CONTROL_SET}"
echo "active_control_min_frac=${ACTIVE_CONTROL_MIN_FRAC}"
echo "batch_size=${BATCH_SIZE}"
echo "loss_batch_size=${LOSS_BATCH_SIZE}"
echo "patch_chunk_size=${PATCH_CHUNK_SIZE}"
echo

CAUSAL_JOB=$(
  sbatch --parsable \
    --partition="$GPU_PARTITION" \
    --account="$GPU_ACCOUNT" \
    --mem="$GPU_CAUSAL_MEM" \
    --time="$GPU_CAUSAL_TIME" \
    --export=ALL \
    slurm/eval_token_delta_sae_necessity.template.sbatch
)

BEST_ROWS_CSV="${OUTPUT_DIR}/token_delta_sae_necessity_best_rows.csv"
OUT_DIR="$BOOTSTRAP_DIR"
export BEST_ROWS_CSV OUT_DIR N_BOOTSTRAP SEED

BOOTSTRAP_JOB=$(
  sbatch --parsable \
    --dependency=afterok:${CAUSAL_JOB} \
    --export=ALL \
    slurm/bootstrap_token_delta_sae_necessity_cpu.template.sbatch
)

cat > "${COMMON_OUTPUT_ROOT}/submitted_${TAG}.env" <<EOM
COMMON_OUTPUT_ROOT=${COMMON_OUTPUT_ROOT}
TAG=${TAG}
CAUSAL_JOB=${CAUSAL_JOB}
BOOTSTRAP_JOB=${BOOTSTRAP_JOB}
GPU_PARTITION=${GPU_PARTITION}
GPU_ACCOUNT=${GPU_ACCOUNT}
GPU_CAUSAL_TIME=${GPU_CAUSAL_TIME}
GPU_CAUSAL_MEM=${GPU_CAUSAL_MEM}
OUTPUT_DIR=${OUTPUT_DIR}
BOOTSTRAP_DIR=${BOOTSTRAP_DIR}
LAYER=${LAYER}
LATENT_MULT=${LATENT_MULT}
TOPK=${TOPK}
TOP_SETS=${TOP_SETS}
CONTROL_SET=${CONTROL_SET}
ACTIVE_CONTROL_MIN_FRAC=${ACTIVE_CONTROL_MIN_FRAC}
BATCH_SIZE=${BATCH_SIZE}
LOSS_BATCH_SIZE=${LOSS_BATCH_SIZE}
PATCH_CHUNK_SIZE=${PATCH_CHUNK_SIZE}
TOKEN_DELTA_DTYPE=${TOKEN_DELTA_DTYPE}
MAX_PAIRS=${MAX_PAIRS}
N_BOOTSTRAP=${N_BOOTSTRAP}
SEED=${SEED}
EOM

cat <<EOM
submitted_qwen3_8b_token_necessity_gpu=1
causal_job=${CAUSAL_JOB}
bootstrap_job=${BOOTSTRAP_JOB}
partition=${GPU_PARTITION}
account=${GPU_ACCOUNT}
causal_mem=${GPU_CAUSAL_MEM}
causal_time=${GPU_CAUSAL_TIME}
tag=${TAG}
output_dir=${OUTPUT_DIR}
bootstrap_dir=${BOOTSTRAP_DIR}
batch_size=${BATCH_SIZE}
loss_batch_size=${LOSS_BATCH_SIZE}
patch_chunk_size=${PATCH_CHUNK_SIZE}
EOM
