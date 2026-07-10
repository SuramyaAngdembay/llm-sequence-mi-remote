#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

# Short A100 gpu-debug probe for the r4.2 Qwen3-8B token-necessity run.
# This uses the same audited native r4.2 feature branch as the full run but
# caps receiver pairs so it fits the gpu-debug queue walltime.
GPU_DEBUG_PARTITION="${GPU_DEBUG_PARTITION:-gpu-debug}"
GPU_ACCOUNT="${GPU_ACCOUNT:-cis230270-gpu}"
GPU_MEM="${GPU_MEM:-120G}"
GPU_TIME="${GPU_TIME:-00:30:00}"
GPU_CPUS="${GPU_CPUS:-16}"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
EXTRACT_DIR="${EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on}"
FRONTIER_DIR="${FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on}"

COMMON_OUTPUT_ROOT="${COMMON_OUTPUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_probe_qwen3_8b_r42_gpu_debug}"
LAYER="${LAYER:-26}"
LATENT_MULT="${LATENT_MULT:-2}"
TOPK="${TOPK:-4}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
ACTIVE_CONTROL_MIN_FRAC="${ACTIVE_CONTROL_MIN_FRAC:-0.002}"
CONTEXT_MODES="${CONTEXT_MODES:-team,role,dept,dept_role}"

BATCH_SIZE="${BATCH_SIZE:-12}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-0}"
PATCH_CHUNK_SIZE="${PATCH_CHUNK_SIZE:-12}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-2048}"
TOKEN_DELTA_DTYPE="${TOKEN_DELTA_DTYPE:-float32}"
ALPHAS="${ALPHAS:-0.25,0.5,0.75,1.0}"
MAX_PAIRS="${MAX_PAIRS:-16}"
GPU_POLL_SEC="${GPU_POLL_SEC:-2}"

TAG="${TAG:-l${LAYER}_m$(printf '%02d' "$LATENT_MULT")_k$(printf '%02d' "$TOPK")_top5_control5_active_probe_bs${BATCH_SIZE}_pairs${MAX_PAIRS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${COMMON_OUTPUT_ROOT}/${TAG}}"

mkdir -p "$COMMON_OUTPUT_ROOT" "$OUTPUT_DIR"

export REPO_DIR CONDA_ENV CONFIG DATA_DIR ADAPTER_DIR EXTRACT_DIR FRONTIER_DIR
export OUTPUT_DIR LAYER LATENT_MULT TOPK BATCH_SIZE LOSS_BATCH_SIZE SAE_BATCH_SIZE PATCH_CHUNK_SIZE TOKEN_DELTA_DTYPE
export CONTEXT_MODES TOP_SETS CONTROL_SET ACTIVE_CONTROL_MIN_FRAC ALPHAS MAX_PAIRS GPU_POLL_SEC

echo "Submitting r4.2 Qwen3-8B token-necessity gpu-debug probe..."
echo "partition=${GPU_DEBUG_PARTITION}"
echo "account=${GPU_ACCOUNT}"
echo "time=${GPU_TIME}"
echo "mem=${GPU_MEM}"
echo "tag=${TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "batch_size=${BATCH_SIZE}"
echo "loss_batch_size=${LOSS_BATCH_SIZE}"
echo "patch_chunk_size=${PATCH_CHUNK_SIZE}"
echo "max_pairs=${MAX_PAIRS}"
echo

PROBE_JOB=$(
  sbatch --parsable \
    --partition="$GPU_DEBUG_PARTITION" \
    --account="$GPU_ACCOUNT" \
    --mem="$GPU_MEM" \
    --time="$GPU_TIME" \
    --cpus-per-task="$GPU_CPUS" \
    --export=ALL \
    slurm/eval_token_delta_sae_necessity.template.sbatch
)

cat > "${COMMON_OUTPUT_ROOT}/submitted_${TAG}.env" <<EOM
COMMON_OUTPUT_ROOT=${COMMON_OUTPUT_ROOT}
TAG=${TAG}
PROBE_JOB=${PROBE_JOB}
GPU_DEBUG_PARTITION=${GPU_DEBUG_PARTITION}
GPU_ACCOUNT=${GPU_ACCOUNT}
GPU_TIME=${GPU_TIME}
GPU_MEM=${GPU_MEM}
GPU_CPUS=${GPU_CPUS}
OUTPUT_DIR=${OUTPUT_DIR}
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
GPU_POLL_SEC=${GPU_POLL_SEC}
EOM

cat <<EOM
submitted_qwen3_8b_r42_token_necessity_gpu_debug_probe=1
probe_job=${PROBE_JOB}
partition=${GPU_DEBUG_PARTITION}
account=${GPU_ACCOUNT}
output_dir=${OUTPUT_DIR}

After completion:
  sacct -j ${PROBE_JOB} --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AveRSS,ExitCode -P
  tail -n 5 ${OUTPUT_DIR}/gpu_poll_${PROBE_JOB}.csv
EOM
