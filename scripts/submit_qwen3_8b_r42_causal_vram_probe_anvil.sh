#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
OUT_ROOT="${OUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_r42_causal_vram_probe}"

LAYER="${LAYER:-26}"
MAX_EXAMPLES="${MAX_EXAMPLES:-4096}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
BENCH_STEPS="${BENCH_STEPS:-2}"
SAMPLE_MODE="${SAMPLE_MODE:-longest}"
ATTN_IMPL="${ATTN_IMPL:-auto}"
GPU_POLL_SEC="${GPU_POLL_SEC:-2}"

AI_BATCH_SIZES="${AI_BATCH_SIZES:-8,12,16,20,24,28,32}"
GPU_BATCH_SIZES="${GPU_BATCH_SIZES:-4,8,12,16,20,24}"
AI_MEM="${AI_MEM:-180G}"
GPU_MEM="${GPU_MEM:-180G}"
AI_TIME="${AI_TIME:-02:30:00}"
GPU_TIME="${GPU_TIME:-02:30:00}"
AI_TARGET_GIB="${AI_TARGET_GIB:-70}"
GPU_TARGET_GIB="${GPU_TARGET_GIB:-0}"
TARGET_FRACTION="${TARGET_FRACTION:-0.87}"
TARGET_TOLERANCE_GIB="${TARGET_TOLERANCE_GIB:-2}"

mkdir -p "$OUT_ROOT"

submit_probe() {
  local label="$1"
  local partition="$2"
  local account="$3"
  local mem="$4"
  local walltime="$5"
  local batch_sizes="$6"
  local target_gib="$7"
  local out_dir="${OUT_ROOT}/${label}"

  mkdir -p "$out_dir"
  REPO_DIR="$REPO_DIR" \
    CONDA_ENV="$CONDA_ENV" \
    CONFIG="$CONFIG" \
    DATA_DIR="$DATA_DIR" \
    ADAPTER_DIR="$ADAPTER_DIR" \
    OUT_DIR="$out_dir" \
    BATCH_SIZES="$batch_sizes" \
    LAYER="$LAYER" \
    MAX_EXAMPLES="$MAX_EXAMPLES" \
    WARMUP_STEPS="$WARMUP_STEPS" \
    BENCH_STEPS="$BENCH_STEPS" \
    SAMPLE_MODE="$SAMPLE_MODE" \
    ATTN_IMPL="$ATTN_IMPL" \
    TARGET_GIB="$target_gib" \
    TARGET_FRACTION="$TARGET_FRACTION" \
    TARGET_TOLERANCE_GIB="$TARGET_TOLERANCE_GIB" \
    GPU_POLL_SEC="$GPU_POLL_SEC" \
    sbatch --parsable \
      --partition="$partition" \
      --account="$account" \
      --mem="$mem" \
      --time="$walltime" \
      --export=ALL \
      slurm/benchmark_token_causal_vram.template.sbatch
}

AI_JOB="$(submit_probe ai ai cis230270-ai "$AI_MEM" "$AI_TIME" "$AI_BATCH_SIZES" "$AI_TARGET_GIB")"
GPU_JOB="$(submit_probe gpu gpu cis230270-gpu "$GPU_MEM" "$GPU_TIME" "$GPU_BATCH_SIZES" "$GPU_TARGET_GIB")"

cat > "${OUT_ROOT}/submitted_jobs.env" <<EOM
OUT_ROOT=${OUT_ROOT}
AI_JOB=${AI_JOB}
GPU_JOB=${GPU_JOB}
JOB_IDS=${AI_JOB},${GPU_JOB}
AI_BATCH_SIZES=${AI_BATCH_SIZES}
GPU_BATCH_SIZES=${GPU_BATCH_SIZES}
LAYER=${LAYER}
SAMPLE_MODE=${SAMPLE_MODE}
ATTN_IMPL=${ATTN_IMPL}
EOM

cat <<EOM
submitted_qwen3_8b_r42_causal_vram_probe=1
out_root=${OUT_ROOT}
ai_job=${AI_JOB}
gpu_job=${GPU_JOB}
ai_batch_sizes=${AI_BATCH_SIZES}
gpu_batch_sizes=${GPU_BATCH_SIZES}
layer=${LAYER}
sample_mode=${SAMPLE_MODE}
attn_impl=${ATTN_IMPL}

After completion, collect Slurm RSS/state with:
  sacct -j ${AI_JOB},${GPU_JOB} --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AveRSS,ExitCode -P
EOM
