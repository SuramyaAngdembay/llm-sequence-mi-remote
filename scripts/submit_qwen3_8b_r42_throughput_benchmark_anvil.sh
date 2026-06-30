#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
BUILD_CONDA_ENV="${BUILD_CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
INPUT_DIR="${INPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_r42_throughput_benchmark}"

MICRO_BATCHES="${MICRO_BATCHES:-12,13,14}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-4096}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
BENCH_STEPS="${BENCH_STEPS:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-longest}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GC_MODE="${GC_MODE:-on}"
DL_WORKERS="${DL_WORKERS:-4}"
TARGET_GIB="${TARGET_GIB:-70}"
TARGET_TOLERANCE_GIB="${TARGET_TOLERANCE_GIB:-2}"

BUILD_JOB=""
DEPENDENCY_ARGS=()
if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/build_summary.json" ]]; then
  BUILD_JOB=$(
    REPO_DIR="$REPO_DIR" \
      CONDA_ENV="$BUILD_CONDA_ENV" \
      INPUT_DIR="$INPUT_DIR" \
      OUTPUT_DIR="$DATA_DIR" \
      BUILDER=fast \
      sbatch --parsable --export=ALL slurm/build_jsonl.template.sbatch
  )
  DEPENDENCY_ARGS+=(--dependency=afterok:${BUILD_JOB})
fi

BENCH_JOB=$(
  REPO_DIR="$REPO_DIR" \
    CONDA_ENV="$CONDA_ENV" \
    CONFIG="$CONFIG" \
    DATA_DIR="$DATA_DIR" \
    OUT_DIR="$OUT_DIR" \
    MICRO_BATCHES="$MICRO_BATCHES" \
    MAX_TRAIN_EXAMPLES="$MAX_TRAIN_EXAMPLES" \
    WARMUP_STEPS="$WARMUP_STEPS" \
    BENCH_STEPS="$BENCH_STEPS" \
    SAMPLE_MODE="$SAMPLE_MODE" \
    ATTN_IMPL="$ATTN_IMPL" \
    GC_MODE="$GC_MODE" \
    DL_WORKERS="$DL_WORKERS" \
    TARGET_GIB="$TARGET_GIB" \
    TARGET_TOLERANCE_GIB="$TARGET_TOLERANCE_GIB" \
    sbatch --parsable \
      "${DEPENDENCY_ARGS[@]}" \
      --export=ALL \
      slurm/benchmark_qwen3_8b_throughput.template.sbatch
)

cat <<EOM
submitted_qwen3_8b_r42_throughput_benchmark=1
build_job=${BUILD_JOB}
benchmark_job=${BENCH_JOB}
input_dir=${INPUT_DIR}
data_dir=${DATA_DIR}
out_dir=${OUT_DIR}
micro_batches=${MICRO_BATCHES}
sample_mode=${SAMPLE_MODE}
target_gib=${TARGET_GIB}
target_tolerance_gib=${TARGET_TOLERANCE_GIB}
gc_mode=${GC_MODE}
EOM
