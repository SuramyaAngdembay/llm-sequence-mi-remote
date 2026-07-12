#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
JSONL_PATH="${JSONL_PATH:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42/all.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
SCORE_OUTPUT_DIR="${SCORE_OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r42_fullpop_scores}"
SCORES_PARQUET="${SCORES_PARQUET:-${SCORE_OUTPUT_DIR}/example_scores.parquet}"
OUT_DIR="${OUT_DIR:-results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned}"
RUN_NAME="${RUN_NAME:-qwen3_8b_r42_fold_aligned}"
SEED="${SEED:-42}"
BENIGN_TEST_USERS="${BENIGN_TEST_USERS:-800}"
FOLD_LIMIT="${FOLD_LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
FLUSH_ROWS="${FLUSH_ROWS:-4096}"
LOG_EVERY="${LOG_EVERY:-4096}"
USERS_FILE="${USERS_FILE:-}"
SKIP_SCORING="${SKIP_SCORING:-0}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_ACCOUNT="${GPU_ACCOUNT:-cis230270-gpu}"
GPU_MEM="${GPU_MEM:-180G}"
GPU_TIME="${GPU_TIME:-36:00:00}"
GPU_CPUS="${GPU_CPUS:-24}"
GPU_POLL_SEC="${GPU_POLL_SEC:-10}"
SEPARATE_BASE_MODEL="${SEPARATE_BASE_MODEL:-0}"

export REPO_DIR CONDA_ENV CONFIG JSONL_PATH ADAPTER_DIR OUTPUT_DIR="$SCORE_OUTPUT_DIR"
export BATCH_SIZE LOSS_BATCH_SIZE MAX_EXAMPLES FLUSH_ROWS LOG_EVERY USERS_FILE GPU_POLL_SEC SEPARATE_BASE_MODEL

if [[ "$SKIP_SCORING" == "1" || "$SKIP_SCORING" == "true" ]]; then
  SCORE_JOB=""
else
  SCORE_JOB=$(
    sbatch --parsable \
      --partition="$GPU_PARTITION" \
      --account="$GPU_ACCOUNT" \
      --mem="$GPU_MEM" \
      --time="$GPU_TIME" \
      --cpus-per-task="$GPU_CPUS" \
      --export=ALL \
      slurm/score_adapter_examples.template.sbatch
  )
fi

export REPO_DIR CONDA_ENV SCORES_PARQUET RUN_NAME SEED BENIGN_TEST_USERS FOLD_LIMIT OUT_DIR
if [[ -n "${SCORE_JOB}" ]]; then
  EVAL_JOB=$(
    sbatch --parsable --dependency=afterok:${SCORE_JOB} --export=ALL \
      slurm/eval_fold_aligned_detector_metrics_cpu.template.sbatch
  )
else
  EVAL_JOB=$(
    sbatch --parsable --export=ALL \
      slurm/eval_fold_aligned_detector_metrics_cpu.template.sbatch
  )
fi

cat <<EOM
submitted_qwen3_8b_r42_fold_aligned_detector=1
score_job=${SCORE_JOB:-skipped}
eval_job=${EVAL_JOB}
jsonl_path=${JSONL_PATH}
adapter_dir=${ADAPTER_DIR}
score_output_dir=${SCORE_OUTPUT_DIR}
scores_parquet=${SCORES_PARQUET}
out_dir=${OUT_DIR}
run_name=${RUN_NAME}
seed=${SEED}
benign_test_users=${BENIGN_TEST_USERS}
fold_limit=${FOLD_LIMIT}
batch_size=${BATCH_SIZE}
loss_batch_size=${LOSS_BATCH_SIZE}
gpu_partition=${GPU_PARTITION}
gpu_account=${GPU_ACCOUNT}
gpu_time=${GPU_TIME}
gpu_mem=${GPU_MEM}
users_file=${USERS_FILE}
EOM
