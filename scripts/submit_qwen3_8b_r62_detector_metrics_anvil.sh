#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
SCORES_PARQUET="${SCORES_PARQUET:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2/example_scores.parquet}"
RUN_NAME="${RUN_NAME:-qwen3_8b_r62}"
SPLIT="${SPLIT:-eval}"
OUT_DIR="${OUT_DIR:-results/qwen3_8b_token_causal/detector_metrics}"

export REPO_DIR CONDA_ENV SCORES_PARQUET RUN_NAME SPLIT OUT_DIR

JOB_ID=$(
  sbatch --parsable --export=ALL \
    slurm/eval_example_scores_detector_metrics_cpu.template.sbatch
)

cat <<EOM
submitted_qwen3_8b_r62_detector_metrics=1
job_id=${JOB_ID}
run_name=${RUN_NAME}
scores_parquet=${SCORES_PARQUET}
out_dir=${OUT_DIR}
EOM
