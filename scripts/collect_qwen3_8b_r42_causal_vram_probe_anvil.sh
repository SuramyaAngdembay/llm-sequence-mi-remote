#!/bin/bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_r42_causal_vram_probe}"
JOBS_FILE="${JOBS_FILE:-${OUT_ROOT}/submitted_jobs.env}"

if [[ -f "$JOBS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$JOBS_FILE"
fi

JOB_IDS="${JOB_IDS:-${AI_JOB:-},${GPU_JOB:-}}"
JOB_IDS="${JOB_IDS#,}"
JOB_IDS="${JOB_IDS%,}"
if [[ -z "$JOB_IDS" ]]; then
  echo "No JOB_IDS found. Set JOB_IDS=job1,job2 or provide ${JOBS_FILE}." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
sacct -j "$JOB_IDS" \
  --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AveRSS,AllocTRES,ExitCode \
  -P > "${OUT_ROOT}/slurm_sacct.csv"

cat "${OUT_ROOT}/slurm_sacct.csv"
