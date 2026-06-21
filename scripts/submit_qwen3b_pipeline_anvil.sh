#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

JSONL_JOB=$(sbatch --parsable slurm/build_jsonl.template.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${JSONL_JOB} slurm/train_qlora.template.sbatch)
EXTRACT_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm/extract_deltas.template.sbatch)
SAE_JOB=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} slurm/train_delta_sae.template.sbatch)

cat <<EOF
submitted_pipeline=1
jsonl_job=${JSONL_JOB}
train_job=${TRAIN_JOB}
extract_job=${EXTRACT_JOB}
sae_job=${SAE_JOB}
EOF
