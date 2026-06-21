#!/bin/bash
#SBATCH --job-name=session_jsonl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=cis230270
#SBATCH --partition=shared

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
mkdir -p logs

module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora

INPUT_DIR="${INPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"

python scripts/build_session_jsonl.py \
  --input-dir "$INPUT_DIR" \
  --out-dir "$OUTPUT_DIR"
