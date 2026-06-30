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
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

INPUT_DIR="${INPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
BUILDER="${BUILDER:-slow}"
VAL_FRAC="${VAL_FRAC:-0.10}"
MAX_SESSIONS="${MAX_SESSIONS:-24}"
USER_MAP="${USER_MAP:-}"
LABELS="${LABELS:-}"

EXTRA_ARGS=()
if [[ -n "$USER_MAP" ]]; then
  EXTRA_ARGS+=(--user-map "$USER_MAP")
fi
if [[ -n "$LABELS" ]]; then
  EXTRA_ARGS+=(--labels "$LABELS")
fi

case "$BUILDER" in
  fast)
    BUILDER_SCRIPT="scripts/build_session_jsonl_fast.py"
    ;;
  slow)
    BUILDER_SCRIPT="scripts/build_session_jsonl.py"
    ;;
  *)
    echo "Invalid BUILDER=$BUILDER (expected fast|slow)" >&2
    exit 2
    ;;
esac

echo "job=$SLURM_JOB_ID node=$(hostname) builder=$BUILDER input_dir=$INPUT_DIR output_dir=$OUTPUT_DIR"
python "$BUILDER_SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --val-frac "$VAL_FRAC" \
  --max-sessions "$MAX_SESSIONS" \
  "${EXTRA_ARGS[@]}"
