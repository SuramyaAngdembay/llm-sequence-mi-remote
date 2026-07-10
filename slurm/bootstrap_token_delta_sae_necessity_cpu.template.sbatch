#!/bin/bash
# CPU-only bootstrap over token necessity best-row CSVs.
#SBATCH -A cis230270
#SBATCH -p shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J tok_need_boot
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
BEST_ROWS_CSV="${BEST_ROWS_CSV:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3b/l18_m02_k08/token_delta_sae_necessity_best_rows.csv}"
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3b/l18_m02_k08/bootstrap}"
TOP_SETS="${TOP_SETS:-top5}"
CONTROL_SET="${CONTROL_SET:-control5_active}"
N_BOOTSTRAP="${N_BOOTSTRAP:-4000}"
SEED="${SEED:-42}"

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

cd "$REPO_DIR"
python scripts/bootstrap_token_delta_sae_necessity.py \
  --best-rows-csv "$BEST_ROWS_CSV" \
  --out-dir "$OUT_DIR" \
  --top-sets "$TOP_SETS" \
  --control-set "$CONTROL_SET" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED"
