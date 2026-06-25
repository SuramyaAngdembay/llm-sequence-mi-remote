#!/bin/bash
#SBATCH -A cis230270-ai
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J tok_boot
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
BEST_ROWS_CSV="${BEST_ROWS_CSV:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/token_delta_sae_causal_best_rows.csv}"
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/bootstrap}"
TOP_SETS="${TOP_SETS:-top1,top3,top5}"
CONTROL_SET="${CONTROL_SET:-control3}"
N_BOOTSTRAP="${N_BOOTSTRAP:-4000}"
SEED="${SEED:-42}"

module load conda/2026.03
source activate /anvil/projects/x-cis230270/x-sangdembay/.conda/envs/cert-qlora

cd "$REPO_DIR"
python scripts/bootstrap_token_delta_sae_causal.py \
  --best-rows-csv "$BEST_ROWS_CSV" \
  --out-dir "$OUT_DIR" \
  --top-sets "$TOP_SETS" \
  --control-set "$CONTROL_SET" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED"
