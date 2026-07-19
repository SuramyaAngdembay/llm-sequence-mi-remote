#!/bin/bash
# User-level cluster bootstrap over the four final same-user-excluded bundles.
# CPU-only; reads existing *_best_rows.csv, writes cluster_bootstrap/ subdirs.
#SBATCH -A cis230270
#SBATCH -p shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J cluster_boot_cpu
#SBATCH -o /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.out
#SBATCH -e /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
OUT="/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

module load conda/2026.03
CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora}"
conda activate "$CONDA_ENV"

cd "$REPO_DIR"

run () {
  local mode="$1" rows="$2" outdir="$3"
  python scripts/cluster_bootstrap_token_delta_sae.py \
    --best-rows-csv "$rows" \
    --out-dir "$outdir" \
    --mode "$mode" \
    --top-sets top5 \
    --control-set control5_active \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed 42
}

# r6.2 causal
run causal \
  "$OUT/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_no_same_user/l18_m04_k08_top5_control5_active_no_same_user/token_delta_sae_causal_best_rows.csv" \
  "$OUT/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_no_same_user/l18_m04_k08_top5_control5_active_no_same_user/cluster_bootstrap"

# r6.2 necessity
run necessity \
  "$OUT/token_delta_sae_necessity_qwen3_8b_no_same_user/l18_m04_k08_top5_control5_active_necessity_no_same_user/token_delta_sae_necessity_best_rows.csv" \
  "$OUT/token_delta_sae_necessity_qwen3_8b_no_same_user/l18_m04_k08_top5_control5_active_necessity_no_same_user/cluster_bootstrap"

# r4.2 native causal
run causal \
  "$OUT/token_delta_sae_causal_qwen3_8b_r42_native_active_control_no_same_user/l26_m02_k04_top5_control5_active_no_same_user/token_delta_sae_causal_best_rows.csv" \
  "$OUT/token_delta_sae_causal_qwen3_8b_r42_native_active_control_no_same_user/l26_m02_k04_top5_control5_active_no_same_user/cluster_bootstrap"

# r4.2 native necessity
run necessity \
  "$OUT/token_delta_sae_necessity_qwen3_8b_r42_no_same_user/l26_m02_k04_top5_control5_active_necessity_no_same_user/token_delta_sae_necessity_best_rows.csv" \
  "$OUT/token_delta_sae_necessity_qwen3_8b_r42_no_same_user/l26_m02_k04_top5_control5_active_necessity_no_same_user/cluster_bootstrap"

echo DONE
