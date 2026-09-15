#!/bin/bash
#SBATCH -p shared
#SBATCH -A tra250034
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J fold_eval
#SBATCH -o /anvil/scratch/x-bbhusal1/p1_8b/fold_eval_%j.out
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1_8b
for M in full no_psy no_profile shuffle_profile; do
  echo "=== fold-aligned scoring: $M ==="
  python scripts/eval_fold_aligned_detector_metrics.py --scores-parquet $P/deltas_$M/example_scores.parquet --run-name 8b_$M --benign-test-users 800 --seed 42 --out-dir $P/fold_$M
done
echo "=== rows CSV schema (for bootstrap column mapping) ==="
head -2 $P/fold_full/fold_aligned_detector_rows.csv
echo "=== cluster bootstrap ==="
python $P/fold_boot.py $P
