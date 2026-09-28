#!/bin/bash
# PCA reconstruction-error detector on CERT r4.2 day-level features (thesis), 2026-09-28. CPU only.
#SBATCH --job-name=pca_recon_r42
#SBATCH --partition=node
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/01/srangdembay/InsiderThreatDetection/r4.2/pca_reconstruction_2026_09_28/%x-%j.out
set -euo pipefail
D=/homes/01/srangdembay/InsiderThreatDetection/r4.2
W=$D/pca_reconstruction_2026_09_28
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $W
md5sum pca_reconstruction_r42.py $D/ExtractedData/dayr4.2.csv r42_user_splits.csv sessionr4.2_user_map.csv
/usr/bin/time -v ~/miniconda3/bin/python -u pca_reconstruction_r42.py --day-csv $D/ExtractedData/dayr4.2.csv \
  --user-map sessionr4.2_user_map.csv --splits r42_user_splits.csv --out-dir $W --pve 0.95
