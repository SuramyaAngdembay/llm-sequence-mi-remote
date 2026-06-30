#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-250000}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/transfer_package_r42}"

cd "${ROOT}"

python3 scripts/prepare_transfer_package.py \
  --dataset-tag r4.2 \
  --rows-per-shard "${ROWS_PER_SHARD}" \
  --out-dir "${OUT_DIR}" \
  --force

echo
echo "Prepared r4.2 transfer package:"
echo "  ${OUT_DIR}"
echo
echo "Next:"
echo "  bash scripts/rsync_to_anvil.sh ${OUT_DIR}"
echo
echo "Default Anvil target for this package:"
echo "  /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/"
