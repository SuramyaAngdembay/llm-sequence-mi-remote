#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/transfer_package}"
DEST_HOST="${ANVIL_HOST:-x-sangdembay@anvil.rcac.purdue.edu}"
DEST_DIR="${ANVIL_DEST:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/}"

echo "Source: ${SRC_DIR}"
echo "Destination: ${DEST_HOST}:${DEST_DIR}"

rsync -avP --partial \
  "${SRC_DIR%/}/" \
  "${DEST_HOST}:${DEST_DIR}"

echo
echo "Transfer finished. Next on Anvil:"
echo "  cd ${DEST_DIR}"
echo "  sha256sum -c sha256sums.txt"

