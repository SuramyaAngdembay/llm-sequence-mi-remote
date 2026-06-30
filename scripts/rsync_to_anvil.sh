#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/transfer_package}"
DEST_HOST="${ANVIL_HOST:-x-sangdembay@anvil.rcac.purdue.edu}"

if [[ -n "${ANVIL_DEST:-}" ]]; then
  DEST_DIR="${ANVIL_DEST}"
elif [[ "${SRC_DIR}" == *"transfer_package_r42"* ]]; then
  DEST_DIR="/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/"
else
  DEST_DIR="/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/"
fi

echo "Source: ${SRC_DIR}"
echo "Destination: ${DEST_HOST}:${DEST_DIR}"

rsync -avP --partial --checksum \
  "${SRC_DIR%/}/" \
  "${DEST_HOST}:${DEST_DIR}"

echo
echo "Transfer finished. Next on Anvil:"
echo "  cd ${DEST_DIR}"
echo "  sha256sum -c sha256sums.txt"
