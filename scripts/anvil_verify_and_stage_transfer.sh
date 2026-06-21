#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data}"
PROJECT_ROOT="$(cd "${DATA_DIR}/.." && pwd)"

cd "${DATA_DIR}"

echo "[1/4] Verifying checksums in ${DATA_DIR}"
sha256sum -c sha256sums.txt

echo "[2/4] Creating runtime directories"
mkdir -p \
  "${PROJECT_ROOT}/logs" \
  "${PROJECT_ROOT}/outputs" \
  "${PROJECT_ROOT}/hf_cache" \
  "${PROJECT_ROOT}/token_cache" \
  "${PROJECT_ROOT}/delta_cache" \
  "${PROJECT_ROOT}/checkpoints"

echo "[3/4] Writing transfer receipt"
python - <<'PY'
import datetime as dt
import json
import os
import socket
from pathlib import Path

data_dir = Path(os.getcwd())
manifest = json.loads((data_dir / "manifest.json").read_text())

receipt = {
    "verified_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    "host": socket.gethostname(),
    "data_dir": str(data_dir),
    "num_shards": manifest["num_shards"],
    "rows_per_shard": manifest["rows_per_shard"],
    "labels": manifest["labels"]["path"],
    "shards": [item["path"] for item in manifest["shards"]],
    "status": "TRANSFER_VERIFIED",
}

(data_dir / "transfer_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY

echo "[4/4] Done"
echo "Next state: READY_FOR_JSONL"
echo "Receipt: ${DATA_DIR}/transfer_receipt.json"

