#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

echo "Submitting optional Qwen2.5-3B appendix suite on the known positive r6.2 config..."
echo "[1/2] active-control audit: layer=18 latent_mult=2 k=8"
bash scripts/submit_qwen3b_active_control_anvil.sh
echo
echo "[2/2] necessity audit: layer=18 latent_mult=2 k=8"
bash scripts/submit_qwen3b_token_necessity_bundle_anvil.sh
echo

cat <<'EOM'
Optional Qwen2.5-3B appendix suite submitted.
This is not a main-paper blocker.

Expected outputs:
- results/qwen3b_pilot/detector_metrics/
- active-control outputs under outputs/token_delta_sae_causal_qwen3b_active_control_v1/
- necessity outputs under outputs/token_delta_sae_necessity_qwen3b/
EOM
