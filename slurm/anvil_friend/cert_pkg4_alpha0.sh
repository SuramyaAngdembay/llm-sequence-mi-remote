#!/bin/bash
# CERT package-4 reconstruction-only control (alpha 0), collaborator account.
# Design: docs/PREREGISTRATION_CERT_PACKAGE4.md, addendum item 8 implementation
# (recorded 2026-09-24 before the full run's rows were read).
#
# Identical to the full run 20879549 (scripts/submit_r42_token_class_causal_collab_anvil.sh
# "full") except: alpha 0.0, one candidate donor per donor type, context `team`.
# The full run's code directory is COPIED, never re-synced: that job is still
# running from it.
set -euo pipefail
S=/anvil/scratch/x-sangdembay/pkg4_share
SRC_REPO=/anvil/scratch/x-bbhusal1/pkg4_out/repo
W=/anvil/scratch/x-bbhusal1/pkg4_alpha0
EXPECT_MD5=6fe1d8650368716c0343de0f5664001b   # eval_token_delta_sae_causal.py of the full run

mkdir -p "$W"
[ -d "$W/repo" ] || cp -a "$SRC_REPO" "$W/repo"
got=$(md5sum "$W/repo/scripts/eval_token_delta_sae_causal.py" | cut -d' ' -f1)
[ "$got" = "$EXPECT_MD5" ] || { echo "code differs from the full run ($got)"; exit 2; }

export REPO_DIR=$W/repo
export CONDA_ENV=/anvil/scratch/x-bbhusal1/conda_envs/cert-qlora
export HF_HOME=/anvil/scratch/x-bbhusal1/hf_cache
export HF_HUB_CACHE=/anvil/scratch/x-bbhusal1/hf_cache
export CONFIG=$REPO_DIR/configs/qwen3_8b_qlora_session_targeted.yaml
export DATA_DIR=$S/session_jsonl_r42
export ADAPTER_DIR=$S/adapter
export EXTRACT_DIR=$S/token_deltas
export FRONTIER_DIR=$S/frontier
export RECEIVER_USER_FILE=$S/user_splits_r42/confirmation_users.txt
export LAYER=26 LATENT_MULT=2 TOPK=4
export BATCH_SIZE=12 LOSS_BATCH_SIZE=4 PATCH_CHUNK_SIZE=8 SAE_BATCH_SIZE=2048
export TOKEN_DELTA_DTYPE=float32
export TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002
export EXCLUDE_SAME_USER_DONORS=1
export GPU_POLL_SEC=5
export TOKEN_CLASS_SCHEMA=cert
export MAX_RECEIVERS=0
# the only differences from the full run
export ALPHAS=0.0 MAX_CANDIDATE_DONORS=1 CONTEXT_MODES=team
export OUTPUT_DIR=$W/confirmation_alpha0/l26_m02_k04_top5_control5_active
mkdir -p "$OUTPUT_DIR"

cd "$REPO_DIR"
mkdir -p logs
sbatch -p gpu-debug -A cis260991-gpu -t 00:30:00 --mem=120G --export=ALL \
  slurm/eval_token_delta_sae_causal.template.sbatch
