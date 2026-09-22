#!/bin/bash
# Work package 4, CERT r4.2, run from the COLLABORATOR account (x-bbhusal1,
# allocation cis260991-gpu) because cis230270-gpu is exhausted (2.2 SU left).
#
# Everything the run needs is staged by x-sangdembay under a world-readable
# scratch directory; nothing is copied into this account's private space. The
# staged adapter is fingerprint-verified identical to the published one
# (results/provenance/adapter_fingerprints/). All scientific parameters are the
# pre-registered ones in docs/PREREGISTRATION_CERT_PACKAGE4.md, identical to
# scripts/submit_r42_token_class_causal_anvil.sh. Only account, paths, conda env
# and HF cache differ.
#
# Environment difference to record: this account's env is transformers 5.16 /
# torch 2.5.1 / peft 0.20, against 4.53 / 2.7 / 0.14 for the smoke on the owner
# account. Base and patched scores are computed within one run, so the CONTRAST
# is unaffected; absolute NLLs may differ from the published run.
#
# Usage:
#   bash scripts/submit_r42_token_class_causal_collab_anvil.sh smoke
#   bash scripts/submit_r42_token_class_causal_collab_anvil.sh full
set -euo pipefail
MODE="${1:-smoke}"
S=/anvil/scratch/x-sangdembay/pkg4_share          # staged, world-readable
OUTROOT=/anvil/scratch/x-bbhusal1/pkg4_out        # this account's own space

export REPO_DIR=$S/repo
export CONDA_ENV=/anvil/scratch/x-bbhusal1/conda_envs/cert-qlora
export HF_HOME=/anvil/scratch/x-bbhusal1/hf_cache
export CONFIG=$S/repo/configs/qwen3_8b_qlora_session_targeted.yaml
export DATA_DIR=$S/session_jsonl_r42
export ADAPTER_DIR=$S/adapter
export EXTRACT_DIR=$S/token_deltas
export FRONTIER_DIR=$S/frontier
export RECEIVER_USER_FILE=$S/user_splits_r42/confirmation_users.txt

export LAYER=26 LATENT_MULT=2 TOPK=4
export BATCH_SIZE=12 LOSS_BATCH_SIZE=4 PATCH_CHUNK_SIZE=8 SAE_BATCH_SIZE=2048
export TOKEN_DELTA_DTYPE=float32
export CONTEXT_MODES=team,role,dept,dept_role
export TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002
export ALPHAS=0.25,0.5,0.75,1.0
export EXCLUDE_SAME_USER_DONORS=1
export GPU_POLL_SEC=5
export TOKEN_CLASS_SCHEMA=cert

cd "$REPO_DIR"
mkdir -p logs

if [[ "$MODE" == "smoke" ]]; then
  export MAX_RECEIVERS=24 ALPHAS=1.0 CONTEXT_MODES=team
  export OUTPUT_DIR=$OUTROOT/SMOKE/l26_m02_k04
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu-debug -A cis260991-gpu -t 00:30:00 --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch
elif [[ "$MODE" == "full" ]]; then
  export MAX_RECEIVERS=0
  export OUTPUT_DIR=$OUTROOT/confirmation/l26_m02_k04_top5_control5_active
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu -A cis260991-gpu -t 24:00:00 --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch
else
  echo "usage: $0 {smoke|full}" >&2; exit 2
fi
