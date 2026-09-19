#!/bin/bash
# Work package 4: re-run the r4.2 confirmation causal intervention with
# per-token-class loss accumulation, so each repair can be read against the
# full, profile-only and behaviour-only views instead of the pooled mean only.
#
# Everything except --token-class-schema is byte-identical to
# scripts/submit_r42_confirmation_anvil.sh, which produced job 19379904. The
# adapter, SAE frontier, layer/latent_mult/k, feature sets, context modes,
# alphas and donor-exclusion rule are unchanged, so the new columns describe
# the SAME interventions the published result describes.
#
# Usage:
#   bash scripts/submit_r42_token_class_causal_anvil.sh smoke   # debug queue first
#   bash scripts/submit_r42_token_class_causal_anvil.sh full
set -euo pipefail
MODE="${1:-smoke}"
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
P=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI

export CONDA_ENV=/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3
export CONFIG=configs/qwen3_8b_qlora_session_targeted.yaml
export DATA_DIR=$P/outputs/session_jsonl_r42
export ADAPTER_DIR=$P/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter
export EXTRACT_DIR=$P/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on
export FRONTIER_DIR=$P/outputs/token_delta_sae_frontier_r42_discovery_split
export LAYER=26 LATENT_MULT=2 TOPK=4
export BATCH_SIZE=12 LOSS_BATCH_SIZE=4 PATCH_CHUNK_SIZE=8 SAE_BATCH_SIZE=2048
export TOKEN_DELTA_DTYPE=float32
export CONTEXT_MODES=team,role,dept,dept_role
export TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002
export ALPHAS=0.25,0.5,0.75,1.0
export RECEIVER_USER_FILE=$P/outputs/user_splits_r42/confirmation_users.txt
export EXCLUDE_SAME_USER_DONORS=1
export GPU_POLL_SEC=5

# The new work: decompose every intervention by token class.
export TOKEN_CLASS_SCHEMA=cert

if [[ "$MODE" == "smoke" ]]; then
  # Debug queue, a handful of receivers and one alpha. The point is to confirm
  # the per-class assertions hold on real tokenized CERT text -- the per-batch
  # reconstruction check and the base/patched count equality -- before paying
  # for the full run. Cheap enough to discard.
  export MAX_RECEIVERS=24
  export ALPHAS=1.0
  export CONTEXT_MODES=team
  export OUTPUT_DIR=$P/outputs/token_class_causal_r42_SMOKE/l26_m02_k04
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu-debug -A cis230270-gpu -t 00:30:00 --export=ALL \
    slurm/eval_token_delta_sae_causal.template.sbatch
elif [[ "$MODE" == "full" ]]; then
  export MAX_RECEIVERS=0
  export OUTPUT_DIR=$P/outputs/token_class_causal_r42_confirmation/l26_m02_k04_top5_control5_active
  mkdir -p "$OUTPUT_DIR"
  # 20 h was enough for the original; the extra per-pair-batch base forward
  # adds roughly one forward per eight patched forwards (~12%), so 24 h.
  sbatch -p gpu -A cis230270-gpu -t 24:00:00 --export=ALL \
    slurm/eval_token_delta_sae_causal.template.sbatch
else
  echo "usage: $0 {smoke|full}" >&2
  exit 2
fi
