#!/bin/bash
# Submit r6.2 LOUO per-fold causal + necessity evals on gpu/A100.
# Batch 16 from gpu-debug probe 19406535 (peak 25.7 GiB worst-case).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
P=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI

export CONDA_ENV=/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3
export CONFIG=configs/qwen3_8b_qlora_session_targeted.yaml
export DATA_DIR=$P/outputs/session_jsonl
export ADAPTER_DIR=$P/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter
export EXTRACT_DIR=$P/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2
export LAYER=18 LATENT_MULT=4 TOPK=8
export BATCH_SIZE=16 LOSS_BATCH_SIZE=4 PATCH_CHUNK_SIZE=8 SAE_BATCH_SIZE=2048
export TOKEN_DELTA_DTYPE=float32
export CONTEXT_MODES=team,role,project_role,dept_role
export TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002
export ALPHAS=0.25,0.5,0.75,1.0
export GPU_POLL_SEC=5

for USER_ID in ACM2278 CDE1846 CMP2946 MBG3183; do
  export FRONTIER_DIR=$P/outputs/token_delta_sae_frontier_r62_louo/louo_${USER_ID}
  export RECEIVER_USER_FILE=$P/outputs/user_splits_r62/louo_${USER_ID}_confirmation_users.txt

  export OUTPUT_DIR=$P/outputs/token_delta_sae_causal_qwen3_8b_r62_louo/louo_${USER_ID}
  export EXCLUDE_SAME_USER_DONORS=1
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu -A cis230270-gpu -t 02:30:00 -J louo_causal_${USER_ID} --export=ALL \
    slurm/eval_token_delta_sae_causal.template.sbatch

  export OUTPUT_DIR=$P/outputs/token_delta_sae_necessity_qwen3_8b_r62_louo/louo_${USER_ID}
  export EXCLUDE_SAME_USER_MATCHES=1 MAX_PAIRS=0
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu -A cis230270-gpu -t 01:00:00 -J louo_necess_${USER_ID} --export=ALL \
    slurm/eval_token_delta_sae_necessity.template.sbatch
done
