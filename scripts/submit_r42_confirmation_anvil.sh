#!/bin/bash
# Submit the r4.2 confirmation-user causal + necessity evals on the A100 gpu
# partition. Feature sets come from the discovery-only reselection; receivers
# are restricted to held-out confirmation users. Batch size from the gpu-debug
# VRAM probe (bs=12 -> ~30.0 GiB peak on worst-case sequences; bs=16 measured
# 37.9 GiB on a 39.5 GiB card and bs>=20 OOMs).
set -euo pipefail
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
export GPU_POLL_SEC=5

export OUTPUT_DIR=$P/outputs/token_delta_sae_causal_qwen3_8b_r42_confirmation/l26_m02_k04_top5_control5_active_confirmation
export EXCLUDE_SAME_USER_DONORS=1
mkdir -p "$OUTPUT_DIR"
sbatch -p gpu -A cis230270-gpu -t 20:00:00 --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch

export OUTPUT_DIR=$P/outputs/token_delta_sae_necessity_qwen3_8b_r42_confirmation/l26_m02_k04_top5_control5_active_necessity_confirmation
export EXCLUDE_SAME_USER_MATCHES=1 MAX_PAIRS=0
mkdir -p "$OUTPUT_DIR"
sbatch -p gpu -A cis230270-gpu -t 06:00:00 --export=ALL slurm/eval_token_delta_sae_necessity.template.sbatch
