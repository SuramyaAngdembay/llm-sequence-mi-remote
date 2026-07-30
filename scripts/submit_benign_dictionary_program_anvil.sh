#!/bin/bash
# Dictionary-level independence program: benign-only SAEs -> held-out feature
# reselection -> confirmation/LOUO evals -> attribution. Fully chained.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"
P=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI

# A: benign-only SAE trainings
A1=$(DATASET=r42 SAE_SEED=42 BENIGN_ONLY=1 OUT_TAG=benign sbatch --parsable -p gpu-debug -t 00:30:00 -J bsae_r42 --export=ALL slurm/sae_seed_retrain.sbatch)
A2=$(DATASET=r62 SAE_SEED=42 BENIGN_ONLY=1 OUT_TAG=benign sbatch --parsable -p gpu -t 01:00:00 -J bsae_r62 --export=ALL slurm/sae_seed_retrain.sbatch)
echo "A: $A1 $A2"

# B: reselection under new dictionaries (held-out discipline)
B1=$(sbatch --parsable -p gpu-debug -t 00:30:00 -J bresel_r42 --dependency=afterok:$A1 \
  -A cis230270-gpu --gres=gpu:1 --cpus-per-task=16 --mem=180G \
  -o $P/logs/%x_%j.out -e $P/logs/%x_%j.err \
  --wrap "module load conda/2026.03 && conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3 && cd $REPO_DIR && python scripts/reselect_token_sae_features.py --extract-dir $P/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on --data-dir $P/outputs/session_jsonl_r42 --frontier-dir $P/outputs/token_delta_sae_frontier_r42_benign --out-frontier-dir $P/outputs/token_delta_sae_frontier_r42_benign_discovery --layer 26 --latent-mult 2 --k 4 --discovery-user-file $P/outputs/user_splits_r42/discovery_users.txt --benign-sample-prob 0.25 --device cuda --seed 42")
B2=$(sbatch --parsable -p gpu -t 01:00:00 -J bresel_r62 --dependency=afterok:$A2 \
  -A cis230270-gpu --gres=gpu:1 --cpus-per-task=16 --mem=240G \
  -o $P/logs/%x_%j.out -e $P/logs/%x_%j.err \
  --wrap "module load conda/2026.03 && conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3 && cd $REPO_DIR && python scripts/reselect_token_sae_features.py --extract-dir $P/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2 --data-dir $P/outputs/session_jsonl --frontier-dir $P/outputs/token_delta_sae_frontier_r62_benign --out-frontier-dir $P/outputs/token_delta_sae_frontier_r62_benign_louo --layer 18 --latent-mult 4 --k 8 --louo-splits-dir $P/outputs/user_splits_r62 --benign-sample-prob 0.05 --device cuda --seed 42")
echo "B: $B1 $B2"

# C1: r4.2 confirmation evals under benign dictionary
export CONDA_ENV=/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3
export CONFIG=configs/qwen3_8b_qlora_session_targeted.yaml
export DATA_DIR=$P/outputs/session_jsonl_r42
export ADAPTER_DIR=$P/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter
export EXTRACT_DIR=$P/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on
export FRONTIER_DIR=$P/outputs/token_delta_sae_frontier_r42_benign_discovery
export LAYER=26 LATENT_MULT=2 TOPK=4
export BATCH_SIZE=12 LOSS_BATCH_SIZE=4 PATCH_CHUNK_SIZE=8 SAE_BATCH_SIZE=2048
export TOKEN_DELTA_DTYPE=float32 CONTEXT_MODES=team,role,dept,dept_role
export TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002
export ALPHAS=0.25,0.5,0.75,1.0 GPU_POLL_SEC=5
export RECEIVER_USER_FILE=$P/outputs/user_splits_r42/confirmation_users.txt
export OUTPUT_DIR=$P/outputs/token_delta_sae_causal_qwen3_8b_r42_confirmation_benignsae/l26_m02_k04
export EXCLUDE_SAME_USER_DONORS=1
mkdir -p "$OUTPUT_DIR"
C1A=$(sbatch --parsable -p gpu -A cis230270-gpu -t 20:00:00 -J bconf_causal --dependency=afterok:$B1 --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch)
export OUTPUT_DIR=$P/outputs/token_delta_sae_necessity_qwen3_8b_r42_confirmation_benignsae/l26_m02_k04
export EXCLUDE_SAME_USER_MATCHES=1 MAX_PAIRS=0
mkdir -p "$OUTPUT_DIR"
C1B=$(sbatch --parsable -p gpu -A cis230270-gpu -t 06:00:00 -J bconf_necess --dependency=afterok:$B1 --export=ALL slurm/eval_token_delta_sae_necessity.template.sbatch)
echo "C1: $C1A $C1B"

# C2: r6.2 LOUO evals under benign dictionary
export DATA_DIR=$P/outputs/session_jsonl
export ADAPTER_DIR=$P/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter
export EXTRACT_DIR=$P/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2
export LAYER=18 LATENT_MULT=4 TOPK=8 BATCH_SIZE=16
export CONTEXT_MODES=team,role,project_role,dept_role
for USER_ID in ACM2278 CDE1846 CMP2946 MBG3183; do
  export FRONTIER_DIR=$P/outputs/token_delta_sae_frontier_r62_benign_louo/louo_${USER_ID}
  export RECEIVER_USER_FILE=$P/outputs/user_splits_r62/louo_${USER_ID}_confirmation_users.txt
  export OUTPUT_DIR=$P/outputs/token_delta_sae_causal_qwen3_8b_r62_benignsae_louo/louo_${USER_ID}
  export EXCLUDE_SAME_USER_DONORS=1
  mkdir -p "$OUTPUT_DIR"
  if [ "$USER_ID" = "CDE1846" ]; then T=01:30:00; else T=01:00:00; fi
  sbatch -p gpu -A cis230270-gpu -t $T -J bl_causal_${USER_ID} --dependency=afterok:$B2 --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch
  export OUTPUT_DIR=$P/outputs/token_delta_sae_necessity_qwen3_8b_r62_benignsae_louo/louo_${USER_ID}
  export EXCLUDE_SAME_USER_MATCHES=1 MAX_PAIRS=0
  mkdir -p "$OUTPUT_DIR"
  sbatch -p gpu -A cis230270-gpu -t 00:30:00 -J bl_necess_${USER_ID} --dependency=afterok:$B2 --export=ALL slurm/eval_token_delta_sae_necessity.template.sbatch
done
echo "chain submitted"
