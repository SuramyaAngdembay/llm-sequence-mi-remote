#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp}"
TOKEN_CACHE_ROOT="${TOKEN_CACHE_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted}"
FRONTIER_ROOT="${FRONTIER_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b}"

NPROC="${NPROC:-4}"
MICRO_BS="${MICRO_BS:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DL_WORKERS="${DL_WORKERS:-16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GC_MODE="${GC_MODE:-on}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VAL="${MAX_VAL:-0}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-0}"
SKIP_RNG_STATE_RESUME="${SKIP_RNG_STATE_RESUME:-1}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"

EXTRACT_LAYERS="${EXTRACT_LAYERS:-18,26,34}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
EXTRACT_CHUNK_EXAMPLES="${EXTRACT_CHUNK_EXAMPLES:-512}"
EXTRACT_MAX_EXAMPLES="${EXTRACT_MAX_EXAMPLES:-0}"

SAE_MAX_ROWS="${SAE_MAX_ROWS:-2000000}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-4096}"
LATENT_MULTIPLIERS="${LATENT_MULTIPLIERS:-2,4}"
TOPK_VALUES="${TOPK_VALUES:-4,8}"

TRAIN_DEPENDENCY_ARGS=()
if [[ -n "$TRAIN_DEPENDENCY" ]]; then
  TRAIN_DEPENDENCY_ARGS+=(--dependency="$TRAIN_DEPENDENCY")
fi

TRAIN_JOB=$(
  REPO_DIR="$REPO_DIR" \
    CONDA_ENV="$CONDA_ENV" \
    CONFIG="$CONFIG" \
    DATA_DIR="$DATA_DIR" \
    OUTPUT_DIR="$CHECKPOINT_ROOT" \
    NPROC="$NPROC" \
    MICRO_BS="$MICRO_BS" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    DL_WORKERS="$DL_WORKERS" \
    ATTN_IMPL="$ATTN_IMPL" \
    GC_MODE="$GC_MODE" \
    MAX_TRAIN="$MAX_TRAIN" \
    MAX_VAL="$MAX_VAL" \
    SAVE_STEPS="$SAVE_STEPS" \
    EVAL_STRATEGY="$EVAL_STRATEGY" \
    EVAL_STEPS="$EVAL_STEPS" \
    RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
    IGNORE_DATA_SKIP="$IGNORE_DATA_SKIP" \
    SKIP_RNG_STATE_RESUME="$SKIP_RNG_STATE_RESUME" \
    sbatch --parsable \
    "${TRAIN_DEPENDENCY_ARGS[@]}" \
    --export=ALL \
    slurm/train_qlora_ddp.template.sbatch
)

EXTRACT_JOB=$(
  REPO_DIR="$REPO_DIR" \
    CONDA_ENV="$CONDA_ENV" \
    CONFIG="$CONFIG" \
    DATA_DIR="$DATA_DIR" \
    ADAPTER_DIR="${CHECKPOINT_ROOT}/adapter" \
    OUTPUT_DIR="$TOKEN_CACHE_ROOT" \
    LAYERS="$EXTRACT_LAYERS" \
    BATCH_SIZE="$EXTRACT_BATCH_SIZE" \
    CHUNK_EXAMPLES="$EXTRACT_CHUNK_EXAMPLES" \
    MAX_EXAMPLES="$EXTRACT_MAX_EXAMPLES" \
    sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
    --export=ALL \
    slurm/extract_token_deltas.template.sbatch
)

SAE_JOB=$(
  REPO_DIR="$REPO_DIR" \
    CONDA_ENV="$CONDA_ENV" \
    EXTRACT_DIR="$TOKEN_CACHE_ROOT" \
    OUTPUT_DIR="$FRONTIER_ROOT" \
    LAYERS="$EXTRACT_LAYERS" \
    MAX_ROWS="$SAE_MAX_ROWS" \
    BATCH_SIZE="$SAE_BATCH_SIZE" \
    LATENT_MULTIPLIERS="$LATENT_MULTIPLIERS" \
    TOPK_VALUES="$TOPK_VALUES" \
    sbatch --parsable --dependency=afterok:${EXTRACT_JOB} \
    --export=ALL \
    slurm/train_token_delta_sae.template.sbatch
)

cat <<EOM
submitted_qwen3_8b_targeted_pipeline=1
train_job=${TRAIN_JOB}
extract_job=${EXTRACT_JOB}
sae_job=${SAE_JOB}
conda_env=${CONDA_ENV}
config=${CONFIG}
checkpoint_root=${CHECKPOINT_ROOT}
token_cache_root=${TOKEN_CACHE_ROOT}
frontier_root=${FRONTIER_ROOT}
nproc=${NPROC}
micro_bs=${MICRO_BS}
grad_accum=${GRAD_ACCUM}
gc_mode=${GC_MODE}
eval_strategy=${EVAL_STRATEGY}
resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}
ignore_data_skip=${IGNORE_DATA_SKIP}
skip_rng_state_resume=${SKIP_RNG_STATE_RESUME}
train_dependency=${TRAIN_DEPENDENCY}
extract_layers=${EXTRACT_LAYERS}
EOM
