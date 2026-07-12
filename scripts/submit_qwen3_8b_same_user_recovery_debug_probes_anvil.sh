#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

# Exact evaluator probes for the four same-user-excluded recovery branches.
# These are not paper results. They validate the repaired matching flags,
# A100/gpu-debug resource wiring, VRAM profile, and obvious OOM behavior before
# launching full recovery jobs.

GPU_DEBUG_PARTITION="${GPU_DEBUG_PARTITION:-gpu-debug}"
GPU_ACCOUNT="${GPU_ACCOUNT:-cis230270-gpu}"
GPU_MEM="${GPU_MEM:-240G}"
GPU_TIME="${GPU_TIME:-00:45:00}"
GPU_CPUS="${GPU_CPUS:-24}"
GPU_POLL_SEC="${GPU_POLL_SEC:-2}"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"

OUT_ROOT="${OUT_ROOT:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/same_user_recovery_debug_probes}"
mkdir -p "$OUT_ROOT"

R62_DATA_DIR="${R62_DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl}"
R62_ADAPTER_DIR="${R62_ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter}"
R62_EXTRACT_DIR="${R62_EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2}"
R62_FRONTIER_DIR="${R62_FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream}"

R42_DATA_DIR="${R42_DATA_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42}"
R42_ADAPTER_DIR="${R42_ADAPTER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter}"
R42_EXTRACT_DIR="${R42_EXTRACT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on}"
R42_FRONTIER_DIR="${R42_FRONTIER_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on}"

submit_causal_probe() {
  local name="$1"
  local data_dir="$2"
  local adapter_dir="$3"
  local extract_dir="$4"
  local frontier_dir="$5"
  local layer="$6"
  local latent_mult="$7"
  local topk="$8"
  local context_modes="$9"
  local batch_size="${10}"
  local loss_batch_size="${11}"
  local patch_chunk_size="${12}"
  local max_receivers="${13}"
  local out_dir="${OUT_ROOT}/${name}"

  mkdir -p "$out_dir"
  REPO_DIR="$REPO_DIR" \
  CONDA_ENV="$CONDA_ENV" \
  CONFIG="$CONFIG" \
  DATA_DIR="$data_dir" \
  ADAPTER_DIR="$adapter_dir" \
  EXTRACT_DIR="$extract_dir" \
  FRONTIER_DIR="$frontier_dir" \
  OUTPUT_DIR="$out_dir" \
  LAYER="$layer" \
  LATENT_MULT="$latent_mult" \
  TOPK="$topk" \
  TOP_SETS=top5 \
  CONTROL_SET=control5_active \
  ACTIVE_CONTROL_MIN_FRAC=0.002 \
  CONTEXT_MODES="$context_modes" \
  BATCH_SIZE="$batch_size" \
  LOSS_BATCH_SIZE="$loss_batch_size" \
  SAE_BATCH_SIZE=2048 \
  PATCH_CHUNK_SIZE="$patch_chunk_size" \
  TOKEN_DELTA_DTYPE=float32 \
  ALPHAS=0.25,0.5,0.75,1.0 \
  MAX_RECEIVERS="$max_receivers" \
  MAX_CANDIDATE_DONORS=16 \
  EXCLUDE_SAME_USER_DONORS=1 \
  GPU_POLL_SEC="$GPU_POLL_SEC" \
  sbatch --parsable \
    --partition="$GPU_DEBUG_PARTITION" \
    --account="$GPU_ACCOUNT" \
    --mem="$GPU_MEM" \
    --time="$GPU_TIME" \
    --cpus-per-task="$GPU_CPUS" \
    --export=ALL \
    slurm/eval_token_delta_sae_causal.template.sbatch
}

submit_necessity_probe() {
  local name="$1"
  local data_dir="$2"
  local adapter_dir="$3"
  local extract_dir="$4"
  local frontier_dir="$5"
  local layer="$6"
  local latent_mult="$7"
  local topk="$8"
  local context_modes="$9"
  local batch_size="${10}"
  local loss_batch_size="${11}"
  local patch_chunk_size="${12}"
  local max_pairs="${13}"
  local out_dir="${OUT_ROOT}/${name}"

  mkdir -p "$out_dir"
  REPO_DIR="$REPO_DIR" \
  CONDA_ENV="$CONDA_ENV" \
  CONFIG="$CONFIG" \
  DATA_DIR="$data_dir" \
  ADAPTER_DIR="$adapter_dir" \
  EXTRACT_DIR="$extract_dir" \
  FRONTIER_DIR="$frontier_dir" \
  OUTPUT_DIR="$out_dir" \
  LAYER="$layer" \
  LATENT_MULT="$latent_mult" \
  TOPK="$topk" \
  TOP_SETS=top5 \
  CONTROL_SET=control5_active \
  ACTIVE_CONTROL_MIN_FRAC=0.002 \
  CONTEXT_MODES="$context_modes" \
  BATCH_SIZE="$batch_size" \
  LOSS_BATCH_SIZE="$loss_batch_size" \
  SAE_BATCH_SIZE=2048 \
  PATCH_CHUNK_SIZE="$patch_chunk_size" \
  TOKEN_DELTA_DTYPE=float32 \
  ALPHAS=0.25,0.5,0.75,1.0 \
  MAX_PAIRS="$max_pairs" \
  EXCLUDE_SAME_USER_MATCHES=1 \
  GPU_POLL_SEC="$GPU_POLL_SEC" \
  sbatch --parsable \
    --partition="$GPU_DEBUG_PARTITION" \
    --account="$GPU_ACCOUNT" \
    --mem="$GPU_MEM" \
    --time="$GPU_TIME" \
    --cpus-per-task="$GPU_CPUS" \
    --export=ALL \
    slurm/eval_token_delta_sae_necessity.template.sbatch
}

R62_CAUSAL_BATCH="${R62_CAUSAL_BATCH:-24}"
R62_NECESSITY_BATCH="${R62_NECESSITY_BATCH:-32}"
R42_CAUSAL_BATCH="${R42_CAUSAL_BATCH:-16}"
R42_NECESSITY_BATCH="${R42_NECESSITY_BATCH:-16}"
MAX_RECEIVERS_PROBE="${MAX_RECEIVERS_PROBE:-16}"
MAX_PAIRS_PROBE="${MAX_PAIRS_PROBE:-16}"

R62_CAUSAL_JOB="$(
  submit_causal_probe \
    r62_causal_l18_m04_k08_no_same_user_probe_bs${R62_CAUSAL_BATCH} \
    "$R62_DATA_DIR" "$R62_ADAPTER_DIR" "$R62_EXTRACT_DIR" "$R62_FRONTIER_DIR" \
    18 4 8 team,role,project_role,dept_role \
    "$R62_CAUSAL_BATCH" 1 "$R62_CAUSAL_BATCH" "$MAX_RECEIVERS_PROBE"
)"

R62_NECESSITY_JOB="$(
  submit_necessity_probe \
    r62_necessity_l18_m04_k08_no_same_user_probe_bs${R62_NECESSITY_BATCH} \
    "$R62_DATA_DIR" "$R62_ADAPTER_DIR" "$R62_EXTRACT_DIR" "$R62_FRONTIER_DIR" \
    18 4 8 team,role,project_role,dept_role \
    "$R62_NECESSITY_BATCH" 4 "$R62_NECESSITY_BATCH" "$MAX_PAIRS_PROBE"
)"

R42_CAUSAL_JOB="$(
  submit_causal_probe \
    r42_causal_l26_m02_k04_no_same_user_probe_bs${R42_CAUSAL_BATCH} \
    "$R42_DATA_DIR" "$R42_ADAPTER_DIR" "$R42_EXTRACT_DIR" "$R42_FRONTIER_DIR" \
    26 2 4 team,role,dept,dept_role \
    "$R42_CAUSAL_BATCH" 1 "$R42_CAUSAL_BATCH" "$MAX_RECEIVERS_PROBE"
)"

R42_NECESSITY_JOB="$(
  submit_necessity_probe \
    r42_necessity_l26_m02_k04_no_same_user_probe_bs${R42_NECESSITY_BATCH} \
    "$R42_DATA_DIR" "$R42_ADAPTER_DIR" "$R42_EXTRACT_DIR" "$R42_FRONTIER_DIR" \
    26 2 4 team,role,dept,dept_role \
    "$R42_NECESSITY_BATCH" 4 "$R42_NECESSITY_BATCH" "$MAX_PAIRS_PROBE"
)"

cat > "${OUT_ROOT}/submitted_jobs.env" <<EOM
OUT_ROOT=${OUT_ROOT}
GPU_DEBUG_PARTITION=${GPU_DEBUG_PARTITION}
GPU_ACCOUNT=${GPU_ACCOUNT}
GPU_MEM=${GPU_MEM}
GPU_TIME=${GPU_TIME}
GPU_CPUS=${GPU_CPUS}
GPU_POLL_SEC=${GPU_POLL_SEC}
MAX_RECEIVERS_PROBE=${MAX_RECEIVERS_PROBE}
MAX_PAIRS_PROBE=${MAX_PAIRS_PROBE}
R62_CAUSAL_BATCH=${R62_CAUSAL_BATCH}
R62_NECESSITY_BATCH=${R62_NECESSITY_BATCH}
R42_CAUSAL_BATCH=${R42_CAUSAL_BATCH}
R42_NECESSITY_BATCH=${R42_NECESSITY_BATCH}
R62_CAUSAL_JOB=${R62_CAUSAL_JOB}
R62_NECESSITY_JOB=${R62_NECESSITY_JOB}
R42_CAUSAL_JOB=${R42_CAUSAL_JOB}
R42_NECESSITY_JOB=${R42_NECESSITY_JOB}
JOB_IDS=${R62_CAUSAL_JOB},${R62_NECESSITY_JOB},${R42_CAUSAL_JOB},${R42_NECESSITY_JOB}
EOM

cat <<EOM
submitted_same_user_recovery_debug_probes=1
out_root=${OUT_ROOT}
r62_causal_job=${R62_CAUSAL_JOB}
r62_necessity_job=${R62_NECESSITY_JOB}
r42_causal_job=${R42_CAUSAL_JOB}
r42_necessity_job=${R42_NECESSITY_JOB}
job_ids=${R62_CAUSAL_JOB},${R62_NECESSITY_JOB},${R42_CAUSAL_JOB},${R42_NECESSITY_JOB}

After completion:
  sacct -j ${R62_CAUSAL_JOB},${R62_NECESSITY_JOB},${R42_CAUSAL_JOB},${R42_NECESSITY_JOB} --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AveRSS,ExitCode -P
  find ${OUT_ROOT} -name 'gpu_poll_*.csv' -maxdepth 2 -print -exec tail -n 5 {} \\;
EOM

