#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cert-qlora-MI/llm-sequence-mi-remote}"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3}"
CONFIG="${CONFIG:-configs/qwen3_8b_qlora_session_targeted.yaml}"
OUT_DIR="${OUT_DIR:-/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_vram_benchmark_70k_probe}"

# Prior checkpointed benchmark:
#   micro_bs=12 -> 64.134 GiB allocated / 65.490 GiB reserved
# Linear extrapolation puts micro_bs=13 near 69 GiB allocated, close to 70k MiB.
# Keep 14 out of the default probe; set MICRO_BATCHES=12,13,14 manually if needed.
MICRO_BATCHES="${MICRO_BATCHES:-12,13}"
SEQ_LEN="${SEQ_LEN:-2048}"
BENCH_STEPS="${BENCH_STEPS:-2}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GC_MODE="${GC_MODE:-on}"

MICRO_BATCHES="$MICRO_BATCHES" \
  SEQ_LEN="$SEQ_LEN" \
  BENCH_STEPS="$BENCH_STEPS" \
  ATTN_IMPL="$ATTN_IMPL" \
  GC_MODE="$GC_MODE" \
  OUT_DIR="$OUT_DIR" \
  REPO_DIR="$REPO_DIR" \
  CONDA_ENV="$CONDA_ENV" \
  CONFIG="$CONFIG" \
  sbatch --parsable --export=ALL slurm/benchmark_qwen3_8b_vram.template.sbatch
