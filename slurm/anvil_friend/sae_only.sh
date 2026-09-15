#!/bin/bash
#SBATCH -p gpu
#SBATCH -A tra250034-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 06:00:00
set -e
MODE=$1
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1
echo "[sae+attribution $MODE retry hi-mem]"
python scripts/train_delta_sae_frontier.py --config configs/token_delta_sae_frontier.yaml --extract-dir $P/deltas_$MODE --out-dir $P/sae_$MODE --device cuda --layers 12,18,24 --latent-multipliers 4 --topk 8 --benign-only --seed 42 --max-rows 2000000 --batch-size 4096
python scripts/audit3b_select_attribute.py --extract-dir $P/deltas_$MODE --data-dir $P/pool_$MODE --frontier-dir $P/sae_$MODE --adapter-dir $P/adapter_$MODE/adapter --out-dir $P/audit_$MODE --latent-mult 4 --k 8
echo "P1_SAE_${MODE}_DONE"
