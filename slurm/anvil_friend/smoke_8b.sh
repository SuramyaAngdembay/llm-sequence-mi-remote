#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH -A tra250034-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J smoke8b
#SBATCH -o /anvil/scratch/x-bbhusal1/p1_8b/smoke8b_%j.out
set -e
module load conda/2025.02
source activate $SCRATCH/conda_envs/cert-qlora
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$SCRATCH/hf_cache TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
export PYTHONPATH=$SCRATCH/lanl/patchlib:$PYTHONPATH
cd $HOME/llm-sequence-mi-remote
P=$SCRATCH/p1_8b; SRC=$SCRATCH/p1/jsonl_r62_full; D=$P/smoke_data; CFG=configs/qwen3_8b_qlora_session_targeted.yaml
rm -rf $D $P/smoke_adapter $P/smoke_deltas $P/smoke_pool; mkdir -p $D $P/smoke_pool
head -400 $SRC/train.jsonl > $D/train.jsonl
head -64 $SRC/val.jsonl > $D/val.jsonl
head -80 $SRC/eval.jsonl > $P/smoke_pool/eval.jsonl
grep -m 20 "\"y\": 1" $SRC/eval.jsonl >> $P/smoke_pool/eval.jsonl 2>/dev/null || true
cp $P/smoke_pool/eval.jsonl $D/eval.jsonl
echo "[smoke 8b DDP train nproc=2]"
torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_qlora.py   --config $CFG --data-dir $D --output-dir $P/smoke_adapter --seed 42   --micro-batch-size 2 --grad-accum 2 --max-train-examples 256 --max-val-examples 32 --dataloader-workers 4
echo "SMOKE_TRAIN_OK"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[smoke 8b extract]"
CUDA_VISIBLE_DEVICES=0 python scripts/extract_adapter_deltas.py --config $CFG --data-dir $P/smoke_pool --adapter-dir $P/smoke_adapter/adapter --output-dir $P/smoke_deltas --split eval --pool-unit token --layers 12,18,24 --batch-size 1 --chunk-examples 512
echo "SMOKE_EXTRACT_OK"
python - << PY
import pandas as pd
d=pd.read_parquet("$P/smoke_deltas/example_scores.parquet")
print("scores rows:",len(d),"pos:",int((d.y==1).sum()),"cols_ok:", set(["adapted_nll","y","user_id"]).issubset(d.columns))
PY
echo "SMOKE_8B_OK"
