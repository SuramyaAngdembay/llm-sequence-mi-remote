#!/bin/bash
# Stage B.A/B.B (docs/review/STAGE_B_PROTOCOL_2026-09-28.md sections 1-2): bridge the batch-1 class-score
# reference bit-exactly, then score the 1,949 missing r4.2 unseen days of the 10 newly positive insiders.
# Cap 0.5 GPU-h (one A100, 30 min wall). Collaborator allocation only.
#SBATCH --job-name=review_b1
#SBATCH --output=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/x-bbhusal1/cert_review_20260928/logs/%x-%j.err
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=12 --mem=96G
#SBATCH --time=00:30:00
#SBATCH -A cis260991-gpu -p gpu-debug
set -euo pipefail
W=/anvil/scratch/x-bbhusal1/cert_review_20260928
mkdir "$W/lock_b1" 2>/dev/null || { echo "B1 already ran or is running (lock present); exiting"; exit 0; }
module load conda/2025.02
source activate "$SCRATCH/conda_envs/cert-qlora"
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$HOME/llm-sequence-mi-remote"          # scorer identical to the cached run (sha256 checked below)
D=/anvil/scratch/x-sangdembay/r42_portability
CFG=configs/qwen3_8b_qlora_session_targeted.yaml
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch,transformers,peft,bitsandbytes,accelerate;print(torch.__version__,transformers.__version__,peft.__version__,bitsandbytes.__version__,accelerate.__version__)"
sha256sum scripts/score_token_class_decomposition.py scripts/token_class_decomposition.py $CFG \
  "$D/adapter/adapter_model.safetensors" "$W/b1_inputs/bridge/eval.jsonl" "$W/b1_inputs/extension/eval.jsonl"
test "$(sha256sum scripts/score_token_class_decomposition.py | cut -c1-16)" = "9a6ff58f74431a9d"
test "$(sha256sum scripts/token_class_decomposition.py | cut -c1-16)" = "00020e47f97f379e"

echo "=== [1] bridge: 256 cached batch-1 examples ==="
/usr/bin/time -v python scripts/score_token_class_decomposition.py --config $CFG --data-dir "$W/b1_inputs/bridge" \
  --adapter-dir "$D/adapter" --output "$W/b1_out/bridge/class_scores.parquet" --schema cert --batch-size 1 \
  2>&1 | grep -v "Loading weights"

echo "=== [2] bridge gate: bit-exact equality with the cached batch-1 pilot ==="
python - <<'PY'
import json, sys, pandas as pd, numpy as np
W = "/anvil/scratch/x-bbhusal1/cert_review_20260928"
new = pd.read_parquet(f"{W}/b1_out/bridge/class_scores.parquet").set_index("example_id")
old = pd.read_parquet("/anvil/scratch/x-bbhusal1/decomp/r42_headline/pilot_b1/class_scores.parquet").set_index("example_id")
mo = json.load(open(f"{W}/b1_out/bridge/class_scores_manifest.json"))
mc = json.load(open("/anvil/scratch/x-bbhusal1/decomp/r42_headline/class_scores_manifest.json"))
res = {"same_ids_and_order": list(new.index) == list(old.index),
       "adapter_digest_equal": mo["adapter_digest"] == mc["adapter_digest"],
       "tokenizer_digest_equal": mo["tokenizer_digest"] == mc["tokenizer_digest"],
       "gpu": mo["gpu"], "torch": mo["torch"]}
cols = [c for c in old.columns if c.startswith(("loss_sum_", "n_")) or c in ("adapted_nll", "n_tokens", "n_targets")]
diff = {c: int((new.loc[old.index, c].to_numpy() != old[c].to_numpy()).sum()) for c in cols}
res["columns_compared"] = len(cols); res["nonequal_cells_by_column"] = {c: v for c, v in diff.items() if v}
res["bit_exact"] = res["same_ids_and_order"] and res["adapter_digest_equal"] and res["tokenizer_digest_equal"] and not res["nonequal_cells_by_column"]
res["max_abs_adapted_nll_diff"] = float(np.abs(new.loc[old.index, "adapted_nll"].to_numpy() - old.adapted_nll.to_numpy()).max())
json.dump(res, open(f"{W}/b1_out/bridge_gate.json", "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["bit_exact"] else 3)
PY

echo "=== [3] extension: 1,949 missing unseen days ==="
/usr/bin/time -v python scripts/score_token_class_decomposition.py --config $CFG --data-dir "$W/b1_inputs/extension" \
  --adapter-dir "$D/adapter" --output "$W/b1_out/extension/class_scores.parquet" --schema cert --batch-size 1 \
  --progress-every 500 2>&1 | grep -v "Loading weights"
sha256sum "$W"/b1_out/*/class_scores.parquet
echo REVIEW_B1_DONE
