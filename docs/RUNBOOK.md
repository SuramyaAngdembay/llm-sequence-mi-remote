# Runbook

## Phase 0: Remote Discovery

Before running anything, confirm:

- GPU model and count
- per-node RAM
- local scratch path and quota
- batch scheduler details
- preferred environment method
- transfer method:
  - `Globus`
  - `scp`
  - `rsync`
  - shared project storage

Use [ANVIL_DISCOVERY_CHECKLIST.md](./ANVIL_DISCOVERY_CHECKLIST.md).

## Phase 1: Data Staging

1. Copy session shards and labels to remote storage.
2. Verify checksums.
3. Build a smoke subset.
4. Serialize canonical JSONL sequence format.

Primary command on Anvil:

```bash
sbatch slurm/build_jsonl.template.sbatch
```

Expected outputs:

- `data/raw_shards/`
- `data/jsonl/train.jsonl`
- `data/jsonl/val.jsonl`
- `data/jsonl/eval.jsonl`
- `data/smoke/*.jsonl`

## Phase 2: QLoRA Fine-Tuning

First target:

- `Qwen 3B`
- structured session-sequence next-token modeling
- benign-only training split

Required outputs:

- adapter checkpoint
- tokenizer/config snapshot
- training metrics

Primary command on Anvil:

```bash
sbatch slurm/train_qlora.template.sbatch
```

## Phase 3: Delta Extraction

Extract:

- base hidden states
- adapted hidden states
- adapter deltas `h_adapted - h_base`

Initial layer set:

- 3 to 4 middle/late layers

Do not extract all layers on the first pass.

Required outputs:

- chunked delta tensors
- metadata linking chunk -> layer -> example ids

Primary command on Anvil:

```bash
sbatch slurm/extract_deltas.template.sbatch
```

## Phase 4: Delta-SAE

Train SAEs on the delta tensors.

Initial sweep:

- several latent widths
- several `k` values
- plain and optionally denoising variants

Track:

- reconstruction MSE
- effective `L0`
- dead features
- feature overlap
- anomaly selectivity

Primary command on Anvil:

```bash
sbatch slurm/train_delta_sae.template.sbatch
```

## Phase 5: Causal Evaluation

For the best SAE candidates:

- top-vs-control ablation
- semantic grounding
- sparse patch/repair
- residual comparison

## Success Condition

The LLM branch is only a win if it beats the current base session AE branch on:

- **sufficiency / patchability**

not merely on:

- detection quality
