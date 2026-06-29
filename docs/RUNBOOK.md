# Runbook

## Current Direction

The `Qwen 3B` token-level branch is no longer just exploratory:

- matched day-level comparison: remote token > local session-AE > local residual
- positive bootstrap intervals on the strongest remote token configs

So the current path is:

1. keep the `3B` token results as the comparison anchor
2. scale the **same token-level protocol** to `Qwen3-8B`
3. do not rerun broad mean-pooled or generic frontier work first
4. after the `Qwen3-8B` frontier, run model-level causal patching on the top layer-18 configs

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

Completed first target:

- `Qwen 3B`
- structured session-sequence next-token modeling
- benign-only training split

Current scale-up target:

- `Qwen3-8B`
- same structured session JSONL
- same benign-only objective
- `4x H100 80GB` single-node DDP on Anvil `ai`
- submitted start: `MICRO_BS=2`, `GRAD_ACCUM=4`, `NPROC=4`

Required outputs:

- adapter checkpoint
- tokenizer/config snapshot
- training metrics

Primary DDP command on Anvil:

```bash
sbatch slurm/train_qlora_ddp.template.sbatch
```

Targeted `Qwen3-8B` launcher:

```bash
bash scripts/submit_qwen3_8b_targeted_pipeline_anvil.sh
```

## Phase 3: Delta Extraction

Extract:

- base hidden states
- adapted hidden states
- adapter deltas `h_adapted - h_base`

Initial `3B` layer set:

- 3 to 4 middle/late layers

Do not extract all layers on the first pass.

Targeted `Qwen3-8B` layer band:

- `18`
- `26`
- `34`

Do not reopen the old mean-pooled extraction path for `Qwen3-8B`.

Required outputs:

- chunked delta tensors
- metadata linking chunk -> layer -> example ids

Primary command on Anvil:

```bash
sbatch slurm/extract_deltas.template.sbatch
```

Token-level extraction:

```bash
sbatch slurm/extract_token_deltas.template.sbatch
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

Token-level frontier:

```bash
sbatch slurm/train_token_delta_sae.template.sbatch
```

Targeted `Qwen3-8B` frontier policy:

- run only the promising sparse regimes first:
  - `latent_mult=2, k=8`
  - `latent_mult=4, k=4`
- after the completed frontier, prioritize layer `18` for the first causal patch runs
- if a layer sweep is needed, keep it to the narrow `18/26/34` band
- do not rerun a large exploratory frontier unless the targeted path fails

## Phase 5: Causal Evaluation

For the best SAE candidates:

- top-vs-control ablation
- semantic grounding
- sparse patch/repair
- residual comparison

Primary command on Anvil:

```bash
sbatch slurm/eval_delta_sae_causal.template.sbatch
```

Default first target:

- `layer=18`
- `latent_mult=2`
- `k=8`

Next comparison target:

- `layer=18`
- `latent_mult=4`
- `k=4`

Token-level escalation after null mean-pooled patchability:

```bash
sbatch slurm/eval_token_delta_sae_causal.template.sbatch
```

Recommended first token-level path:

1. `extract_token_deltas.template.sbatch`
2. `train_token_delta_sae.template.sbatch`
3. `eval_token_delta_sae_causal.template.sbatch`

Targeted `Qwen3-8B` eval bundle:

```bash
bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
```

Recommended first `Qwen3-8B` causal suite:

```bash
bash scripts/submit_qwen3_8b_recommended_causal_suite_anvil.sh
```

## Phase 6: Qwen3-8B Targeted Scale-Up

This phase is justified only because the `3B` token branch already cleared the matched-comparison bar.

What stays fixed:

- structured session JSONL
- token-level delta extraction
- matched donor/receiver protocol
- matched day-level local comparison protocol
- control-pool fix from the `m04/k04_controlfix` branch

What changes:

- base model becomes `Qwen/Qwen3-8B`
- training uses 4-GPU DDP as the default path
- extraction/eval stay token-level

What not to rerun:

- no mean-pooled delta path
- no broad graph detour first
- no full generic SAE frontier before checking the targeted token configs

## Success Condition

The LLM branch is only a win if it beats the current base session AE branch on:

- **sufficiency / patchability**

not merely on:

- detection quality
