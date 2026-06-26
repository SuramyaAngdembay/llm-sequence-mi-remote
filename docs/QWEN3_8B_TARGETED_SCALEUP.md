# Qwen3-8B Targeted Scale-Up

This supersedes the earlier `Qwen/Qwen2.5-7B` target for the next scale-up run.

## Rationale

The `Qwen 3B` token branch beat the matched local session-AE day-level baseline
and stayed positive under bootstrap. The next run should therefore keep the same
mechanistic protocol and upgrade only the base sequence model.

## Target

- base model: `Qwen/Qwen3-8B`
- training: 4-bit NF4 QLoRA with bf16 compute and LoRA adapters
- Anvil partition: `ai`
- training hardware: one 4x H100 80GB node
- extraction/eval hardware: one H100 80GB
- Qwen3 environment: `/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3`

QLoRA does not require a separate pre-quantized checkpoint. The normal Hugging
Face model weights are loaded with `BitsAndBytesConfig(load_in_4bit=True)`.

## First Targeted Band

`Qwen3-8B` has a deeper 36-layer stack, so the first token-delta extraction band is:

- layer `18`
- layer `26`
- layer `34`

First SAE regimes remain:

- `latent_mult=2, k=8`
- `latent_mult=4, k=4`

## Launch

After the Qwen3-capable env and HF cache are validated:

```bash
bash scripts/submit_qwen3_8b_targeted_pipeline_anvil.sh
```

The original conservative launcher kept the effective batch at `32`:

- `NPROC=4`
- `MICRO_BS=2`
- `GRAD_ACCUM=4`

## VRAM Retune

The first live H100 run, Slurm `18597248`, used only about `20 GB` per `80 GB`
H100 with `MICRO_BS=2`, `GRAD_ACCUM=4`, and gradient checkpointing enabled.
It was healthy but underutilized, and the full validation pass every `1000`
steps cost about `36.5` minutes.

Retune target:

- aim for roughly `60-70 GB` per H100, leaving some headroom under the `80 GB`
  H100 limit
- use `MICRO_BS=16`, `GRAD_ACCUM=1`
- disable gradient checkpointing with `GC_MODE=off`
- disable intermediate eval with `EVAL_STRATEGY=no`
- keep checkpoint saves every `1000` steps

The high-VRAM retune intentionally changes the effective batch from `32` to `64`.
If it OOMs, fall back first to `MICRO_BS=12`, `GRAD_ACCUM=1`, `GC_MODE=off`;
if that still OOMs, use `MICRO_BS=16`, `GRAD_ACCUM=1`, `GC_MODE=config`.

For retunes after an already-running job, use `RESUME_FROM_CHECKPOINT=latest`
so the Slurm script resolves the newest `checkpoint-*` under `OUTPUT_DIR` at
job start.

Submitted high-VRAM retune on 2026-06-26:

- left conservative training job `18597248` running on `h014`
- canceled stale conservative downstream jobs `18597250` and `18597251`
- submitted high-VRAM training job `18615954`, dependency `afterany:18597248`
- submitted high-VRAM token extraction `18615955`, dependency `afterok:18615954`
- submitted high-VRAM token SAE `18615957`, dependency `afterok:18615955`

High-VRAM submission settings:

- `MICRO_BS=16`
- `GRAD_ACCUM=1`
- `GC_MODE=off`
- `EVAL_STRATEGY=no`
- `SAVE_STEPS=1000`
- `RESUME_FROM_CHECKPOINT=latest`
- `IGNORE_DATA_SKIP=1`

This is the first attempt to push VRAM utilization toward the requested
`~70 GB/H100` range. Verify actual memory with:

```bash
srun --jobid=18615954 --overlap --nodes=1 --ntasks=1 \
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

## Original Conservative Pipeline

Submitted on 2026-06-26 from Anvil `login02`:

- training: Slurm `18597248`, `qwen_qlora_ddp`, partition `ai`, `4x H100`
- token extraction: Slurm `18597250`, canceled after the VRAM retune was queued
- token SAE frontier: Slurm `18597251`, canceled after the VRAM retune was queued

Runtime inputs:

- conda env: `/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3`
- config: `configs/qwen3_8b_qlora_session_targeted.yaml`
- checkpoint root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp`
- token cache root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted`
- frontier root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b`

Qwen3 cache and environment validation:

- `Qwen/Qwen3-8B` is cached under project `HF_HOME`
- offline `AutoConfig` loads as `model_type=qwen3`
- offline tokenizer loads with the fast tokenizer
- `transformers=4.51.3`, `tokenizers=0.21.4`

## Success Criterion

The run only advances the branch if it improves mechanistic repair evidence:

- beat the current control-fixed token best effect, `0.001446`
- keep positive bootstrap intervals
- widen the margin over the matched local session-AE day-level baseline, `0.001133`
- ideally stay positive across more than one context/top-set row
