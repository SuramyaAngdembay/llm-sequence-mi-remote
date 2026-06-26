# Qwen 7B Targeted Scale-Up

This note records the next remote direction after the `Qwen 3B` token-level branch cleared the strict comparison bar.

## Why 7B Is Justified Now

The `3B` token-level branch is no longer speculative:

- matched day-level comparison beat the local session-AE baseline
- `m02/k08` bootstrap stayed positive
- `m04/k04_controlfix` bootstrap also stayed positive with a real 3-feature control

So `7B` is now a scale-up of a working intervention path, not a blind model-size gamble.

## What Stays Fixed

- structured session JSONL representation
- benign-only next-token modeling objective
- token-level adapter-delta extraction
- matched donor/receiver protocol
- matched local day-level comparison protocol
- control-pool fix used in `m04/k04_controlfix`

## What Changes

- base model: `Qwen/Qwen2.5-7B`
- training path: `4x H100 80GB` single-node DDP on Anvil `ai`
- first extraction band: layers `14/20/26`
- first SAE regimes:
  - `latent_mult=2, k=8`
  - `latent_mult=4, k=4`

## Hardware / Parallelization Assumptions

Training:

- `1` Anvil `ai` node
- `4x H100 80GB`
- `64` CPU cores on the Slurm task
- `480G` RAM
- `torchrun` single-node DDP

Recommended training start point:

- `NPROC=4`
- `MICRO_BS=4`
- `GRAD_ACCUM=2`
- effective batch `= 32`
- `GC_MODE=config` first, then test `off` only if memory is clearly comfortable
- `ATTN_IMPL=sdpa` by default; use `flash_attention_2` only if the environment already supports it

Downstream stages:

- token extraction: `1x H100`
- token causal eval: `1x H100`
- bootstrap: CPU-only shared partition

## What Not To Do

- do not reopen the mean-pooled delta path
- do not rerun a broad exploratory frontier first
- do not jump to graph-first escalation before checking whether `7B` strengthens the already-positive token branch

## Success Criterion

`7B` is only a real win if it improves the mechanistic result, not just likelihood fit:

- beat the current best remote token effect (`0.001446`)
- keep positive bootstrap intervals
- widen the margin over the matched local session-AE day-level baseline (`0.001133`)
- ideally stay positive across more than one context/top-set row
