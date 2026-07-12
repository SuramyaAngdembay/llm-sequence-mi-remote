# Fold-Aligned Remote Detector Benchmark

Status: added after the detector-metrics split audit showed that the earlier
remote detector artifacts were not directly comparable to the local detector
tables.

## Why This Exists

The corrected remote detector artifacts now score the full extracted remote eval
pool, but that pool is still:

- `val + positive-user days`

The local detector baselines instead use:

- leave-one-malicious-user-out folds
- with up to `800` benign test users sampled per fold

So the detector table still needed a more faithful remote evaluation path.

## What This Benchmark Does

This path fixes the **evaluation** mismatch, not the full training mismatch.

It does two things:

1. scores the remote `Qwen3-8B` detector on `all.jsonl`
2. reconstructs the same test-user folds used by the local detector baselines
   and computes the same day/user metrics on those folds

The resulting remote detector table is therefore:

- fold-aligned on test users
- fold-aligned on metrics
- still a **fixed benign-trained remote model**

It is **not**:

- per-fold remote retraining

That limitation should be stated explicitly if this table is used in the paper.

## New Files

Scoring:

- `scripts/score_adapter_examples.py`
- `slurm/score_adapter_examples.template.sbatch`

Fold evaluation:

- `scripts/eval_fold_aligned_detector_metrics.py`
- `slurm/eval_fold_aligned_detector_metrics_cpu.template.sbatch`

Wrappers:

- `scripts/submit_qwen3_8b_r62_fold_aligned_detector_anvil.sh`
- `scripts/submit_qwen3_8b_r42_fold_aligned_detector_anvil.sh`

## Fold Protocol

The fold logic matches the local benchmark scripts:

- seed: `42`
- benign test users per fold: `800`
- one held-out positive user per fold

This is the same construction used in the local detector reports.

## Recommended Anvil Runs

For `r6.2`:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r62_fold_aligned_detector_anvil.sh
```

For `r4.2`:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_fold_aligned_detector_anvil.sh
```

## Default Inputs

`r6.2`:

- JSONL:
  - `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl/all.jsonl`
- adapter:
  - `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter`

`r4.2`:

- JSONL:
  - `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42/all.jsonl`
- adapter:
  - `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter`

## Expected Outputs

`r6.2` result dir:

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`

`r4.2` result dir:

- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

Each should contain:

- `fold_aligned_detector_summary.csv`
- `fold_aligned_detector_rows.csv`
- `fold_aligned_test_users.csv`
- `FOLD_ALIGNED_DETECTOR_REPORT.md`

## How To Use This Scientifically

If the fold-aligned detector table looks competitive:

- the detector claim becomes much cleaner

If it looks weak:

- do not force a detector-superiority story
- keep the paper centered on:
  - mechanistic interpretability
  - benchmark-specific causal structure
  - direct-transfer failure + native rediscovery

That is still a valid paper.
