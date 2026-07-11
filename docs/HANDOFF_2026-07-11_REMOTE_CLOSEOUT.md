# Remote Closeout Plan

Status: after the `8B` necessity runs, the main CERT remote science is closed.
The remaining Anvil-side work is table closeout and one optional `3B`
appendix path.

## What Must Still Be Run On Anvil

Only one remote artifact is still missing for the main paper tables:

1. formal `r6.2` `Qwen3-8B` detector metrics from `example_scores.parquet`

This is bookkeeping, not a new training/casual-eval branch. It should produce:

- `detector_metrics.csv`
- `detector_metrics.json`
- `DETECTOR_METRICS.md`

under:

- `results/qwen3_8b_token_causal/detector_metrics/`

Launch:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r62_detector_metrics_anvil.sh
```

Default score source:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2/example_scores.parquet`

Default run name:

- `qwen3_8b_r62`

## Good-To-Have Detector Row

If we want the remote detector table to show the `3B -> 8B` scale step more
explicitly, also materialize the older `Qwen2.5-3B` detector metrics:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3b_r62_detector_metrics_anvil.sh
```

Default score source:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18/example_scores.parquet`

Default output:

- `results/qwen3b_pilot/detector_metrics/`

This row is useful for the final detector table, but it is not a blocker for
the main `8B` headline claim.

## What Does Not Need To Be Re-Run

Do **not** recreate the Magnolia detector baselines on Anvil just to match
hardware. For the paper, fairness comes from:

- same dataset
- same split / eval protocol
- same metrics

not “all models ran on the same machine.”

Do **not** rerun:

- `r6.2` `8B` causal patching
- `r4.2` `8B` causal patching
- local AE / SVDD / RNN baselines on Anvil
- `31B` scale-up

unless a results bug forces it.

## Optional Appendix Path

If time remains after the detector-metrics closeout, the most useful appendix
remote run is:

- `r6.2` `Qwen2.5-3B` active-control audit on the known positive config
- then `r6.2` `Qwen2.5-3B` necessity

That is an appendix-level scale-consistency check, not a main-paper blocker.
The dedicated appendix launcher is added separately.
