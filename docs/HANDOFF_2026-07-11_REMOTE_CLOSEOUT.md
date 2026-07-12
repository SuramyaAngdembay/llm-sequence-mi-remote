# Remote Closeout Plan

Status: after the `8B` necessity runs, the main CERT remote science is closed.
The remaining Anvil-side work is table closeout and one optional `3B`
appendix path.

## Current Must-Do Status

The formal `r6.2` `Qwen3-8B` detector-metrics artifact has now been materialized,
but it required an audit correction.

Current detector artifact path:

- `results/qwen3_8b_token_causal/detector_metrics/`

Current detector audit note:

- `docs/HANDOFF_2026-07-11_DETECTOR_METRICS_AUDIT.md`

So the remaining remote-side task is no longer “run detector metrics once.” It is:

1. use the corrected detector artifacts
2. do not rely on the earlier `eval`-only detector read
3. run the fold-aligned remote detector benchmark if the final paper needs a fair remote-vs-local detector table

Fold-aligned benchmark handoff:

- `docs/HANDOFF_2026-07-11_FOLD_ALIGNED_REMOTE_DETECTOR.md`

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

## Detector Metrics Correction

After the first detector-metrics commit, we found that the metric script had
been run with:

- `--split eval`

even though the extracted `example_scores.parquet` already represents the
remote `eval.jsonl` pool and contains both:

- benign validation-user days (`split="val"`)
- positive-user days (`split="eval"`)

So the corrected detector-metric launchers now default to:

- no extra split filter

See:

- `docs/HANDOFF_2026-07-11_DETECTOR_METRICS_AUDIT.md`

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

See:

- `docs/HANDOFF_2026-07-11_QWEN3B_APPENDIX.md`
