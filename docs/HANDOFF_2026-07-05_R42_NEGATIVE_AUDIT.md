# R4.2 Negative Transfer Audit

Status: Magnolia audit after rsync of the full `r4.2` candidate-row CSVs.

## Current Read

The full uncapped `r4.2` token-causal result looks **real and negative**, not
like an obvious artifact of the streamed OOM fix.

Why:

- the streamed uncapped runs completed on all three configs and stayed
  consistently negative
- the local `r4.2` detector baselines are not collapsing
- the remote `r4.2` detector itself still scores well at the day level
- the candidate-row audit shows a real semantic difference from `r6.2`

## Detector-Level Context

Local `r4.2` session baselines on Magnolia:

| method | day PR-AUC | day ROC-AUC | user PR-AUC |
| --- | ---: | ---: | ---: |
| `Deep SVDD` | `0.0337` | `0.7429` | `0.3815` |
| `GRU AE` | `0.0254` | `0.6958` | `0.1244` |
| `LSTM AE` | `0.0236` | `0.7141` | `0.1197` |
| `Isolation Forest` | `0.000254` | `0.7146` | `0.00794` |

Remote `Qwen3-8B` detector metrics computed directly from Anvil
`example_scores.parquet`:

| dataset | score | day PR-AUC | day ROC-AUC |
| --- | --- | ---: | ---: |
| `r6.2` | `adapted_nll` | `0.0835` | `0.6809` |
| `r4.2` | `adapted_nll` | `0.1788` | `0.6756` |

Interpretation:

- `r4.2` did **not** fail because the remote detector became useless
- at the day level, the `r4.2` remote detector still separates positives
  meaningfully and is stronger than the local classical/sequential baselines
- so the transfer failure currently looks more like a **mechanism transfer
  failure** than a pure detector failure

## Candidate-Row Audit

Compared to the positive `r6.2` headline config (`l18_m04_k08`):

- `r6.2` top patches had much larger absolute repair magnitude
  - example: `role/top5`
    - benign: about `-0.0682`
    - anomalous: about `-0.0497`
    - advantage: positive
- `r4.2` top patches are much weaker in absolute magnitude
  - example: `role/top5`
    - benign: about `-0.00471`
    - anomalous: about `-0.00808`
    - advantage: negative
- `r4.2` control patches remain strong
  - example: `role/control3`
    - benign: about `-0.01882`
    - anomalous: about `-0.02000`

So the key difference is:

- in `r4.2`, the chosen top sparse sets do **not** produce strong targeted
  repair
- while the control set still produces large global shifts

This does **not** look like an inert-control issue. It looks like the selected
top sparse features lost specific sufficiency on transfer.

## Why The OOM Rewrite Is Unlikely To Be The Cause

The streamed evaluator changed memory behavior, not the causal protocol:

- same receiver/donor matching
- same per-token sparse patch logic
- same score delta definition
- same bootstrap pipeline

The observed `r4.2` pattern is also structured rather than random:

- all three configs are negative
- stricter contexts (`team`, `role`, `dept_role`) are more negative than `dept`
- top-feature alpha winners shift toward smaller values like `0.25`
- control stays strong while top weakens

That is more consistent with a real semantic mismatch than with a sign bug.

## Remaining Verification Worth Paying For

### 1. R6.2 streamed-evaluator confirmation on Anvil

This is the cleanest way to fully rule out a semantic regression from the OOM
rewrite.

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r6_stream_confirm_anvil.sh
```

Expected outcome:

- if this stays clearly positive and close to the earlier `r6.2` headline,
  then the `r4.2` negative result should be treated as real

### 2. Detector metrics from example scores

Recompute detector metrics reproducibly from any remote run:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote

/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3/bin/python \
  scripts/eval_example_scores_detector_metrics.py \
  --scores-parquet /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on/example_scores.parquet \
  --run-name qwen3_8b_r42 \
  --split eval \
  --out-dir results/qwen3_8b_r42_token_causal/detector_metrics
```

and for `r6.2`:

```bash
/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3/bin/python \
  scripts/eval_example_scores_detector_metrics.py \
  --scores-parquet /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2/example_scores.parquet \
  --run-name qwen3_8b_r62 \
  --split eval \
  --out-dir results/qwen3_8b_token_causal/detector_metrics
```

## What Is Still Missing

There is still no local `r4.2` session-AE mechanistic rerun analogous to the
`r6.2` session `usb_activity` repair branch. Right now the `r4.2` comparison is:

- remote token-causal transfer result
- local `r4.2` detector baselines

not yet:

- remote token-causal vs local `r4.2` session-AE causal repair on the same
  dataset

## Anvil Follow-Up Status

Executed on Anvil at `2026-07-05`.

Housekeeping:

- canceled nine stale `tok_boot_cpu` jobs that were pending with
  `DependencyNeverSatisfied` from earlier failed/canceled causal submissions

Detector metrics:

- materialized `qwen3_8b_r42` detector metrics under
  `results/qwen3_8b_r42_token_causal/detector_metrics/`
- `adapted_nll`: day PR-AUC `0.178786`, day ROC-AUC `0.675574`
- `base_nll`: day PR-AUC `0.152894`, day ROC-AUC `0.662855`
- `neg_delta_nll`: day PR-AUC `0.086712`, day ROC-AUC `0.531068`

R6.2 streamed-evaluator confirmation:

- submitted `scripts/submit_qwen3_8b_r6_stream_confirm_anvil.sh`
- causal job: `18866598`
- bootstrap job: `18866599`
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_stream_confirm/l18_m04_k08_stream_confirm_v1`
- initial queue state: causal pending on `Priority`; bootstrap pending on
  dependency
