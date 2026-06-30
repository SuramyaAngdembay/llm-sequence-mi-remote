# Qwen3-8B Token Causal Audit Findings

This note records the post-hoc audit that used the rsynced
`token_delta_sae_causal_candidate_rows.csv` files for the strongest `Qwen3-8B`
causal configs.

## Why This Audit Was Needed

The lightweight git bundle was enough to show that `Qwen3-8B` strongly beat the
earlier `3B` token branch and the local session-AE baseline on the headline
causal metric.

But the raw effect for `l18_m04_k04` was unusually large, so the missing
candidate-row CSVs were pulled back from Anvil and inspected locally to check:

- whether control features were actually active
- whether donor/receiver patching looked nontrivial
- whether the strongest result was just a weak-control artifact

## Main Audit Findings

Candidate-row source paths inspected locally:

- `artifacts/anvil_token_causal/qwen3_8b_token_causal/l18_m04_k04/token_delta_sae_causal_candidate_rows.csv`
- `artifacts/anvil_token_causal/qwen3_8b_token_causal/l18_m04_k08/token_delta_sae_causal_candidate_rows.csv`

### `l18_m04_k04`

This config is very strong in the committed causal summaries, but its `control3`
set is inert:

- `control3` mean active receiver tokens:
  - benign: `0.0`
  - anomalous: `0.0`
- `top5` mean active receiver tokens:
  - benign: `3.742857`
  - anomalous: `3.795687`

So `l18_m04_k04` should be treated as a **strong upper-bound result with a
weak-control caveat**, not as the cleanest headline.

### `l18_m04_k08`

This config remains clearly positive while using a nontrivial active control:

- `control3` mean active receiver tokens:
  - benign: `23.185714`
  - anomalous: `23.090806`
- `top5` mean active receiver tokens:
  - benign: `3.014286`
  - anomalous: `3.062883`

And its committed bootstrap result is clean:

- `role / top5`: `0.018769`
- 95% CI: `[0.013444, 0.024649]`

That makes `l18_m04_k08` the **audited headline result**.

### `l18_m02_k04`

This config is negative and acts as an internal failure case, which improves the
credibility of the overall branch:

- best remote day-level advantage: `-0.000203`

## Matched Comparison To Earlier Baselines

Strict matched day-level comparison reports:

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k04/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m02_k04/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Best matched numbers:

- local adaptive session-AE day-level: `0.001133`
- local residual day-level: `0.000654`
- remote `Qwen3-3B` token best: `0.001446`
- remote `Qwen3-8B` audited headline (`l18_m04_k08`): `0.018769`

## Final Read

The right directional conclusion after the audit is:

1. `Qwen3-8B` token-level patching is the strongest mechanistic branch so far.
2. `l18_m04_k08` is the cleanest headline result.
3. `l18_m04_k04` is real but partially inflated by an inert control set.
4. `l18_m02_k04` is a useful negative control configuration.

So the remote token branch should now be advanced using:

- `l18_m04_k08` as the conservative headline result
- `l18_m04_k04` as a supporting upper-bound variant

## Active-Control Follow-Up

The active-control follow-up reran the two strongest configs with
`control5_active`:

- `l18_m04_k04_top5_control5_active`
- `l18_m04_k08_top5_control5_active`

Both completed successfully on Anvil. The best bootstrap rows were:

| config | context | estimate | 95% CI |
|---|---|---:|---:|
| `l18_m04_k04_top5_control5_active` | `project_role` | `0.041975` | `[0.033279, 0.050582]` |
| `l18_m04_k04_top5_control5_active` | `role` | `0.040655` | `[0.032362, 0.049114]` |
| `l18_m04_k08_top5_control5_active` | `role` | `0.018653` | `[0.013396, 0.024391]` |
| `l18_m04_k08_top5_control5_active` | `dept_role` | `0.016378` | `[0.012051, 0.021068]` |

This follow-up preserves the conservative `l18_m04_k08` headline and reduces the
concern that `l18_m04_k04` was only an inert-control artifact.
