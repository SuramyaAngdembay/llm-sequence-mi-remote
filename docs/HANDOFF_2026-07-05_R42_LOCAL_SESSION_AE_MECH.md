# Handoff: R4.2 Local Session-AE Mechanistic Comparator

This note records the Magnolia-side setup for the missing local `r4.2`
session-AE mechanistic comparator.

## Purpose

The current `r4.2` evidence says:

- remote Qwen3-8B detector signal transfers
- remote token-causal mechanism does not transfer
- the remaining missing comparator is local `r4.2` session-AE causal repair on
  the same dataset

This run creates that comparator using the same clean LCDAL session-AE repair
pipeline used for the `r6.2` day-level comparison.

## Magnolia Launcher

Prepared on Magnolia:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/scripts/submit_r42_session_lcdal_mech_pipeline_himem.sh`

Submitted from Magnolia with:

```bash
bash /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/scripts/submit_r42_session_lcdal_mech_pipeline_himem.sh full
```

The launcher submits a dependency chain:

1. train local `r4.2` session LCDAL autoencoders
2. compute counterfactual feature summaries
3. run adaptive feature-bundle repair
4. run residual-baseline repair
5. compare local day-level summaries against all three committed remote `r4.2`
   token-causal configs

## Inputs

Feature table:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_lcdal_session_features_clean/sessionr4.2_features.parquet`

Labels:

- `/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet`

Validation before submission:

- feature rows: `470611`
- unique feature days: `330295`
- label rows: `1883`
- matched positive labeled days in feature table: `1309`
- context columns available: `dept`, `role`, `team`

Context modes used:

- `dept_role,dept,team,role`

Targets used:

- `all_semantic,org_context,psychometric,temporal_context,session_context,activity_summary,logon_activity,usb_activity,file_activity,email_activity,http_activity,host_context`

## Submitted Jobs

Submitted on Magnolia at `2026-07-05 16:30`.

| Job | Name | Role | Dependency | State at setup |
| ---: | --- | --- | --- | --- |
| `570141` | `r42_ses_lcd` | train session LCDAL AE | none | `RUNNING` on `himem008` |
| `570142` | `r42_ses_lcfc` | counterfactual feature summary | afterok:`570141` | `PENDING` |
| `570143` | `r42_ses_lfra` | adaptive repair | afterok:`570142` | `PENDING` |
| `570144` | `r42_ses_lfrr` | residual repair | afterok:`570141` | `PENDING` |
| `570145` | `r42_cmp_l18_m04_k08` | strict comparison | afterok:`570143:570144` | `PENDING` |
| `570146` | `r42_cmp_l18_m04_k04` | strict comparison | afterok:`570143:570144` | `PENDING` |
| `570147` | `r42_cmp_l18_m02_k04` | strict comparison | afterok:`570143:570144` | `PENDING` |

Training log at setup showed the job had loaded the dataset and started
successfully:

- rows: `470611`
- features: `122`
- positive session rows after label merge: `1749`
- folds: `60`
- methods: `plain`, `denoising`

## Output Roots

Training:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_session_lcdal_autoencoder_mech_clean`

Counterfactual feature summary:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_session_lcdal_counterfactual_features_clean`

Adaptive repair:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_session_lcdal_feature_bundle_repair_adaptive_clean`

Residual repair:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_session_lcdal_feature_bundle_repair_residual_clean`

Strict comparison outputs:

- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m04_k08/`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m04_k04/`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m02_k04/`

## Interpretation Plan

When the comparison jobs finish:

- if local `r4.2` session-AE repair is also weak or negative, the `r4.2`
  shift is probably a broader mechanistic shift
- if local `r4.2` session-AE repair is positive while remote token-causal stays
  negative, the failure is more specifically a remote token-mechanism transfer
  failure
