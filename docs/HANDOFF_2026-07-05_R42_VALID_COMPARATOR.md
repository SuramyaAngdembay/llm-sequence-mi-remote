# R4.2 Valid Comparator Handoff

Status: this note defines the **missing local comparator** for the `r4.2`
remote token-causal transfer result.

## What Is Missing

We now have:

- remote `r4.2` `Qwen3-8B` detector metrics
- remote `r4.2` full uncapped token-causal results
- local `r4.2` detector baselines:
  - `Deep SVDD`
  - `GRU AE`
  - `LSTM AE`
  - `Isolation Forest`

We do **not** yet have:

- local `r4.2` **session-AE mechanistic repair** results analogous to the
  `r6.2` session `usb_activity` branch

That is the missing comparator needed to answer:

- did only the **remote** causal mechanism fail to transfer?
- or did the **local session-AE mechanism** also weaken / change on `r4.2`?

## What Counts As A Valid Comparator

The comparator must be the local `r4.2` **session LC-DAL autoencoder**
mechanistic pipeline, not just another detector table.

Minimum required pieces:

1. `r4.2` session LC-DAL autoencoder training
2. latent family ablation on that trained model
3. counterfactual feature extraction
4. adaptive feature-bundle repair
5. residual baseline repair
6. stats aggregation from:
   - family ablation raw rows
   - adaptive repair best rows
   - residual repair best rows

This should mirror the existing `r6.2` local session mainline as closely as
possible.

## Why Simpler Comparators Are Not Enough

These are **not** sufficient:

- remote `r4.2` causal vs local `r4.2` detector-only baselines
- remote `r4.2` causal vs local `r6.2` session-AE repair
- remote `r4.2` causal vs remote `r4.2` detector metrics alone

Why:

- the claim we need to evaluate is about **mechanistic transfer**
- detector transfer and mechanism transfer can diverge
- that is already what the current evidence suggests

## Existing Magnolia Inputs

These already exist locally on Magnolia:

- `r4.2` session feature parquet:
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r42_lcdal_session_features_clean/sessionr4.2_features.parquet`
- `r4.2` labels:
  - `/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet`

The local `r4.2` detector baseline report already uses the same feature parquet:

- [R42_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md](/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/reports/R42_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md)

So the feature-prep blocker is already cleared.

## R6.2 Scripts To Mirror

These are the known-good `r6.2` local session-mech launchers to copy/retarget:

- autoencoder train:
  - `submit_r62_session_lcdal_autoencoder_mech_himem.sh`
- family ablation:
  - `submit_r62_session_lcdal_latent_family_ablation_himem.sh`
- counterfactual features:
  - `submit_r62_session_lcdal_counterfactual_features_clean_himem.sh`
- adaptive repair:
  - `submit_r62_session_lcdal_feature_bundle_repair_adaptive_clean_himem.sh`
- residual repair:
  - `submit_r62_session_lcdal_feature_bundle_repair_residual_clean_himem.sh`
- adaptive stats:
  - `submit_r62_session_lcdal_mech_stats_adaptive_clean_himem.sh`
- residual stats:
  - `submit_r62_session_lcdal_mech_stats_residual_clean_himem.sh`

All live in:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/scripts/`

## Recommended R4.2 Launcher Mapping

The clean way is to create direct `r4.2` counterparts, not modify the `r6.2`
files in place.

Suggested names:

- `submit_r42_session_lcdal_autoencoder_mech_himem.sh`
- `submit_r42_session_lcdal_latent_family_ablation_himem.sh`
- `submit_r42_session_lcdal_counterfactual_features_clean_himem.sh`
- `submit_r42_session_lcdal_feature_bundle_repair_adaptive_clean_himem.sh`
- `submit_r42_session_lcdal_feature_bundle_repair_residual_clean_himem.sh`
- `submit_r42_session_lcdal_mech_stats_adaptive_clean_himem.sh`
- `submit_r42_session_lcdal_mech_stats_residual_clean_himem.sh`

Each should switch only the dataset-specific paths and output roots:

- feature path:
  - `results_r42_lcdal_session_features_clean/sessionr4.2_features.parquet`
- label path:
  - `/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet`
- run root:
  - `results_r42_session_lcdal_autoencoder_mech_clean`
- family ablation out:
  - `results_r42_session_lcdal_latent_family_ablation_clean`
- counterfactual out:
  - `results_r42_session_lcdal_counterfactual_features_clean`
- adaptive repair out:
  - `results_r42_session_lcdal_feature_bundle_repair_adaptive_clean`
- residual repair out:
  - `results_r42_session_lcdal_feature_bundle_repair_residual_clean`
- adaptive stats out:
  - `results_r42_session_lcdal_mech_stats_adaptive_clean`
- residual stats out:
  - `results_r42_session_lcdal_mech_stats_residual_clean`

## Protocol Constraints To Keep Fixed

To make the comparator valid, keep these aligned with the `r6.2` session mainline:

- same one-class training regime
- same leave-one-malicious-user-out fold logic
- same session feature family
- same context modes for repair:
  - `dept_role,project_role,team,role`
- same intervention families and targets
- same adaptive-vs-residual split

Do **not** simplify the comparator to:

- just family ablation
- just residual repair
- or just the local detector score table

## What We Need Back

Minimum outputs to commit or sync back:

- `results_r42_session_lcdal_autoencoder_mech_clean/` summary report
- `results_r42_session_lcdal_latent_family_ablation_clean/latent_family_ablation_raw.csv`
- `results_r42_session_lcdal_counterfactual_features_clean/counterfactual_feature_summary.csv`
- `results_r42_session_lcdal_feature_bundle_repair_adaptive_clean/adaptive_feature_bundle_repair_best_rows.csv`
- `results_r42_session_lcdal_feature_bundle_repair_residual_clean/residual_feature_bundle_repair_best_rows.csv`
- `results_r42_session_lcdal_mech_stats_adaptive_clean/R42_*_REPORT.md` or equivalent final report
- `results_r42_session_lcdal_mech_stats_residual_clean/R42_*_REPORT.md` or equivalent final report

If the local script family still emits the generic filename:

- `R62_DAILY_AUTOENCODER_MECH_STATS_REPORT.md`

that is acceptable temporarily, but the output directory must still be the
`r42` one.

## Decision Rule Once The Comparator Exists

After these local `r4.2` mechanistic runs complete, compare:

1. remote `r4.2` token-causal result
2. local `r4.2` adaptive session-AE repair
3. local `r4.2` residual baseline repair

Interpretation cases:

- if local `r4.2` adaptive session-AE is also weak/negative:
  - the mechanism may genuinely shift on `r4.2`
- if local `r4.2` adaptive session-AE stays positive while remote is negative:
  - the transfer failure is more specifically a **remote token-mechanism failure**
- if both are positive but remote is weaker:
  - the mechanism transfers partially but the remote sparse token basis is less aligned

## Current Best Remote Context

Remote `r4.2` currently looks like:

- detector transfer: **yes**
  - `adapted_nll` day PR-AUC `0.178786`
- token-causal transfer: **no**
  - full uncapped token-causal suite negative on all three configs

Remote `r6.2` streamed confirmation:

- still positive under the current streamed evaluator
- so the `r4.2` negative result is unlikely to be a memory-rewrite artifact

That means the local `r4.2` mechanistic comparator is now the highest-value
remaining experiment.
