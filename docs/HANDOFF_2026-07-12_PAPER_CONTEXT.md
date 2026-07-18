# Paper Research Context Memo

This memo is for paper-level logical cross-checks, not just code/runtime checks.
It defines what the paper is trying to claim, which artifacts support each
claim, which results are currently invalid or provisional after the July 11-12
bug audit, and what the Anvil side should treat as headline-safe versus
appendix-only.

## Scope

The main paper is currently a **CERT-focused** paper, not a LANL paper.

The intended scientific story is:

- a benign-trained `Qwen3-8B` QLoRA session-language detector
- on insider-threat-style CERT benchmarks
- with mechanistic evidence that is benchmark-specific rather than universally
  transferable

The strongest safe framing is:

- benchmark-specific causal structure
- direct transfer can fail
- native rediscovery can succeed

Not:

- universal insider-threat mechanism
- universal anomaly detector
- strong detector-superiority claim without the fold-aligned detector table

## 1. Paper Claim Map

### Claim A: benign-only remote QLoRA training is valid one-class training

Support:

- `scripts/build_session_jsonl.py`
- `scripts/train_qlora.py`

Read:

- positive users are assigned to `split="eval"`
- QLoRA trains only on `train.jsonl`
- validation uses `val.jsonl`
- positives are not used in remote QLoRA training

Status:

- valid
- the July 11-12 audit did **not** find label leakage into training

### Claim B: the remote `Qwen3-8B` branch shows strong mechanistic structure on `r6.2`

Support:

- `results/qwen3_8b_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_token_necessity/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Current headline branch:

- `layer=18`
- `latent_mult=4`
- `k=8`
- `top5`
- `control5_active`

Read:

- active-control sufficiency is clearly positive
- necessity is clearly positive
- streamed-evaluator confirmation is positive

Status:

- scientifically strong
- final-table-safe on the remote mechanistic side
- use the `same_user_recovery/` bundles, not the older permissive-donor rows

### Claim C: direct remote token-mechanism transfer from `r6.2` to `r4.2` fails

Support:

- `results/qwen3_8b_r42_token_causal/stream_uncapped_v2/RESULTS.md`

Read:

- transferred `r6.2`-style configs on `r4.2` were negative under the full
  uncapped streamed evaluator

Status:

- this is a supporting scientific claim
- it is strong enough to motivate native search
- it is **not** the main headline mechanistic result after the bug audit

### Claim D: `r4.2` has its own native remote token mechanism

Support:

- `docs/HANDOFF_2026-07-05_R42_NATIVE_TOKEN_SEARCH.md`
- `results/qwen3_8b_r42_token_causal/native_search_v3_bs24/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/native_active_control_v1/ACTIVE_CONTROL_RESULTS.md`
- `results/qwen3_8b_r42_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_necessity/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l26_m02_k04_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Current headline branch:

- `layer=26`
- `latent_mult=2`
- `k=4`
- `top5`
- `control5_active`

Read:

- native search found a positive remote token-causal config
- active-control stayed positive
- necessity is weaker/partial, mainly `role` and `dept_role`
- local `r4.2` mechanistic comparator is positive on the same receiver set

Status:

- scientifically meaningful
- final-table-safe on the remote mechanistic side
- use the `same_user_recovery/` bundles, not the older permissive-donor rows

### Claim E: the remote detector is competitive with local detector baselines

Intended support:

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`
- Magnolia local detector reports:
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/reports/R62_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md`
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/reports/R42_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md`

Status:

- scientifically usable only through the fold-aligned detector artifacts
- detector quality is mixed / weak on day PR-AUC relative to the stronger local
  Magnolia baselines
- the old `results/*/detector_metrics/DETECTOR_METRICS.md` artifacts are
  audit-only and are **not** headline detector rows

### Claims that are now invalid or unsafe

Do **not** make these claims:

- the remote detector is competitive based on the old `detector_metrics/`
  artifacts
- a strong detector-superiority claim on CERT
- direct token-mechanism transfer succeeds across CERT benchmarks
- the same sparse mechanism transfers unchanged across datasets
- `r4.2` necessity is equally strong as `r6.2`
- old `REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md` files are citation-safe without
  regeneration after `ee6a75a`

## 2. Dataset / Protocol Definitions

### Remote data splits

Defined in:

- `scripts/build_session_jsonl.py`

Split rule:

- if `user_id` is positive, assign `split="eval"`
- otherwise hash user to `val` with `val_frac=0.10`, else `train`

Generated files:

- `all.jsonl` = all user-days
- `train.jsonl` = benign train users only
- `val.jsonl` = benign validation users only
- `eval.jsonl` = `val + eval`

Important:

- `eval.jsonl` is **not** pure positives
- it intentionally contains benign validation users plus positive eval users

### Intended remote training population

Defined in:

- `scripts/train_qlora.py`

Training:

- `train.jsonl` only

Validation / early stopping:

- `val.jsonl` only

Positives in training:

- none

### Intended detector benchmark population

Final paper detector table should use:

- full-population scoring on `all.jsonl`
- then fold-aligned evaluation matching the local detector benchmark

Defined in:

- `scripts/score_adapter_examples.py`
- `scripts/eval_fold_aligned_detector_metrics.py`
- `docs/HANDOFF_2026-07-11_FOLD_ALIGNED_REMOTE_DETECTOR.md`

Fold rules:

- seed `42`
- one held-out malicious user per fold
- `800` benign test users per fold

Scoring:

- fixed benign-trained remote model
- not per-fold retraining

Metrics:

- computed on the same fold test users as the local detector baselines
- day-level metrics on all user-day rows in those test users
- user-level metrics after max-aggregation over days per user

### What went wrong in the old detector artifacts

Documented in:

- `docs/HANDOFF_2026-07-11_DETECTOR_METRICS_AUDIT.md`

The old `detector_metrics/` path:

- used `example_scores.parquet` extracted from `eval.jsonl`
- then applied `--split eval` again

That incorrectly dropped the benign validation slice and overstated detector
quality, especially on `r6.2`.

### Remote mechanistic evaluation population

For causal patching:

- receivers = positive eval examples only

For necessity:

- positive eval receivers matched to benign receivers by context

Current dataset sizes reflected in committed remote artifacts:

- `r6.2` remote positive receiver-days / examples: `70`
- `r4.2` remote positive receiver-days / examples: `1309`

## 3. Mechanistic Estimands

### Causal patching estimand

Defined in:

- `scripts/eval_token_delta_sae_causal.py`

The causal estimator asks:

- if we patch selected sparse token-SAE features in a positive receiver toward a
  matched donor prototype, does anomaly evidence change more for the selected
  top features than for a matched control feature set?

Protocol:

- receivers = positive eval examples only
- donors = matched benign donors and same-class anomalous donor controls
- patch only token positions where the selected sparse features are active
- patch toward donor token-feature prototypes in delta-SAE space

Headline summary terms:

- `top_repair_advantage = top_anomalous_mean_best_delta - top_benign_mean_best_delta`
- `control_repair_advantage = control_anomalous_mean_best_delta - control_benign_mean_best_delta`
- `top_minus_control_advantage = top_repair_advantage - control_repair_advantage`

Interpretation:

- positive `top_minus_control_advantage` means the selected top sparse features
  produce stronger anomaly-specific repair than the control features

### Necessity estimand

Defined in:

- `scripts/eval_token_delta_sae_necessity.py`
- `docs/HANDOFF_2026-07-09_REMOTE_TOKEN_NECESSITY.md`

The necessity estimator asks:

- if we ablate selected sparse token features, does anomaly evidence weaken more
  for positive receivers than for matched benign receivers, and more than for
  an active control feature set?

Headline summary terms:

- `top_necessity_advantage = top_benign_mean_best_delta - top_positive_mean_best_delta`
- `control_necessity_advantage = control_benign_mean_best_delta - control_positive_mean_best_delta`
- `top_minus_control_necessity = top_necessity_advantage - control_necessity_advantage`

Interpretation:

- positive `top_minus_control_necessity` means the selected top features are
  more necessary than the control features

### What “active control” means

Active control was introduced because some earlier `control3` comparisons were
weak or inert.

Headline-safe control:

- `control5_active`

How it is defined:

- a control feature set chosen from SAE features with nontrivial activation
  support
- using `active_control_min_frac=0.002`

Control policy for the paper:

- use `control5_active` for final headline mechanistic rows
- treat old `control3` bundles as exploratory / historical unless they are only
  being used to confirm evaluator behavior

Important example:

- `r6.2` `l18_m04_k04` with `control3` is **not** a clean headline row because
  the control was inert
- the active-control rerun preserved the effect, so the active-control bundle is
  the valid one

### Same-user donor / receiver exclusion

Current final-rule policy:

- same-user donor exclusion is required for the final headline causal results
- same-user benign-match exclusion is required for the final headline necessity
  results

Why:

- the July 12 validity audit found nontrivial same-user matching rates in the
  existing headline bundles
- that can inflate repair by borrowing the user’s own benign style

Current status:

- the no-same-user rerun path is implemented
- the reruns are part of the required recovery plan

## 4. Final-Table Requirements

### Headline-safe versus audit-only artifacts

Headline-safe detector rows:

- only the outputs under:
  - `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
  - `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

Audit-only / not headline detector rows:

- `results/qwen3_8b_token_causal/detector_metrics/`
- `results/qwen3_8b_r42_token_causal/detector_metrics/`

Headline-safe mechanistic rows:

- `r6.2` remote:
  - `results/qwen3_8b_token_causal/same_user_recovery/`
  - `results/qwen3_8b_token_necessity/same_user_recovery/`
- `r4.2` remote:
  - `results/qwen3_8b_r42_token_causal/same_user_recovery/`
  - `results/qwen3_8b_r42_token_necessity/same_user_recovery/`

Supporting / appendix / historical context:

- `results/qwen3_8b_token_causal/CAUSAL_EVAL_RESULTS.md`
- `results/qwen3_8b_token_causal/stream_confirm/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/stream_uncapped_v2/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/native_search_v3_bs24/RESULTS.md`
- all `Qwen3B` results under `results/qwen3b_pilot/` or `docs/HANDOFF_2026-07-11_QWEN3B_APPENDIX.md`

### Compare reports

Any cited `REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md` should come from the regenerated
post-`ee6a75a` reports.

Headline-safe regenerated reports:

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l26_m02_k04_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Historical compare reports are appendix / audit-only unless explicitly
re-regenerated:

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_*`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_*`
- `results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix/*`

### Hardware requirements

Remote Qwen runs:

- must run on Anvil, not Magnolia
- reason: live checkpoints, token-delta caches, and validated runtime live on
  Anvil project storage

Magnolia-only work:

- local session baselines
- local session-AE mechanistic comparator
- local compare/report scripts

H100 versus A100:

- H100 is **not** scientifically required for final tables
- A100 is acceptable if the same checkpoint, code, and protocol are used and
  the run completes successfully
- use H100/AI only when required for runtime or storage locality

Current validated hardware:

- `r6.2` initial token causal work used Anvil AI/H100-side runtime
- `r4.2` native active-control and necessity were validated on Anvil
  `gpu`/A100
- fold-aligned detector scoring was explicitly optimized for Anvil
  `gpu`/A100

### Which comparisons require identical hardware

Identical hardware is **not** required for:

- remote Qwen detector versus Magnolia local baselines
- remote Qwen mechanistic rows versus Magnolia local mechanistic comparator

What must be identical:

- dataset
- fold construction
- score definition
- aggregation unit
- metric definition
- control / target protocol

## 5. Recovery Plan

### Must-rerun jobs

These are the minimum required recovery jobs.

1. Finish fold-aligned detector benchmark:
   - `r6.2` fold-aligned remote detector
   - `r4.2` fold-aligned remote detector
2. Run same-user-excluded `8B` robustness reruns:
   - `r6.2` active-control causal
   - `r6.2` necessity
   - `r4.2` native active-control causal
   - `r4.2` native necessity
3. Regenerate cited local-vs-remote compare reports

Current handoff:

- `docs/HANDOFF_2026-07-12_CERT_RECOVERY.md`

### Optional robustness jobs

Only after the must-rerun jobs are stable:

- `Qwen3B` appendix cleanup
- extra same-user-excluded supporting reruns on non-headline transfer configs
- LANL external validation

Do **not** spend large SU on these before the CERT recovery closes cleanly.

### Success / failure criteria before spending more SU

Success:

- fold-aligned detector table is complete and scientifically usable
- same-user-excluded `r6.2` headline mechanistic runs remain clearly positive
- same-user-excluded `r4.2` native active-control remains positive
- same-user-excluded `r4.2` necessity remains at least weak/partial in the same
  direction, or else the paper text is adjusted honestly

If that happens:

- freeze the CERT story
- stop expanding scope
- write

Failure / downgrade triggers:

- fold-aligned detector table looks weak relative to Magnolia baselines
  - then soften the detector-competitiveness claim
- same-user-excluded mechanistic reruns flip sign broadly
  - then downgrade the mechanistic claim and do not broaden scope until the
    cause is understood

## Bottom Line For Anvil Cross-Checks

When reviewing any future run, ask:

1. Is this run targeting the final paper estimand, or only a convenience proxy?
2. Is it using the correct evaluation population?
3. Is it using `control5_active` if it is a headline mechanistic row?
4. Is same-user exclusion enforced if the row is intended for the final paper?
5. Is this run replacing a damaged artifact, or only adding scope?

If the answer to (5) is “only adding scope” before the detector recovery and
same-user-exclusion recovery are done, reject the run as scientifically
misaligned with the current paper needs.
