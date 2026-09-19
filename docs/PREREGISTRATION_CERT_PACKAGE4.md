# Pre-registration: token-class decomposition of SAE interventions on CERT r4.2

Written **2026-09-19**, while Anvil is unreachable and this run is therefore
impossible to have performed. Committed before any CERT per-class intervention
data exists. Motivated by the post-hoc-selection findings in
`docs/AI_SCIENTIST_FAILURE_MODE_AUDIT.md`.

If anything below is changed after the run, the change and its reason must be
recorded in this file with the date, and the original text left visible.

## Question

The published r4.2 causal result reports that patching the top delta-SAE
features lowers the adapted-NLL anomaly score more than an activity-matched
control set does. That score is a mean over all scored tokens. **Does the repair
act on profile tokens, on behaviour tokens, or both?**

## Primary comparison, declared

`delta_profile_only` against `delta_behavior_only`, for `top5` **minus**
`control5_active`, aggregated per receiver user, at **α = 1.0**.

This is the primary test. `day_only` and `psy_only` are decompositions of the
profile view and are **secondary**; they may explain a primary result but may
not stand in for one. `full` is reported as the quantity the published pooled
metric corresponds to.

## Prediction

Given package 3 (the profile acts as an identity key carrying no behavioural
predictive value) and the TWOS result (repairs land on identity tokens with a
dose-response while behaviour stays flat), the prediction is:

* `profile_only` contrast **negative**, interval excluding zero;
* `behavior_only` contrast at least an order of magnitude smaller in
  magnitude, with an interval spanning zero;
* monotone growth of the profile contrast across α = 0.25, 0.5, 0.75, 1.0.

**A result contradicting this is reportable as-is.** In particular, if the
behaviour contrast is also clearly negative, the conclusion is that the features
are not purely an identity shortcut, and the paper's mechanistic section must
say so.

## Fixed before the run

| item | value |
|---|---|
| adapter | `qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter` |
| SAE frontier | `token_delta_sae_frontier_r42_discovery_split` |
| layer / latent_mult / k | 26 / 2 / 4 |
| feature sets | `top5` against `control5_active`, `active_control_min_frac` 0.002 |
| alphas | 0.25, 0.5, 0.75, 1.0 — **all four reported** |
| context modes | team, role, dept, dept_role |
| receivers | the r4.2 confirmation-user split; `exclude_same_user_donors` on |
| batching | `batch_size` 12, `loss_batch_size` 4, `patch_chunk_size` 8 |
| token-class views | frozen 2026-09-17 in commit `007ca54`, unmodified |
| clustering unit | receiver user |
| bootstrap | 5,000 draws, seed 42, percentile interval |

Every value except `--token-class-schema cert` matches the run that produced
job 19379904, so the new columns describe the **same interventions** as the
published result.

## Rules fixed before the run

1. **The base score is recomputed** through the identical code path and batch
   composition as the patched score. The cached `adapted_nll` is **not** used
   for deltas. Justification, measured on TWOS: cache minus recomputation has
   mean absolute difference 1.045e-02 nats, larger than every effect measured.
   The discrepancy is reported.
2. **The control set is the comparison.** A delta against zero is not a result.
3. **All six views are reported** whatever they show.
4. **No view may be promoted to headline after the run.** The primary
   comparison is the one named above. A larger secondary effect may be
   discussed as explanation, labelled secondary.
5. **Per-batch assertions must pass**: per-class sums reconstruct the scalar
   NLL, and patched per-class target counts equal the base's. If either fails
   the run is void, not adjusted.
6. **Smoke first** on `gpu-debug` with `--max-receivers 24`, one alpha, one
   context mode, before the full job.
7. **Adapter provenance must be verified** before these numbers enter the
   paper: the md5 of the Anvil adapter must be compared against the Aquaman
   copies recorded in V34.

## What would make this result uninterpretable

* Per-class target counts differing between base and patched conditions.
* A smoke run whose per-class sums fail to reconstruct the scalar score.
* Fewer than 8 distinct receiver users surviving, which would make the
  clustered interval meaningless.
