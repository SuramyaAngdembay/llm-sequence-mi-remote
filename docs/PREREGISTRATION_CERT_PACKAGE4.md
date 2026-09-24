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
This file was added in commit `c0131c4`. That commit, against the job record of the
eventual run, is the evidence that it predates it -- and a commit hash proves
only that an artifact was recorded at a time, not that anyone was bound by it.

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

(Same evidence: commit `c0131c4`, with the same caveat.)

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

## Addendum, 2026-09-24 — clarifications recorded before any per-class result exists

Written while jobs 20879548 (smoke) and 20879549 (full) are still PENDING on
the collaborator allocation. No per-class CERT intervention output from the
collaborator environment exists. The original text above is unchanged.

**What this addendum is not.** These clarifications are fixed before the
per-class outcomes, but not independently of every outcome: the published
pooled endpoint (0.000758, dept) and its per-context pattern are known, and so
is the (invalid, see below) TWOS per-class run. The analysis below is
therefore pre-specified with respect to the per-class CERT outcomes only. It
must not be described as independently confirmatory.

**Correction to the Question and Prediction sections.**
- The published endpoint is not "patched top features lower the score more
  than an activity-matched control". It is the best-candidate donor
  difference-in-differences
  `(δ_top,anomalous − δ_top,benign) − (δ_control,anomalous − δ_control,benign)`,
  over receivers with complete support (`eval_token_delta_sae_causal.py`,
  `summarize_best`). The controls meet a 0.2% activity threshold and a
  lowest-|gap| rule; they are not activity-matched. On the staged ranking
  they fire about twice as often as the selected features (mean active
  fraction 0.0036 versus 0.0018).
- The Prediction cites the TWOS result. The TWOS token-class runs patched
  features `[0, 1, 2, 3, 4]` selected from an undefined ranking (all gaps
  NaN; see `docs/TWOS_CORRECTED_INTERVENTION_PLAN.md`). That motivation is
  void. The prediction text stays as written, and a result matching it may
  not be presented as support for the TWOS-based reasoning.

**Choices the original left open, now fixed:**

1. *Donor type.* Benign and anomalous donors are analysed separately, never
   pooled. The primary per-class comparison uses **benign donors** (moving the
   selected features toward a benign colleague). Anomalous-donor results are
   reported beside it. The published difference-in-differences is recomputed
   from the new rows for continuity, labelled as the historical endpoint.
2. *Context modes.* Each of team, role, dept and dept_role is reported
   separately. None is primary. dept is not privileged, although it was the
   only mode whose published cluster interval excluded zero, because that is
   an outcome. Any pooled-across-mode figure is descriptive only; the modes
   share receivers and are not independent.
3. *Unit, weighting and complete cases.* Receiver = mean over all candidate
   donors of the given type at the given alpha (not the best candidate).
   Complete case = receivers present in both arms for that mode and donor
   type; excluded receivers are counted and reported. Primary aggregate = mean
   of receiver-user means, 95% cluster bootstrap over users (the original
   5,000-draw specification; `analyze_intervention_endpoints.py` uses 10,000,
   and both are reported if they differ in the third significant figure).
4. *Alpha.* α = 1.0 is primary, as declared. The other alphas are reported per
   alpha; no best-over-alpha quantity enters the primary comparison.
5. *The direct contrast.* The primary quantity is the per-receiver paired
   difference `(sel − ctrl)_profile_only − (sel − ctrl)_behavior_only`, with its
   own cluster interval. A significant profile contrast beside a
   non-significant behaviour contrast is not evidence that they differ.
6. *Negligible effects.* No "negligible behaviour effect" claim will be made
   unless the 90% cluster interval of the behaviour contrast lies inside
   ±0.0002 nats per token (TOST), a margin of about a quarter of the published
   pooled endpoint. An interval crossing zero is otherwise inconclusive.
7. *Absolute effects.* Every table reports the unpatched base (fresh
   recomputation), the selected arm's delta, the control arm's delta and their
   difference. A near-zero difference means the two patches act similarly,
   not that neither acts.
8. *Reconstruction.* When any token of a receiver day has an active feature of
   an arm, every token of that day is replaced by its dictionary
   reconstruction and only the active tokens are edited. The queued run has no
   α = 0 arm, so reconstruction effects are not separated from feature edits.
   Absolute per-arm effects are therefore not interpreted mechanistically until
   a bounded α = 0 run on the same receivers exists; the selected-minus-control
   difference is reported with this caveat.
9. *Environment.* The deltas and SAE came from the owner environment; the
   queued runs use the collaborator environment. Computing both arms in one run
   does not guarantee cancellation. Gate: the collaborator smoke (20879548) is
   compared row by row with the owner-environment smoke (job 20840686,
   `outputs/token_class_causal_r42_SMOKE/l26_m02_k04`) on matching receiver,
   donor, arm and alpha keys (`scripts/compare_intervention_runs.py`). If the
   matched selected-minus-control contrast differs by more than 1e-4 nats
   between environments, or any per-row delta by more than 1e-3, the full run
   is not interpreted until the difference is explained.
10. *Baselines.* In runs with `--token-class-schema`, `delta`, `repair`,
    `strong_repair`, the best rows and the summaries all use the fresh base
    (verified in code, `base_used`). Older outputs used the cached base for
    those fields: job 19379904, the TWOS v3 runs, and the 2026-09-19 TWOS
    token-class runs, which predate the switch (their per-class `delta_<view>`
    columns do use the fresh base; their `delta` and repair flags do not). A
    difference-in-differences cancels a per-receiver constant base, but the
    absolute deltas and repair flags of those outputs are not comparable with
    fresh-base runs.
