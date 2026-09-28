# CERT calendar-label recovery protocol

Declared before corrected metric inspection, 2026-09-28; starting commit `017480c`.
Scope: MI research only. Historical outputs remain unchanged.

## Primary target

Use the official `insiders.csv` incident windows as half-open intervals [start,end),
as in the original CTMC label generator, but intersect them with calendar days.
The session day origin is 2009-12-28 at midnight. Preserve the timestamps' recorded
calendar dates; do not silently convert time zones or add five to label integers.
Repair only the known malformed CDE1846 start, using the earliest matching-user
observable in its official incident file. Any other malformed timestamp fails.
Keep named-user observable-day labels as a separately named sensitivity target;
they are not assumed interchangeable with incident windows or session flags.

Source: official answer archive, published MD5
`c0c96eaa06915e417504f932756f58d8`; record SHA-256 and member hashes too.
Acceptance populations: r4.2 1,362 observed positive days / 70 users;
r6.2 73 / 5. These counts are regression checks, not label-construction rules.

## Preservation and identity

- Work on a separate branch/worktree. Keep original training exposure, serialized
  text, checkpoints, caches and labels. Write corrections to new paths.
- Record checksums of every input copy and every generated label table.
- The stable join is user + calendar date; check example ID and day index against it.
- Refuse ambiguous integer-origin label files, conflicting duplicate score rows,
  unrecognized users/dates, nonfinite scores, and silently incomplete populations.
- Build canonical metadata overlays. Do not retrain the existing main LM adapters.

## Cached reanalysis, declared populations and estimands

Use the shared `eval_metrics_core` for ROC/AP, user maxima, within-user ROC and
operating-point summaries. Report AP with prevalence. Scores rank larger as more
anomalous; report adapted NLL, base NLL and negative delta NLL separately.

1. Main detector: full population, unseen pool (original val + eval), and
   seen-benign comparison pool (original train + eval). Keep input rows identical
   when comparing old and corrected labels on these pools.
2. Fold summaries: reproduce the historical seed-42 fold algorithm for each label
   version, recording user lists; adding a positive user changes the evaluated
   population and may change sampled negative users. This is not called a pure
   label-only effect. New comparisons across model arms use identical key sets
   and the same corrected fold manifest.
3. Masking and factorial arms: compare identical available user-day keys. Remove
   duplicates only after checking metadata, labels, score and target-count
   agreement. Report the duplicate correction separately from label correction.
4. Token-class views: full, profile_only, behavior_only, behavior_ses_only,
   psy_only and day_only. Sum class losses / sum class target counts; validate
   the class partition. Profile remains in input context. Any missing user/day
   is reported; an incomplete cache yields a restricted-cohort result only.
5. Target-mask training comparison: pair behavior-only scores with the existing
   full-training adapter on exactly the same keys.
6. Within-user ROC: equal weight over positive users with both day classes.
7. History-prefix probes: report fixed-sample paired loss contrasts and revised
   class counts. No population detection claim on these selected samples.
8. Time-series caches: preserve original training split; pair all models on their
   common has-history mask. Scores without day IDs require the saved matrix order
   and provenance/reload gates; do not infer row identity from length alone.

Uncertainty for fold-mean contrasts: paired bootstrap of held-out positive users,
10,000 draws, seed 42, same draws for all arms. At five r6.2 users this is descriptive,
conditional on these models and the fixed negative cohort. Undefined metrics stay
undefined; no silent dropping of unfavorable folds. Report all specified views and
contrasts. No thresholds or feature sets are selected to recover historical signs.

## Mechanistic and comparison-model gates

Old fixed-input/state responses can be retained as diagnostics. Corrected
attack/benign conclusions require corrected dictionary fitting, selection,
receiver/donor eligibility, matched controls, and discovery/confirmation exposure.
An old dictionary may be held fixed for a labeled sensitivity analysis; that does
not reproduce the corrected label-sampled or benign-only training method.

Before scaling GPU work: freeze a primary corrected dictionary protocol, run a
small complete intervention pilot, validate fresh baselines / alpha-zero controls /
realized patch coverage, and estimate runtime from actual changed tuples.
Restore retained robustness claims afterward, including negative outcomes.

Classical architecture probes and earlier Deep SVDD/GRU/LSTM/IF and local AE
comparators require their own training/fold repair. Their old summaries do not
become valid when LM scores are relabeled. TWOS/LANL issues remain separately tracked.

## Deliverables

- Checksum-bound labels, master/detail discrepancy report, corrected metadata,
  and original-to-corrected change ledger.
- One old-versus-corrected results registry, population/coverage manifests,
  fold records, and paired comparisons with explicit scope.
- Tests of calendar boundaries, malformed sources, split preservation,
  duplicate conflicts, nonfinite scores, target counts and metric agreement.
- A handoff for independent review before accepting paper claims. Manuscript
  numbers remain pending until the relevant scientific and provenance gates pass.
