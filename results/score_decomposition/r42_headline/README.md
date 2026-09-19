# Behaviour-only scoring on the r4.2 headline adapter

Anvil job 20827646 (A100, batch 1, 1 h 22 m, ~1.4 SU), 2026-09-18.

**Separate line of evidence from the r6.2 factorial.** This is the paper's
*headline* 8B r4.2 adapter (`qwen3_8b_session_qlora_r42_ddp_mb22_gc_on`); its
scores, population and baselines must not be mixed with the capped-corpus
factorial adapters.

## Population

All days of the 60 malicious users plus all validation-split benign days:
40,519 rows, 139 users (60 positive + 79 benign), 1,309 positive rows. Every
user is excluded from adapter training, so the comparison is user-disjoint on
both sides. No duplicates (unlike the factorial audit pool).

**Sixty malicious users means sixty bootstrap clusters**, against four on r6.2 —
this is much better powered than any CERT r6.2 result in the paper.

## Provenance

Structural checks exact: class sums reconstruct the total to 4.0e-13,
`n_targets = n_tokens − 1` for every row, zero boundary-crossing targets, zero
OTHER/SPECIAL, token counts identical to the cache.

The cached r4.2 scores came from a **batch-56 forward pass**, so a batch-1
rescore does not reproduce them exactly (mean |ΔNLL| 9.96e-03 over all 40,519
rows). Rescoring at batch 56 cuts that disagreement about fourfold, which
establishes a **batching-dependent discrepancy**. It does not isolate padding
specifically: batch shape, kernel selection, padding and batch composition all
change together, and the control did not hold composition fixed while varying
only the mask. Nor does it establish that every score view is equally
insensitive to batching — that would need the views compared across batch
sizes, which was not done. What is checked is the metric level:

| | day ROC | day AP | user ROC | user AP |
|---|---|---|---|---|
| recomputed `full` (batch 1) | 0.6591 | 0.0691 | 0.6525 | 0.5712 |
| cached `full` (batch 56) | 0.6594 | 0.0697 | 0.6517 | 0.5783 |

Agreement is close on the ROC metrics (day 0.0002, user 0.0008) and on day AP
(0.0006), but **not uniform**: pooled user AP differs by 0.0071 (0.5712 vs
0.5783). The anchor holds for the ranking metrics the argument rests on; it is
not exact agreement across the board.

Note on the published figure: `results/valonly_detector/r42.json` reports
user-disjoint day ROC 0.668 / user ROC 0.565. This pool includes the malicious
users' **own benign days** as negatives, which raises user-level ROC (a user's
score is the max over all their days, not over their positive days alone). That
is the population the fold-aligned protocol uses and the operationally natural
one, but it is not identical to the `valonly_detector` construction, so these
numbers are close to rather than a reproduction of that figure.

## Verification status

Independently recomputed **from the saved per-fold CSV** (fold means and
behaviour-versus-full bootstrap contrasts reproduce to ~3.6e-15). The raw
per-example scores, example ids, token counts and reference joins have **not**
been re-verified since the run, because Anvil SSH has been timing out
(`connect to host anvil.rcac.purdue.edu port 22: Operation timed out`,
2026-09-19). That is a failed check, recorded as such; no inference is drawn
about job state or its cause. The original outputs are preserved unchanged.

## Result — pooled user-disjoint

| view | day ROC | day AP | user ROC | user AP |
|---|---|---|---|---|
| full | 0.6591 | 0.0691 | 0.6525 | 0.5712 |
| profile_only | 0.4654 | 0.0286 | 0.4728 | 0.4096 |
| **behavior_only** | **0.8225** | **0.1452** | **0.8633** | **0.7871** |
| behavior_ses_only | 0.8329 | 0.1660 | 0.8527 | 0.7779 |
| psy_only | 0.4729 | 0.0320 | 0.4888 | 0.4363 |
| day_only | 0.4312 | 0.0291 | 0.2174 | 0.3165 |

## Result — fold-aligned, paired cluster bootstrap over the 60 malicious users

behaviour-only versus the full score:

| metric | full | behaviour-only | delta [95 %] | folds improved |
|---|---|---|---|---|
| user ROC | 0.6525 | 0.8633 | **+0.2108 [+0.1572, +0.2663]** | 51/60 |
| day ROC | 0.6028 | 0.7454 | +0.1426 [+0.0658, +0.2235] | 34/60 |
| day AP (fold-average) | 0.0053 | 0.0164 | +0.0111 [+0.0002, +0.0241] | 49/60 |
| within-user ROC | 0.6810 | 0.7647 | **+0.0836 [+0.0457, +0.1249]** | 44/60 |
| held-out rank | 28.45 | 11.80 | −16.65 [−21.03, −12.42] | 51/60 (3 tied, 6 worse) |
| recall @ 0.1 % FPR | 0 | 0.0172 | +0.0172 [+0.0055, +0.0319] | 8/60 |
| recall @ 1 % FPR | 0.0377 | 0.0562 | +0.0184 [−0.0214, +0.0511] | 17/60 |

**These are empirical ROC operating points, not deployment thresholds.** The
threshold at each budget is taken on the *evaluation* negatives themselves, so
it describes a point on the ROC curve computed from the same data it is scored
against. A deployment claim would need a separate calibration cohort, and none
exists here. In absolute terms recall at a 0.1 % false-positive budget is still
only about **1.7 %**, and the change at 1 % has an interval spanning zero.

Every interval excludes zero except recall at a 1 % false-positive budget.

Mean-gap decomposition averaged over the 60 folds (signed weighted
contributions; exact): DAY −0.00851 (negative in 53/60 folds), PSY −0.00238
(negative in 30/60), SESCOUNT −0.00010, SES **+0.05493** (positive in 55/60).
Profile pushes malicious days *down*; behaviour pushes them *up*.

## Why this matters more than the r6.2 numbers

1. **Sixty independent malicious users**, so the intervals are not the
   descriptive four-cluster kind.
2. **r4.2 is the benchmark whose audited mechanism the paper calls
   behavioural** — four of five selected features concentrate on session
   lines — and yet its *score* still carries a large harmful profile
   component. Where the selected features activate and what the score is made
   of are different things.
3. **Within-user ranking improves here** (+0.084, 44/60 folds), unlike r6.2
   where it degraded. With identity held fixed, dropping the profile's direct
   contribution still helps on r4.2.
4. **Average precision and low-budget recall both improve.** Fold-average day
   AP 0.0053 → 0.0164 (×3.1) and pooled day AP 0.0691 → 0.1452 (×2.1); recall
   at a 0.1 % false-positive budget rises from 0 to 0.0172 with an interval
   excluding zero. On r6.2 the AP ratio was actually larger (0.0001 → 0.0010,
   ×10.9) but from a negligible base, and recall at 0.1 % FPR stayed **exactly
   zero**. So the r6.2/r4.2 difference is not "ROC-only versus not" — it is
   that on r4.2 the gain reaches a usable operating point at all.

## Limits

One adapter, one benchmark, exploratory population (these users have been
inspected in earlier phases). A scoring intervention only: the profile text
stays in the context and still conditions every behavioural prediction, so
nothing here shows the adapter has stopped representing identity.

**Do not pair the fold-average AP with the pooled prevalence.** Fold-average
day AP 0.0164 belongs to folds whose mean prevalence is 0.00080 (one malicious
user against 79 benign users). Pooled prevalence over the whole 40,519-row pool
is 1,309/40,519 = 3.23 %, and the corresponding pooled AP is 0.0691 → 0.1452.
An earlier version of this file quoted "day AP 0.0164 at 3.2 % prevalence",
mixing the two populations.

The bootstrap resamples the 60 malicious users only: it does not resample the
79 benign users and says nothing about training-seed variation.
