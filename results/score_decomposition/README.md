# Token-class score decomposition — behaviour-only scoring on frozen adapters

Phase A/B of the investigation specified in `docs/SCORE_DECOMPOSITION_MANIFEST.md`.
Produced 2026-09-18 by Anvil jobs 20814766 (8B, H100) and 20814767 (3B, A100).

**Scope.** These are the *capped-corpus train-matched factorial* adapters
(`adapter_full`), not the paper's headline mechanistic adapters. All views are
scored on identical examples with the same frozen adapter and the **full input
context**: the profile text is present for every view. A behaviour-only view
removes the profile's *direct contribution to the score*; it does not remove
the profile's influence on behavioural predictions, and it is not identity
removal.

**Population.** 159,064 day-rows, 410 users, 70 positive rows. Four positive
users (ACM2278 5/176 positive days, CDE1846 46/356, CMP2946 18/316, MBG3183
1/356) and 406 benign (405 validation-split + PLJ1771). Every user was excluded
from adapter training, so this is a user-disjoint comparison, **not** a
seen-user detection test. Folds come from
`eval_fold_aligned_detector_metrics.make_folds` (seed 42), giving 407 test
users per fold and the same benign cohort in all four folds. Day-level
prevalence per fold: 0.000032 (5 of 158,036 rows).

**Provenance.** The recomputed `full` view reproduces the published
`adapted_nll` over all 159,064 rows to max |ΔNLL| = 1.4e-07 (mean 2.0e-08,
rank corr 0.9999999999977), after scoring at batch 1 on the hardware that
produced each cached baseline. Class sums reconstruct the total to 4e-13;
`n_targets = n_tokens − 1` for every example; zero boundary-crossing targets;
zero OTHER/SPECIAL. The mean-gap decomposition reconstructs the total gap to
5e-17.

## Headline

Mean over the four folds. `full` is the published detector score.

### 8B (Qwen3-8B)

| view | day ROC | day AP | user ROC | held-out rank /407 | within-user ROC |
|---|---|---|---|---|---|
| full | 0.399 | 0.0001 | 0.532 | 191.0 | 0.751 |
| profile_only | 0.213 | 0.0001 | 0.200 | 326.0 | 0.588 |
| **behavior_only** | **0.836** | **0.0009** | **0.938** | **26.3** | 0.680 |
| behavior_ses_only | 0.846 | 0.0008 | 0.937 | 26.5 | 0.696 |
| day_only | 0.410 | 0.0001 | 0.319 | 277.5 | 0.570 |
| psy_only | 0.208 | 0.0001 | 0.196 | 327.5 | 0.623 |

Paired cluster bootstrap over the four malicious users (descriptive at n=4):
user ROC **+0.406 [+0.163, +0.681]**, day ROC +0.437 [+0.278, +0.562], held-out
rank −164.8 [−276.5, −66.3]. All four folds improve on user ROC
(+0.071, +0.805, +0.438, +0.308).

Per held-out user, user ROC (rank of 407):

| view | ACM2278 | CDE1846 | CMP2946 | MBG3183 |
|---|---|---|---|---|
| full | 0.919 (34) | 0.155 (344) | 0.527 (193) | 0.527 (193) |
| profile_only | 0.414 (239) | 0.017 (400) | 0.096 (368) | 0.271 (297) |
| behavior_only | 0.990 (5) | 0.961 (17) | 0.966 (15) | 0.835 (68) |

### 3B (Qwen2.5-3B)

| view | day ROC | day AP | user ROC | held-out rank /407 | within-user ROC |
|---|---|---|---|---|---|
| full | 0.741 | 0.0003 | 0.926 | 31.0 | 0.739 |
| profile_only | 0.671 | 0.0002 | 0.649 | 143.5 | 0.650 |
| behavior_only | 0.767 | 0.0046 | 0.944 | 23.8 | 0.568 |
| behavior_ses_only | 0.782 | 0.0040 | 0.945 | 23.3 | 0.569 |
| day_only | 0.402 | 0.0001 | 0.314 | 279.5 | 0.470 |
| psy_only | 0.763 | 0.0003 | 0.743 | 105.5 | 0.751 |

User ROC +0.018 [−0.086, +0.123] — interval spans zero; three folds improve,
MBG3183 gets worse (0.941 → 0.815). The `full` adapter starts at 0.926, so
there is little headroom here.

## What must be reported with it

- **Average precision stays near the base rate.** 8B day AP 0.0001 → 0.0009 at
  prevalence 0.000032. A large relative gain, but far too low to be a usable
  alert stream. ROC improved enormously; alert precision did not.
- **At a tight alert budget there is no gain.** Day recall at 0.1 % FPR is
  0.000 for every view and every fold at 8B. At 1 % FPR the gain is
  +0.044 [0.000, +0.088], from two of four folds.
- **Within-user ranking gets worse**, not better: 8B 0.751 → 0.680
  (−0.071 [−0.284, +0.049], one fold −0.389); 3B 0.739 → 0.568. The profile
  contribution *helps* when identity is held fixed. One untested explanation:
  within a user the profile text is constant, so its share of the mean varies
  only through N_profile/N_total, which tracks the number of session lines that
  day — the full score may partly encode day length. Not established.
- **Signed contributions cancel.** In the mean-gap decomposition a class share
  can exceed 100 % or flip sign (e.g. CMP2946 at 8B: PSY +382 %, SES −406 %).
  It is an accounting identity for the *mean* gap, not an AUC decomposition and
  not a causal mechanism.
- **Four malicious users**, dominated by CDE1846 (46 of 70 positive days), which
  also shows the largest gain. Intervals are descriptive and conditional on
  these adapters and this fixed benign cohort; they are not seed replication.
- **Exploratory.** These evaluation users have been inspected in earlier phases.

## The mechanism, as far as this shows it

At 8B the profile contribution is *anti*-predictive: `profile_only` user ROC is
0.200, and below 0.5 for all four users individually. The mean-gap
decomposition shows PSY contributing negatively for three of four users
(−0.069, −0.117, −0.136, −0.063) and SES positively for all four
(+0.052, +0.040, +0.144, +0.038), with the total negative for three — which is
why the day-level ROC of the full score is *below* chance at 0.399. The
adapter finds these malicious users' profiles less surprising than the unseen
benign users' profiles, and that term drags the total score down far enough to
bury a behavioural signal that is in fact strong (0.836/0.938).

At 3B the same term is mildly *predictive* (`profile_only` 0.649), which is
consistent with the 3B `full` adapter not failing in the first place.

For comparison, the published retrained `no_profile` adapter reaches user ROC
0.903 at 8B. Removing the direct profile contribution from the score of the
*unmodified* adapter reaches 0.938 — no retraining involved. This does not show
that retraining is unnecessary in general, and it does not address whether the
adapter's behavioural predictions are themselves shaped by profile context.

## Files

`score_view_summary.csv` (means with bootstrap CIs per view and metric),
`score_view_contrasts.csv` (paired contrasts vs `full`, with per-fold deltas),
`score_view_folds.csv` (every metric per fold per view, incl. population sizes
and prevalence), `mean_gap_decomposition.csv` (signed weighted class
contributions per fold), `score_view_meta.json` (view definitions, fold seed,
population), `class_scores_manifest.json` (checkpoint, tokenizer digest,
throughput, all reproduction checks).

Regenerate with `slurm/anvil_friend/score_decomposition.sh {8b|3b} full`
(scores the pool, ~3.3 h on H100 / ~4.9 h on A100 at batch 1) followed by
`scripts/eval_score_views.py`.
