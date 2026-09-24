# LANL token-class score decomposition (job 20827647; collected 2026-09-22)

> ## ⚠ POPULATION DEFECT — correction of 2026-09-24. Read before the Reading section.
>
> User sampling (ETL: keep an ordinary user iff `md5(user) % 10 == 0`) and fold
> assignment (split: `fold = md5(user) % 5`, fold 4 unseen) shared a hash, so
> every sampled ordinary user fell in seen fold 0. Verified on the saved
> `eval.jsonl`: seen = 1,063 users (75 with an attack window, 988 without);
> unseen = 23 users, **all** with attack windows, 0 negative-only users, and
> all 30,212 unseen negative windows come from those attack users. The seen and
> unseen pools therefore differ in user composition as well as training
> exposure. The AUCs below describe these two populations only.
>
> **Withdrawn:** "the model learns which identities are anomalous", "LANL
> shows memorisation", and any reading of this comparison as unseen-user
> transfer failure. Training is benign-only, identity-token loss is
> context-dependent, and user/host fields can carry legitimate relational
> behaviour, so the decomposition could not establish memorisation even
> without the defect.
>
> **Corrected:** "AP is near the base rate everywhere" is wrong. Seen AP
> 0.176 is about 17 times its prevalence of 0.0104; unseen AP 0.013 is about
> twice its prevalence of 0.0065.
>
> **Descriptive only (same saved scores, original populations):** restricting
> the seen pool to negatives from attack users, the composition of the unseen
> pool, gives full-score AUC 0.916 (from 0.949). Within each attack user,
> attack windows against the same user's benign windows: mean per-user AUC
> 0.883 seen (70 users) and 0.738 unseen (23 users). So composition explains
> little of the pooled gap, and unseen attack windows are still ranked above
> the same user's benign windows. Seen users were trained on in every one of
> these comparisons; none repairs the design. The corrected protocol is in
> `docs/REVIEW_2026-09-24_ISSUE_LEDGER.md`.


The score-decomposition method applied to the LANL authentication replication,
under LANL's own field-level schema (`su`/`du` → ID_USER, `sc`/`dc` → ID_HOST,
`at`/`lt`/`or`/`res` → BEHAV, `t<hh>` → HOUR). All 52,973 evaluation windows
re-scored at **batch size 1**; the recomputed full score reproduces the cached
one to a mean absolute 3.41e-04 nats, rank correlation 0.99996 — far
tighter than CERT, because these windows are short and batching drift is small.

## Result — window-level AUC, seen versus unseen users

| pool | view | AUC | AP | recall @ 1 % FPR | 95 % AUC interval |
|---|---|---|---|---|---|
| **seen** (22,563 windows, 234 positive, 1,063 users) | full | **0.9490** | 0.176 | 0.295 | [0.932, 0.963] |
| seen | identity only | **0.9395** | 0.178 | 0.321 | [0.918, 0.956] |
| seen | behaviour only | 0.8495 | 0.076 | 0.150 | [0.813, 0.881] |
| seen | behaviour, no hour | 0.8739 | 0.133 | 0.244 | — |
| **unseen** (30,410 windows, 198 positive, **23 users**) | full | 0.4957 | 0.013 | 0.035 | [0.408, 0.883] |
| unseen | identity only | 0.4921 | 0.014 | — | — |
| unseen | behaviour only | 0.4840 | 0.008 | — | — |

## Reading

**On LANL the identity tokens are not a shortcut that hurts — they are the
signal, and it does not transfer.** On seen users, scoring identity tokens
alone gives AUC 0.939, almost the full score's
0.949; dropping to behaviour tokens alone costs a tenth of
an AUC point. On unseen users every view is at chance, 0.48–0.50.

This is the **opposite** of CERT r4.2, where behaviour-only scoring *raises*
user ROC from 0.65 to 0.86. The two are consistent under one account: the
model learns which identities are anomalous. On CERT, profile tokens name a
user's organisational type, which is a shortcut that misleads on new users. On
LANL, identity tokens name the user, and the red-team users are specific
accounts seen in training — so identity is the label on the seen pool, and
nothing on the unseen pool. Neither dataset shows a transferable behavioural
signal in the token-class views; CERT shows a removable shortcut, LANL shows
memorisation.

## Limits

* **23 unseen users.** The unseen AUC interval is [0.41, 0.88] — it
  cannot distinguish chance from a moderate effect. "At chance" is the point
  estimate, not an established null.
* Window prevalence is 1.0 % seen / 0.65 % unseen, so AP is near the base
  rate everywhere and recall at 0.1 % FPR is not informative.
* One adapter seed. Given that a seed change reversed the TWOS package-4 result,
  this deserves a second seed before it carries weight.

Artifacts: `lanl_score_view_summary.csv`, `lanl_score_view_contrasts.csv`,
`lanl_score_view_meta.json`, `class_scores_manifest.json`.
