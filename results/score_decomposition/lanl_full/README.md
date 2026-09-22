# LANL token-class score decomposition (job 20827647; collected 2026-09-22)

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
