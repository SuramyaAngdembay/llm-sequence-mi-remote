# Label day-index misalignment (found 2026-09-28)

**Status: measured on r4.2 and r6.2. The root cause is identified by date arithmetic. Nothing
has been fixed or rerun.** This was found while verifying Chapter III of the thesis.

## What is wrong

The day-level labels (`labels_daily.parquet`) and the LC-DAL session extraction (the `day` column
of `session*_shard_*.csv.gz`) count days from different origins:

| file | day 0 | check |
|---|---|---|
| session extraction | 28 December 2009 | session day 5 = 2 January 2010, the first day of data; day 505 = 17 May 2011 |
| `labels_daily` | 2 January 2010 | the scenario-3 insider BBS0039 has answer-key events on 12 August 2010, which is label day 222 |

The same date, 12 August 2010, is therefore **label day 222** but **session day 227**.

`build_session_jsonl*.py` joins the labels to the session days on the raw index. The key is
`(user_id, day_index)`. Each malicious date's label therefore lands on the session day **five days
earlier**.

## Evidence (`scripts/check_label_alignment.py`, outputs in `results/label_alignment_2026_09_28/`)

The extraction flags its own malicious sessions (`insider` > 0 on a session row). Shifting the
labels by +5 days maximizes their overlap with those flagged session-days for every scenario of
both releases.

| release, scenario | flagged session-days | matched as joined now | matched after +5 |
|---|---|---|---|
| r4.2, scenario 1 | 68 | 25 | 56 |
| r4.2, scenario 2 | 861 | 817 | 861 |
| r4.2, scenario 3 | 20 | 0 | 20 |
| r6.2, all five scenarios | 34 | 24 | 33 |

Scenario 2's long labelled windows mostly overlap the true days even when shifted. The short
episodes of scenarios 1 and 3 mostly miss.

## Effect on the populations used so far

| release | malicious user-days / insiders, as used | after aligning (+5) | days as used that stay malicious | true malicious days missing |
|---|---|---|---|---|
| r4.2 | 1,309 / 60 (scenarios 1, 2, 3: 29, 30, 1 insiders) | 1,355 / 70 (30, 30, 10) | 1,139 of 1,309 | 216 |
| r6.2 | 70 / 4 | 73 / 5 | 60 of 70 | 13 |

- The "60 malicious users" of r4.2 exclude 9 of the 10 scenario-3 insiders and one scenario-1
  insider. They were excluded only because their shifted labels fell on days without sessions.
- The one included scenario-3 insider, BSS0369, is labelled on the days before its malicious
  activity.
- Earlier documents attributed the gap between 1,883 answer-key days and 1,309 matched days to
  "active-session coverage", for example `docs/VALIDITY_AUDIT_2026-07-18_CERT_DATA_AND_MECH.md`.
  Part of that gap is this misalignment.

## What it affects (reasoned, not recomputed)

- **User-level results** (user score = the user's worst day): scores of the included users do not
  depend on day labels. The malicious *population* does change: 60 → 70 users on r4.2, 4 → 5 on
  r6.2. Headline user-level AUCs should be recomputed on the aligned population.
- **Anything using malicious days** is affected, because about 13% of r4.2's labelled days are the
  wrong days and 216 true days are unlabelled:
  - day-level metrics;
  - SAE feature ranking (positive rows);
  - the causal and necessity tests, whose receivers are malicious days;
  - Pilots 1–4;
  - H6 attack days;
  - the imperfect-intervention H6 states.
- **Benign-only analyses and the synthetic study** are unaffected.

## Recommended next step (not taken)

1. Build the label map on calendar dates, or add 5 to `labels_daily.day_index` before the join.
   Then verify with this script that the best shift becomes 0.
2. Rebuild the JSONL labels only. The texts do not change.
3. Recompute the evaluation populations and the analyses listed above, in order of dependence.

Which results to rerun, and how to report the change, is a decision for the paper and the thesis.

## Addendum: the labels are activity windows, not event days

`labels_daily` marks every day in each insider's answer-key window, from the first to the last
malicious event (`insiders.csv` start and end). 1,871 of its 1,883 labelled days fall inside
those windows, which total 1,892 days.

The LC-DAL extraction flags only days with a malicious event: 966 days in `dayr4.2.csv` (85, 861
and 20 by scenario). Both definitions are legitimate, but they differ, and the thesis now states
which one each experiment uses. The five-day misalignment above applies to the window labels as
joined in the language-model pipeline.
