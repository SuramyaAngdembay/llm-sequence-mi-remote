# Four-package review response — verdicts and deliverables

Dated 2026-09-19. Source specification:
`~/Documents/mi-paper-review-2026-09-19/claude-next-experiments-prompt.md`.

All work below ran on Aquaman at **zero cluster SU**. Anvil is down for a
two-day outage from 2026-09-19; the failure is below SSH (100 % ICMP loss, port
22 unreachable) and identical on both accounts. Nothing is assumed about jobs
20816765, 20827646 or 20827647.

The outage did **not** block package 3. Aquaman holds both Qwen3-8B adapters,
the repaired session data for r4.2 and r6.2, and the base model, and runs the
8B in 4-bit across two RTX 3070s. Package 3 is therefore **complete**. Package
4 still needs Anvil for CERT, whose token-delta cache and SAE frontier live
only there, but has been run end to end on the **TWOS** replication.

---

## Verdict per package

| package | verdict |
|---|---|
| 1. Preserve and correct the r4.2 scoring-mitigation result | **Complete.** Result preserved; three overstatements corrected. |
| 2. Forecasting fix, reproducible checkpoints, one shared metric, identical eligible examples | **Complete.** All four gates pass; one new limitation found. |
| 3. Profile copying within the same frozen LM | **Complete.** Run on both releases and both prefix formats. Strong, replicated effect. |
| 4. Checkpoint-matched SAE interventions against full / profile / behaviour loss | **Code complete; run on TWOS, CERT still blocked on Anvil.** |

---

## Corrected metrics, with the population each describes

### Package 1 — r4.2 headline (60 malicious users, fold-aligned)

The result stands: user ROC **0.653 → 0.863** under behaviour-only scoring,
improving in 51 of 60 folds, with every interval excluding zero except
recall@1 % FPR. Three corrections:

* Recall figures are **empirical ROC operating points** calibrated on the same
  negatives they are evaluated against, not deployment thresholds.
* The cache discrepancy is **batching-dependent**, not padding-specific:
  batch 1 gave 1.295e-02, batch 8 gave 1.060e-02, batch 56 gave 2.948e-03.
* Verification status is now explicit: saved fold CSVs reproduce to 3.6e-15;
  the **raw scores were not re-verified** (Anvil timeout).

### Package 2 — numerical detectors (141,662 common eligible rows, 70 positives, 4 malicious users, 410 users)

Population: all days of the answer-key users plus the validation benign cohort
— the same population the language-model evaluator uses. The forecaster's 410
missing-history rows are dropped from **all** models.

| model | view | day ROC | user ROC |
|---|---|---|---|
| recon 1-day | full | 0.6739 | 0.7894 |
| recon 1-day | behaviour only | 0.6856 | 0.7734 |
| recon 1-day | profile only | 0.5927 | 0.7328 |
| recon 7-day | full | 0.6829 | 0.7851 |
| recon 7-day | behaviour only | 0.6979 | 0.7777 |
| recon 7-day | profile only | 0.5968 | 0.8023 |
| forecast 7-day | full | 0.7061 | 0.7389 |
| forecast 7-day | behaviour only | 0.7068 | 0.7389 |
| forecast 7-day | profile only | 0.5900 | 0.6773 |

Profile share of the malicious-benign mean score gap: **−3.7 %**, **+10.2 %**,
**+0.24 %**.

Day AP is 0.0007–0.0019 at a prevalence of 0.00049. Nothing here is a usable
detector.

### Package 3 — history prefix (200 users per release, capped at 15 days each)

Clustered by user, bootstrap over users. r6.2 with the training-format prefix,
2,982 examples over 199 users:

| view | B − A (own profile) | C − A (stranger) |
|---|---|---|
| psychometric only | −3.226 [−3.352, −3.102] | −0.413 [−0.548, −0.274] |
| profile only | −1.304 [−1.362, −1.248] | +1.122 [+1.044, +1.199] |
| behaviour only | +0.016 [+0.014, +0.018] | +0.133 [+0.125, +0.141] |
| behaviour, SES lines only | −0.002 [−0.002, −0.001] | +0.124 [+0.116, +0.132] |

Psychometric loss falls from 3.373 to 0.148 nats when the user's own profile is
in context, in **100 % of users**, replicated on r4.2 and in both prefix
formats. The all-60-malicious-user r4.2 sample gives −3.058 [−3.208, −2.908].

Measured, not assumed: the static-profile overlap between a user's earlier and
current record is **1.0000**, so condition B is a **copying** test, not a
memory-of-a-different-past test.

---

## Implications for the paper's three questions

**Score construction.** The correct behaviour score is the exact conditional
mean over behaviour tokens, `s_B = (loss_sum_total − loss_sum_P) / (N − N_P)`.
The naive `s_full − s_P` equals `w_B(s_B − s_P)` and can reorder examples; the
ranking-preserving subtraction is `s_full − w_P·s_P`. Every evaluator in the
repository now takes the conditional mean from sums and counts directly, and a
test asserts that the naive form reorders while the weighted form does not.

**Context dependence.** Day-level detection by the forecaster genuinely uses
the user's own history: replacing it with a stranger's costs 0.134 ROC. The
user-level ranking costs only 0.025, so it is close to a statement about how
atypical a user's days look on their own. This is the cleanest evidence so far
that "sequence model" claims should be made at the day level, not the user
level. It also predicts what Package 3 should find, and that prediction is
recorded before the run rather than after.

**Causal features.** Partly answerable. The intervention code decomposes every
repair by token class, its correctness is established against an independent
reference implementation, and it has now run end to end on TWOS. The CERT run
still needs Anvil. The TWOS smoke already vindicates one design choice
concretely: the cached base score differs from a matched-batch recomputation by
a mean of 1.1e-02 nats, which is **larger than the repair effects being
measured**, so recomputing the base through the identical code path was not
optional.

**Profile copying.** Answered, and it reshapes the account. The adapted model
has a strong, identity-specific in-context copy mechanism for profile tokens:
the user's own profile collapses psychometric loss by 3.2 nats, a length-matched
stranger's does not. But the correct profile buys **no** behavioural predictive
value (+0.016 nats or less; −0.002 for the SES lines alone), while a wrong one
**hurts** behaviour (+0.12 to +0.21). The profile functions as an identity key
that conditions behaviour prediction, not as evidence about behaviour. The
organizational and psychometric halves also dissociate: day fields are already
cheap with no context (0.44 nats) while psychometric fields are not (3.37), so
"the model memorized the profile" cannot be the whole story.

---

## Remaining uncertainties

1. **Condition B is a copying test, not a memory test.** CERT profiles are
   static, so B re-presents information also present in the current day's own
   profile line. It shows the copy mechanism exists and is identity-specific; it
   does not establish that the *deployed* profile-token loss is copying-driven,
   since in deployment no earlier profile is in context.
2. **Whether SAE repairs act on profile or behaviour tokens is unknown for
   CERT.** The TWOS run will answer it there; CERT remains the load-bearing
   case and needs Anvil.
3. **Four malicious users** in the time-series population. Every interval there
   is descriptive.
4. **CERT profiles may be static per user**, which would make condition B a
   test of copying rather than of temporal history. The probe measures this
   rather than assuming it.
5. **Anvil job state is unknown**, including whether the Phase C 3B training arm
   ever ran.

## Single most informative next experiment

**Package 4 on CERT r4.2**, the moment Anvil returns. It is the only one that speaks
directly to the paper's mechanistic claim: the delta-SAE audit says specific
features carry identity information, and the causal test says patching them
repairs the score — but "the score" is a pooled mean over profile and behaviour
tokens. If repairs turn out to act almost entirely on profile tokens, the
mechanistic story and the mitigation story become one result instead of two.
If they act on behaviour tokens too, the features are doing more than the
shortcut account predicts. Either outcome is publishable; the current pooled
number cannot distinguish them. The smoke run costs under 0.5 SU.

---

## Paragraph for `paper/companion/readable_full.tex`

> We also asked whether the profile shortcut is forced by the data. It is not.
> We trained small numerical detectors on exactly the fields the language model
> sees — the same organizational and psychometric attributes, the same session
> counts and durations — and scored them so that the profile and behaviour parts
> of the score separate exactly. Across a one-day autoencoder, a seven-day
> autoencoder and a seven-day forecaster, the profile fields account for between
> −3.7 % and +10.2 % of the gap between malicious and benign scores; for the
> forecaster the figure is 0.24 %. The language model, given the same fields,
> puts a large share of its gap into the profile tokens. The difference is
> therefore a property of the model and its objective, not of the feature set.
> Two cautions belong with this result. All three detectors are weak in absolute
> terms, with average precision near the base rate, so the comparison is about
> where a score's mass sits and not about detection quality. And for none of the
> four malicious users in this population does any detector rank an actual
> attack day highest: a high user-level score reflects a user whose ordinary days
> look unusual, not a located incident. We report the second point because
> max-aggregated user AUC hides it, and it qualifies every user-level number in
> this literature, including ours.
