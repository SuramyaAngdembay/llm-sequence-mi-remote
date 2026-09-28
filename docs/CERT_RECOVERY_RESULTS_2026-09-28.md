# CERT label recovery: completed cached-analysis stage

Date: 2026-09-28. Base commit: `017480c`. Branch: `repair/cert-calendar-labels-20260928`.
Worktree: `/Users/suramya/llm-sequence-mi-recovery-20260928`. Artifact root: `/Users/suramya/Documents/mi-paper-recovery-2026-09-28`.
Scope: MI research; thesis excluded.

## Verdict

The calendar-label repair and cached-score reanalysis are complete. The main
score-decomposition result survives, with stronger corrected behavior-token day
ranking in several runs. The causal/mechanistic paper is **not yet repaired**:
dictionary fitting, feature selection, intervention eligibility, and contaminated
comparison-model training still require their own recovery work. No GPU jobs were
submitted and this stage used **0 GPU SU**. Main LM adapters were retained.

## What was repaired

- Built versioned labels from the checksum-verified official master incident
  windows, intersected with session calendar days using [start,end). The session
  origin is 2009-12-28. This repairs the mismatched clocks, not an arbitrary integer shift.
- r4.2: 1,309 → **1,362** positive observed days; 60 → **70** positive users;
  **407** changed day labels (177 positive→negative, 230 negative→positive).
- r6.2: 70 → **73** positive observed days; 4 → **5** positive users;
  **23** changes (10 positive→negative, 13 negative→positive).
- Preserved original split assignments. Actual main train/validation JSONL audits
  found no official insiders. Adapter reuse clears this label bug; it is not a
  claim that every aspect of the detector is validated.
- Copied and checksum-verified 50 cached inputs. Wrote new label tables, metadata,
  per-day change ledgers and relabeled score caches. Copied both original evaluation
  JSONLs and wrote corrected copies, checking every non-label value and row order.
- Updated both session builders to require verified calendar labels, emit their
  date/version, and refuse an existing output directory. A separate relabel command
  preserves existing serialized text rather than rebuilding it.

The official master and detailed incident-event lists define different targets.
The known malformed CDE1846 start is repaired from its named-user detail events;
other master/detail discrepancies are recorded. Event-day sensitivity stays separate.

## Reading the numbers

Unless explicitly called pooled, the tables below average one-positive-user folds
equally. Each fold uses available benign users up to the historical 800-user cap.
Corrected r6.2 has five folds, r4.2 main has 70, and the restricted r4.2 class cache
has 60. Old→corrected fold changes include the changed positive-user/negative-cohort
definition; they are not a pure label-only effect on fixed folds. Pooled old/new
tables use the same score rows and are also supplied.

Uncertainty uses 10,000 paired positive-user bootstrap draws, seed 42. With five
r6.2 users, intervals are descriptive, conditional on fixed models and negative
cohorts. They do not establish replication across seeds, datasets or architectures.

### 1. Behavior-only scoring survives

| Run | Full day ROC | Behavior day ROC | Full day AP | Behavior day AP |
| --- | --- | --- | --- | --- |
| r6.2, Qwen2.5-3B factorial adapter | 0.683 | 0.893 | 0.001048 | 0.005480 |
| r6.2, Qwen3-8B factorial adapter | 0.412 | 0.897 | 0.000092 | 0.001288 |
| r4.2 headline adapter, restricted cohort | 0.608 | 0.818 | 0.005658 | 0.029369 |

The profile remains in the model input: the behavior score averages the losses of
behavior targets only. This supports a scoring improvement, not removal of internal
profile use or proof of a particular circuit.

Corrected day-ROC improvements are +0.210 [0.066,0.352] for 3B and +0.484
[0.376,0.573] for 8B. The r4.2 restricted improvement is +0.210 [0.138,0.285].
For within-user day ranking, the r6.2 improvements are much smaller: +0.048
[-0.011,0.119] for 3B and +0.014 [-0.007,0.037] for 8B. Strong across-user
ranking improvement must not be presented as equally strong attack-day localization.

AP is small in absolute terms. Corrected r6.2 fold prevalence averages 0.0001035
(range 0.00000709–0.0003190); all-positive-users pooled prevalence is 0.0005138.
These are different estimands. AP is not precision at a chosen alert budget.

**r4.2 is still incomplete:** its class cache omits 1,949 days from ten users,
including 19 corrected positive days. Its 40,519-row result cannot replace the
42,468-row full unseen-cohort result until those scores are added. Restricted
pooled prevalence is 0.033145; mean fold prevalence is 0.0008212.

### 2. The factorial result survives, but the 3B interpretation changes

| Scale | Training/scoring condition | Corrected day ROC | Corrected user ROC |
| --- | --- | --- | --- |
| 3b | full | 0.683 | 0.778 |
| 3b | no_psy | 0.799 | 0.901 |
| 3b | no_profile | 0.894 | 0.950 |
| 3b | shuffle_profile | 0.432 | 0.550 |
| 8b | full | 0.412 | 0.472 |
| 8b | no_psy | 0.791 | 0.901 |
| 8b | no_profile | 0.899 | 0.902 |
| 8b | shuffle_profile | 0.394 | 0.438 |

These are the 300,000-example train-matched factorial runs, with one adapter seed
per condition. They are distinct from the original headline 8B adapters.

Removing profiles raises 8B user ROC by +0.431 [0.161,0.684]. The 3B point
improvement is now +0.171 [-0.033,0.461], compared with the old +0.030.
The old description of 3B as having only a small removal effect must be revised;
the corrected effect remains imprecise. No universal model-size explanation follows.
Training/scoring removal changes both the training distribution and scored fields.

### 3. Main-detector exposure imbalance remains

| Dataset | Comparison pool | Old fold day ROC | Corrected fold day ROC |
| --- | --- | --- | --- |
| r42 | all_users | 0.964 | 0.960 |
| r42 | unseen_users | 0.619 | 0.538 |
| r62 | all_users | 0.953 | 0.956 |
| r62 | unseen_users | 0.565 | 0.583 |

Strong results against a population containing training users continue to coexist
with weak results on unseen users. The contrast is a population/exposure warning;
it is not a controlled causal identification of memorization.

The full-population scoring pass and token-extraction scoring pass are not
numerically identical on overlapping IDs (mean adapted-NLL differences 0.00917
r4.2 and 0.00521 r6.2). They are kept separate. Masking comparisons use the latter
pass as their full-input baseline. Their exact scoring-configuration difference
has not been resolved here and should be pinned before new interventions.

### 4. Input masking is not interchangeable with score masking

For the original headline adapters, removing PSY at inference raises corrected
fold day ROC from 0.537 to 0.707 on r4.2 and from 0.585 to 0.741 on r6.2.
Removing the whole profile gives 0.509 and 0.340 respectively. Therefore the
behavior-only score result does not justify a blanket instruction to remove all
profile inputs from an already-trained model.

### 5. Target-mask retraining has no clear added benefit

Comparing behavior-only scores on the same r6.2 keys, target-mask training changes
fold day ROC from 0.893 to 0.899: +0.006 [-0.005,0.018]. Mean fold AP falls from
0.005480 to 0.003376. This is not evidence that retraining is superior to rescoring
the existing full-training adapter.

### 6. Numerical time-series results remain usable

All three saved checkpoints were reloaded and rescored on **all 1,393,297 rows**
using CPU. Context/user boundaries, forecast target exclusion, history flags,
feature identities, train-only standardisation and input hashes passed. Comparison
uses the predeclared tolerance rtol=atol=1e-5; CPU/GPU errors are not claimed to be
bitwise zero. Largest absolute error was 0.001160, within the relative/absolute
tolerance for the affected score. No model was retrained.

On the common 141,662-row unseen/history-eligible pool, corrected full-score pooled
day ROC is 0.701 (w1 reconstruction), 0.708 (w7 reconstruction), and 0.726
(forecast). Profile shares of the signed positive-minus-negative mean gap are
**3.56%, 12.78%, and 0.121%**. The previously reported negative w1 share flips sign;
behavior channels still dominate this particular accounting statistic. This is
not a causal attribution and does not prove internal profile irrelevance.

The forecast history-shuffle contrast was not rerun in this stage. w7 reconstruction
includes the target day; the forecaster excludes it and uses up to seven previous
observed rows, with padding. These are not necessarily seven consecutive calendar days.

## Other checks and qualifications

- **96 independent ROC/AP calculations** agreed with the shared evaluator to
  2.22e-16, using pair counts and grouped score thresholds rather than its sklearn calls.
- **18 regression tests** and the existing metric-core correctness checks passed.
- Each r6.2 factorial/class cache contained 16,992 repeated IDs. All supplied
  metadata, scores and counts agreed exactly before deduplication. Raw→deduplicated
  changes are separately recorded; maximum pooled day-ROC difference was 0.000508.
  Old-label recomputation is not automatically identical to every historical evaluator.
- Named-user observable-day sensitivity on r6.2 gives 44 positives, versus 73
  incident-window positives, with 49 changed labels. Pooled behavior-only ROC still
  exceeds full-score ROC for both factorial models (3B 0.956 vs 0.605; 8B 0.951 vs
  0.372). This sensitivity does not redefine the primary target.
- History-prefix and swap results are fixed-historical-sample loss contrasts only.
  Correcting their labels does not repair their label-dependent selection or donors.
- Existing score-cache checks establish reproducible cached analyses. Historical
  LM weights/configuration have not been reloaded to regenerate every cache here.

## What remains before the paper is scientifically repaired

1. Independently review the label and metric recovery, including the two scoring
   passes and every manuscript claim proposed for reinstatement.
2. Add the missing r4.2 class scores and repaired-3B extraction coverage. Bind
   checkpoint, tokenizer, exact text and target scoring to each new run manifest.
3. Freeze the corrected SAE training population and sampling rule. Refit
   label-dependent/benign-only dictionaries, rerank features, rebuild discovery and
   confirmation populations, and verify corrected receiver/donor eligibility.
4. Compare exact old/new intervention tuples. Reuse only genuinely identical
   model/input/dictionary/edit tuples. Run a small complete causal pilot with fresh
   baselines, alpha-zero checks and realized patch coverage, then estimate GPU cost.
5. Refit contaminated classical, older sequence and local-AE comparison pipelines
   on corrected splits. They were not repaired by this cached LM reanalysis.
6. Update all three paper versions together after the relevant gates pass. TWOS,
   LANL and mathematical-novelty limitations remain separately tracked.

The earlier ~204.75 GPU SU figure is historical causal spending, **not** an
estimated repair bill. A defensible rerun estimate still requires the tuple inventory
and timed pilot. Do not blanket-resubmit the old jobs.

## Artifact map

- `labels/{r42,r62}/`: calendar/event labels, manifests, metadata and changed-day ledgers.
- `jsonl/{r42,r62}/eval_calendar_v1.jsonl`: corrected frozen-input evaluation copies.
- `evaluation_v1/`: 134 pooled rows, 2,304 fold rows, 976 fold-summary rows,
  720 paired metric contrasts, coverage and fixed-sample results.
- `evaluation_checks_v1/old_vs_corrected_fold_registry.csv`: paired old/new registry.
- `evaluation_checks_v1/`: independent checks, duplicate effects, sensitivity,
  scoring-pass comparison and numerical mean-gap decomposition.
- `timeseries_verification/`: new read-only CPU checkpoint/cache gates with hashes.
- `claim-status-after-cached-recovery.csv`: all 46 claim families and remaining work.
- `INDEPENDENT_REVIEW_PROMPT.md`: concrete handoff for Claude Code or another reviewer.
