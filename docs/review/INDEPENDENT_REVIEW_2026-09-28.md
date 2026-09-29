# Independent review of the CERT calendar-label recovery (Codex commit c31efcf)

Reviewer worktree `/Users/suramya/llm-sequence-mi-review-20260928`, branch
`review/cert-independent-20260928` (from `c31efcf78be6`). Review directory: this folder.
Python: the recovery `.venv` (numpy 2.0.2, pandas 2.3.3, scipy 1.13.1). Every check below uses
code written for this review (`scripts/review/*.py` on the branch), which imports none of
`cert_calendar_labels.py`, `eval_cert_recovery.py`, `eval_metrics_core.py` or `make_folds`.
Where the algorithm is inherently the same (the historical fold rule), it was re-implemented.

## Verdict

**The label repair and the cached reanalysis are correct within the scope Codex claimed.**
I found no computational error:

- **Labels.** Every positive key reproduces independently.
- **Clock.** The session clock is confirmed at event resolution. This is stronger evidence than
  the recovery had.
- **JSONLs.** The corrected JSONLs preserve every non-label value and the row order.
- **Metrics.** A predetermined set of 48 metric values reproduces to within 2.6e-16 with
  separate code, and the fold user lists match exactly.

I resolved one open question: the unresolved difference between scoring passes. I also found
four things the recovery documents must add or reword:

- **F2.** The scoring-pass difference is batch-shape numerics. It is too small to matter for
  detection metrics but ten times the size of typical intervention effects.
- **F3.** The masking contrast mixes scoring configurations.
- **F5.** The corrected r4.2 main-detector drop is carried by the newly included scenario-3
  insiders.
- **F6.** The w7 time-series gate is a near-boundary pass.

None of these changes a cached-analysis number Codex reported.

No causal, dictionary or comparison-model claim is cleared by this review.

## Findings

| # | severity | finding | evidence |
|---|---|---|---|
| F1 | confirms | **Labels and clock verified end to end.** Checks: archive MD5 `c0c96eaa…` (published); one malformed master field (r6.2 CDE1846 start `/21/2011 11:43:39`), repaired to the named user's earliest event; complete (user, date) key sets for the incident-window and observable-day tables equal my enumeration; overlay `y` equals mine on all 330,295 and 1,393,297 rows; 1,362/70 and 73/5 positives; 407 (177/230) and 23 (10/13) changes; change ledgers equal. **Clock**, on every session row (470,611; 1,949,074): (a) epoch day − day index = 2009-12-28; (b) UTC time of day equals the extractor's recorded start hour (`ses_start`) to < 1 min; (c) **100%** of the 7,273 r4.2 and 420 r6.2 official named-user events fall inside that user's session intervals under the UTC-naive reading, falling to 83–86% at ±1 h and 44–55% at ±5 h. So session dates are recorded calendar dates. | `independent_label_checks.json`; `inputs/session_slim/*_source_sha256.txt` |
| F1b | confirms | r6.2 master/detail discrepancies are exactly the three Codex lists (ACM2278: 2 events; CMP2946: 97 events, 2011-02-02…03-30; PLJ1771: 8 events). r4.2 has none. 530 of r4.2's 1,892 window days and 26 of r6.2's 99 have no session, so they are unobservable in the LM domain. | same |
| F1c | confirms | **Corrected JSONLs** (42,468; 142,072 rows): 0 violations of order, field set, non-label value (text included), `y_original`, `calendar_date` or corrected `y`; keys = metadata val+eval in the same order. The Anvil originals still hash to the staged copies. The main train/val files still hash as in the audit and contain 851 / 79 / 3,590 / 405 users with **0** official insiders (re-derived on Anvil from the files). The r4.2 adapter's `train_summary.json` names that data directory, with n_train 287,827. | `independent_jsonl_checks.json`; `anvil_jsonl_sha256_now.txt` |
| F1d | confirms | **Metrics:** 48 predetermined values reproduce within 2.6e-16 and the fold lists match exactly. The selection was written before computing. It covers both datasets, both scales, the changed-label main-detector case, three newly positive scenario-3 folds and the 8B/3B profile-removal bootstrap. Checks: rank-based ROC, grouped-threshold AP, re-implemented fold rule, exact duplicate agreement, partition and n_tokens − 1 counts (0 mismatches), finite scores. Scores rank higher = more anomalous as documented. Observable-day sensitivity also reproduces (3B 0.605 → 0.956, 8B 0.372 → 0.951, 44 positives). | `metric_reproduction_selection.md`, `independent_metrics.json`, `metric_reproduction_comparison.csv`, `observable_sensitivity_recheck.json` |
| F2 | documentation, **important for causal work** | **The fullpop-vs-extraction scoring difference is resolved.** Same adapter (sha256 `e0bf5112…` for every r4.2 copy), tokenizer, text and token counts in all passes. A controlled batch test already on the collaborator scratch holds the code, adapter and 256 examples fixed. Batch size 1 is **bit-identical** across two separate runs (pilot vs full class pass, 256/256 equal). Batch sizes 1, 8 and 56 differ from each other by 0.010–0.013 mean absolute NLL, with no exact agreements. Fullpop (batch 56, July) and the batch-56 test differ by 0.003, being the same size in different batch compositions. Cause: batch-shape-dependent bf16 / 4-bit numerics. The July-2 extraction pass also computed cross-entropy on **bf16 logits** (`extract_adapter_deltas.py` at `0eafe05`, no `.float()`); the fullpop pass cast to fp32. Effect on detection on identical keys: fold-mean day ROC moves ≤ 0.0025, user ROC ≤ 0.006, Spearman 0.993. **But per-example noise of ~0.01 nats is about 10× the effects measured in H6 and similar to Package 4 contrasts.** Causal baselines must be computed in the same batch configuration as their edits. The fullpop cache's "portability reference" is a copy of fullpop, not a separate pass. | `inputs/batchtest/`, `scoring_pass_metric_sensitivity_r42_restricted.json` |
| F3 | wording | **Masking arms mix scoring configurations.** The full arm is the Anvil extraction pass (A100, batch 4, bf16 CE); the masked arms were scored on Aquaman (RTX 3070, batch 2, fp32 CE) with the identical adapter. From F2 the configuration effect on day ROC is of order 0.003. It cannot explain the r4.2 no_psy effect (+0.170 [0.114, 0.226]), but small masking contrasts (r4.2 no_profile −0.027 [−0.081, 0.027]) must not be read at the 0.005 level. | Aquaman `score_summary.json` |
| F4 | scope | Fold day metrics count an insider's out-of-window days as negatives. Call them **window-negative**, not confirmed benign activity. User metrics use each user's full history (verified). | `independent_metrics.py` |
| F5 | substantive wording (C06, C07) | **The corrected r4.2 main-detector drop (fold day ROC 0.619 → 0.538) is carried by the newly included insiders.** 9 of the 10 are scenario 3: mean fold day ROC 0.059, user ROC 0.125, so the full score ranks their attack days *below* benign days. The original 60 give 0.607 under corrected labels. The restricted r4.2 class cache is exactly the old 139-user pool; its 10 missing users are exactly these. So the behavior-only result on the restricted cohort (0.818) is silent on the hardest scenario. | `r42_main_unseen_corrected_folds_by_scenario.csv` |
| F6 | disclosure | Time-series gates pass as reported. **w7 PROFILE uses 99.5% of its tolerance.** I reran all three checkpoints on Aquaman at 4 and 1 threads: identical errors, deterministic. A tolerance pass is not bitwise equality, as Codex states. | `timeseries_gate_rerun/` |
| F7 | provenance | Coverage of the r4.2 class cache reproduces: 40,519 of 42,468 unseen days; 1,949 missing days, 10 users, 19 corrected positives, 0 old positives, users missing entirely rather than partly. The factorial adapters remain separate from the headline adapters, and their training-byte provenance is weaker (metadata-level). I did not strengthen it. | `independent_metrics.py` coverage block |

**Findings added during Stage B preparation (before any new effect was computed):**

| # | severity | finding | evidence |
|---|---|---|---|
| F8 | reproducibility | `train_sae` never seeds torch, so no historical dictionary can be refit bit-for-bit. Its `--seed` only controls row sampling, which confounds the C18/C19 seed study with initialisation noise. Fixed on the branch with an optional `--torch-seed` (default unchanged). | `scripts/train_delta_sae_frontier.py`, `sae_core.train_sae` (unseeded DataLoader shuffle and initialisation) |
| F9 | **eligibility error in historical interventions** (C15, C25, C35) | The historical donor pools treated every day labelled 0 as a benign donor, including official insiders' days. **Package 4:** 19,619 "benign" donor days from 136 users, of whom **65 are official insiders**: 7,932 days, of which 7,798 are window-negative and 134 positive, making up **43% of benign-donor rows**. **Pilot 1:** 7 of 46 donor days came from insiders (departments 12 and 21), and 80 of its 197 pairs remain label-valid. **Under corrected eligibility (non-insider donors only), 225,888 of Package 4's 530,792 candidate rows are invalid.** No historical tuple is exactly reusable for the corrected claim, because the dictionary is refit. | `tuple_inventory.json`, `pkg4_benign_donor_breakdown.json`, `historical_pilot1_eligibility.json` |
| F10 | provenance | The repaired-3B delta indices are **label-selected subsamples** (5,396 r4.2 / 7,074 r6.2 rows per seed). The audit's missing counts reproduce: 211 r4.2 and 11 r6.2 corrected positives per seed. Their negative pools were drawn with the old labels, so corrected replication (C22) needs a re-drawn pool, not only an extension. | `repaired3b_coverage_recheck.json` |

Targeted tests at `c31efcf`: `test_cert_calendar_recovery.py` 18 passed;
`test_eval_metrics_core.py` all checks passed (`targeted_tests_at_c31efcf.txt`).

## Claim status after this review (cached-analysis scope only)

**Cleared for the narrow cached-analysis scope** (numbers may be stated as cached-score results
with their population, estimand and scope; manuscript text still pending C44):

| claim | scope |
|---|---|
| C01 | corrected counts and target definitions |
| C02 | main adapters' training exposure is unaffected by this bug |
| C05, C06, C07 | LM detector metrics only; C06/C07 need the F5 wording; comparator rows still blocked |
| C08 | within-user discrimination on cached scores |
| C09 | masking, with the F3 caveat |
| C28 | factorial reanalysis, with the new 3B wording; single adapter seed |
| C31 | r6.2 behavior-only and target-mask contrast |
| C33 | primary cached comparison; history-shuffle still pending; F6 disclosure |
| C34 | fixed-sample reanalysis only |

**Restricted, not complete:** C32. The r4.2 behavior-only result covers 60 of 70 positive
users; extension is in Stage B.

**Still blocked:**
- Label-dependent SAE fit, ranking, controls, receivers or donors: C10–C27, C29, C30, C35–C39.
- Classical and older sequence comparators: C27, C45, C46.
- Outside this bug: C40–C43. Their existing scopes and issues are unchanged.
- C44: manuscript synchronisation.

## Corrections proposed from the review

1. Replace "unresolved" in the recovery report and the claim notes with F2's cause. Require
   same-configuration baselines for every intervention. Declare batch size 1 at fp32
   cross-entropy as the scoring reference, since it is the only configuration shown
   reproducible bit-for-bit across runs.
2. Add F3, F4 and F5 wording to the C06, C07 and C09 notes, and the F6 margin to C33.
3. No code defect was found in the recovery modules, so no code fix is made. Stage B adds a
   regression check for F2's reference configuration: the bridge must be bit-exact.

## Smallest next causal pilot (Stage B/C)

See `STAGE_B_PROTOCOL.md` in this directory. It freezes the scoring reference, coverage repair,
dictionary protocol, selection and pilot before any new effect is seen.
