# Issue ledger — review of 2026-09-24

Each finding was re-verified against the artifacts before anything changed.
"Verified" below names the property checked, nothing broader. Original outputs
are kept; corrections are dated notes beside the original text.

Commits: `d6516b3` (code, tests, TWOS plan) and the commit that adds this file.

---

## T1. TWOS token-class runs selected features from an undefined ranking

**Finding (review).** Both runs patched `top5 = [0, 1, 2, 3, 4]`; their ranking
files have no finite `row_gap`.

**Verified.**
- The ranking files `twos_work/v3_sae_s42|s43/layer_24/m04_k08/delta_sae_top_features.csv` (Aquaman) have 8,192 rows and 0 finite gaps.
- The saved selection for seed 42 is `top5 = [0, 1, 2, 3, 4]` against `control5_active = [505, 5342, 561, 4137, 3967]`, with a blank mean gap. Seed 43 saved no selection file. Its rows show the same `[0, 1, 2, 3, 4]`.
- The pre-fix `sae_core` (repo HEAD before `d6516b3`), run on the real seed-42 file, returns exactly those two sets and raises nothing.
- Of the 72 saved selection files under `results/`, only this run's records an undefined top set.

**Root cause: the wrong ranking population, not missing labels.**
1. `train_delta_sae_frontier.py --benign-only` fits the SAE on benign rows, which is legitimate. It then ranked features on that same benign-only matrix, so the population had 0 positive rows. It still wrote `delta_sae_top_features.csv`.
2. The pipeline's re-ranking step was correct. `reselect_l24` used 14,609 positive rows, and the original TWOS intervention `v3_causal_s42` read it (`top5 = [6036, 7375, 3218, 417, 22]`, mean gap 0.078).
3. The 2026-09-19 re-runs (`lm/run_twos_pkg4.sh`, `..._s43.sh`) pointed `--frontier-dir` at the benign-only frontier instead. Their comment claimed full parameter identity. They also differed in batch sizes and omitted `--exclude-same-user-donors`. The seed-43 comment cites a `v3_causal_s43` run that never existed.
4. Nothing refused NaN. Sorting NaN gaps returns ids in order. With `|gap|` undefined, "low-gap" controls fell through to "most active" (2.0% to 2.2% active, against 0.0% to 1.0% for the "selected" set).

**Affected.** Both TWOS token-class result directories and their READMEs. The
P-hacking audit's "prediction confirmed". The failure-mode audit note. The
LANL README's seed caveat. The thesis limitation that cited a "seed reversal".
The package-4 pre-registration's motivation.

**Fix.**
- `sae_core` now provides a population gate, a ranking validator and a guarded loader. Selection refuses non-finite statistics, and missing gaps are never set to zero.
- The benign-only frontier writes activity statistics and a `RANKING_NOT_COMPUTED` marker, not a ranking.
- `reselect` requires positive rows from at least 2 discovery users. It can also drop held-out users' benign rows, and it writes a full manifest.
- The causal eval writes `feature_selection_manifest.json` and refuses receivers that overlap the discovery users.
- The seven scripts that read a ranking file go through `load_ranking`. The four that compute rankings validate their population and their output. `population_subsample_attribution` no longer ranks by −benign mean when it has no positives.

**Validation.** `tests/test_feature_ranking_gates.py` has 39 checks and passes on
CPU (Aquaman, torch 2.5.1). It reproduces the failure, pins a known ordering,
and runs the `reselect` command end to end. All 8 test files pass.

**Remaining uncertainty.** The checks prove the gates reject these conditions.
They do not prove every historical ranking used an appropriate population.
Rankings whose discovery set was every positive user are finite but in-sample
(see T2).

## T2. The original TWOS intervention was valid but not held out

`v3_causal_s42` used a finite ranking, but it selected features on all 16
positive users and evaluated the same users. Status: exploratory, in-sample.
The held-out test is the corrected run, job 20890658 (`docs/TWOS_CORRECTED_INTERVENTION_PLAN.md`).

**Held-out result (job 20890658, 2026-09-24).** Inconclusive on 8 confirmation
users: no selected-versus-control difference is detectable on behaviour or
profile loss. Most of the raw alpha-1 profile contrast came from the controls
reconstructing more receiver days. `results/twos_corrected_bounded/README.md`.

## T3. The TWOS "seed reversal" interpretation

**Withdrawn.** The two runs differ measurably, but both used invalid
selection. Seed sensitivity cannot be separated from the selection defect, so
the runs are not evidence that correctly selected features are unstable.

**Also false on its own terms.**
- "Every sign reverses": the behaviour contrast is negative at both seeds, −0.00011 and −0.00243.
- At seed 43 both arms raise behaviour loss, selected +0.00176 and control +0.00419. The selected patch worsens it less, it does not "improve" it.

## T4. Controls meet a threshold, not activity matching

- CERT staged ranking: controls fire on 0.36% of tokens on average, selected features on 0.18%. The controls' gaps are |g| ≤ 1.2e-4, which meets the "small gap" rule.
- Corrected TWOS ranking: mean activity ratio 1.03, which happens to be close.

`feature_set_criteria` now reports the ratio and labels the criterion
`threshold_only`. Corrected wording is in the thesis, the pre-registration
addendum and the confirmation `RESULTS.md`. `select_matched_controls.py`
exists for a matched alternative. The queued run does not use it.

## L1. LANL user sampling was coupled to the seen/unseen split

**Verified on the saved `eval.jsonl` with an independent script.**

| pool | windows | positives | users | with an attack window | negative-only |
|---|---|---|---|---|---|
| seen | 22,563 | 234 | 1,063 | 75 | 988, all in fold 0 |
| unseen | 30,410 | 198 | 23 | 23 | 0 |

- All 30,212 unseen negative windows come from attack users.
- Training has 300,729 windows over 1,254 users: 1,180 ordinary users, all with `md5 % 10 == 0` and all in fold 0, plus 74 attack users.
- No unseen user is in training.
- The original job (`lanl2015/windows_full.sbatch`) ran `--sample-mod 10` with md5, and the split used md5 folds.

**Root cause.** `md5 % 10 == 0` implies `md5 % 5 == 0`, so every sampled
ordinary user landed in fold 0.

**Affected.** All four LANL conditions (they share the split), their adapters
and deltas, `results/score_decomposition/lanl_full`, the paper's LANL
subsection, abstract, introduction, conclusion and claim-status row, and the
audits' LANL rows.

**Fix.** `lanl_hashing.py` uses independent salted SHA-256 hashes for sampling
and for folds, keeping legacy md5 only behind explicit flags. The ETL and split
use it. The split now writes `population_report.json` and exits non-zero
unless both pools contain attack users and negative-only users, the folds are
not degenerate, and no unseen user is in training.

**Validation.** `tests/test_lanl_population.py` has 12 checks. It reproduces
the coupling, and the real split script refuses the legacy configuration and
passes a salted one.

**Descriptive only (original populations; seen users were trained on).**
- Restricting seen negatives to attack users, the unseen pool's composition, gives full-score AUC 0.916, down from 0.949.
- Mean within-user AUC is 0.883 for seen users (70) and 0.738 for unseen users (23).

Composition explains little of the pooled gap. Unseen attack windows still rank
above the same users' benign windows.

**Retaining all red-team users.** Ordinary users are sampled at 1 in 10 but
every attack user is kept, so attack users are over-represented about tenfold.
A pooled AUC then mixes attack-versus-benign discrimination with
red-team-account-versus-ordinary-account discrimination. The repaired
evaluation reports pooled and within-user AUC, with composition matched.

**Repair options.** The raw data is still on disk:
`/anvil/scratch/x-sangdembay/lanl2015/auth.txt.gz`, 7.6 GB, and `redteam.txt.gz`.
Scratch is subject to purge, so copy it before relying on it.
- **A. No retraining.** Extract a salted sample of ordinary users who were never sampled (`md5 % 10 != 0`), so they were never trained on, and check them against `train.jsonl`. Score them with the existing adapter. Build an unseen pool from them plus the 23 fold-4 attack users.
- **B. The intended design.** Rebuild windows with salted sampling and folds, retrain, and rescore.

Relabelling already-trained users as "unseen" is excluded.

## L2. LANL interpretation claims

- **Withdrawn.** "Learns anomalous identities", "memorisation", and the seen/unseen gap as transfer failure. Training is benign-only, identity-token loss is context-dependent, and user and host fields can carry legitimate relational behaviour.
- **Corrected.** "AP near the base rate everywhere" was wrong. Seen AP is about 17 times its prevalence and unseen AP about 2 times.
- **Retained.** The AUC and AP values themselves, as descriptions of the original pools.

## C1. CERT r4.2 score masking: retained, scoped

- User AUC 0.6525 full against 0.8633 behaviour-only, difference +0.2108 [+0.157, +0.266]. 51 users improve, 3 tie and 6 decline. The external review recomputed it.
- The score removes profile tokens' direct contribution, not their influence, since the profile stays in context.
- Intervals condition on one adapter and a fixed benign cohort. They are not seed uncertainty, and not external validation.

## C2. Phase C 3B: retained, scoped; one verification argument withdrawn

- Behaviour-only scoring gives 0.9440 with ordinary training and 0.9415 with the training-loss mask. The difference is −0.0025 [−0.0148, +0.0074].
- Scope: one training seed and four malicious users. This says nothing about 8B, and nothing general about training-loss masking.
- **Withdrawn.** V39's claim that identical class counts "show the token sequences match". Counts can agree while sequences differ. The review compared 256 records and found 0 token-ID mismatches, with differing backend digests.

## C3. CERT package-4 analysis choices

Clarifications were appended, dated, to `docs/PREREGISTRATION_CERT_PACKAGE4.md`
while the jobs were pending. They cover:
- donor types kept separate, with benign donors primary;
- each context mode separate, none privileged;
- per receiver, the mean over all candidate donors;
- users weighted equally;
- α = 1 fixed;
- the direct profile-minus-behaviour contrast;
- a ±0.0002 TOST margin before any "negligible" claim;
- both arms and the base always reported;
- the missing α = 0 reconstruction control;
- the environment gate;
- fresh-base fields.

These choices are pre-specified relative to the per-class outcomes only, not
independently confirmatory. The pre-registration's prediction cited the invalid
TWOS result, so that motivation is void. Tools:
`analyze_intervention_endpoints.py` (known-answer self-test) and
`compare_intervention_runs.py` (self-test).

## C4. Environment portability

No file on disk states that computing both arms together guarantees
cancellation. Wherever that was argued, it is withdrawn. Two gates replace it.
- **CERT.** The collaborator smoke 20879548 is compared row by row with the owner-environment smoke 20840686. Tolerances: 1e-4 on the contrast and 1e-3 per row.
- **TWOS.** Token IDs for all 2,275 records are identical between the extraction environment (Aquaman, transformers 5.14.1) and the run environment (collaborator, 5.16.1). The Aquaman environment may have changed since the August extraction, and it records no version.

**Update (later on 2026-09-24).**
- *The claim was found.* It is in `scripts/submit_r42_token_class_causal_collab_anvil.sh`, lines 13 to 16 (commits `49424f6` and `67f37c0`, 2026-09-21/22): "Base and patched scores are computed within one run, so the CONTRAST is unaffected". The first search missed it because a zsh glob with no matches aborted the whole grep command. The comment now carries a dated correction beside the original text. The staged copy used to submit the queued job is left untouched; the job does not read it at run time.
- *The CERT environment gap is a major version.* The owner environment, where the CERT deltas and SAE were produced, runs transformers 4.51.3. The collaborator environment runs 5.16 with torch 2.5.1 and PEFT 0.20. This is why the smoke-versus-smoke gate matters.
- *The TWOS extraction environment is now reconstructed.* `~/cert-venv` on Aquaman was created on 2026-07-22, and every relevant package was installed in the same minute: transformers 5.14.1, tokenizers 0.22.2, torch 2.5.1+cu121, PEFT 0.19.1, bitsandbytes 0.49.2, accelerate 1.14.0. None has been replaced since, and `run_v3_seed42.sh` activates this environment. Two artifacts corroborate it. The adapters written by the same script on 2026-08-25 record PEFT 0.19.1 in their own files. And the token rows saved at extraction match today's tokenizer count for all 2,275 examples, with positions contiguous from 0 to n−1. Counts are consistent with identical token IDs but do not prove them; the ID-level identity was shown between today's Aquaman environment and the run environment.
- *Still open.* Numerical differences between Aquaman (RTX 3070, transformers 5.14.1) and Anvil (A100, 5.16.1). The TWOS run's α = 0 arm absorbs them together with reconstruction. A small δ-reproduction job would separate the two; it is proposed in the experiment plan, not run.

**Gate result.** The CERT environment gate PASSED on 2026-09-24. The collaborator smoke
(job 20879548; transformers 5.16.1, PEFT 0.20.0) and the owner smoke (job
20840686; transformers 4.51.3, PEFT 0.13.2) match on all 1,186 rows (same
receivers, donors, arms and alpha; both torch 2.5.1 on A100-SXM4-40GB). The
largest per-row difference in any view's delta is 9.5e-07 nats (tolerance
1e-3), and every selected-minus-control contrast agrees within 6e-09
(tolerance 1e-4). Report: `results/cert_package4_env_gate/GATE_REPORT.txt`.
This verifies that the environment change does not move these smoke rows
(team, alpha 1, 24 receivers). It is not a check of the other context modes
or alphas, which only the full run contains. The smoke's own contrasts are a
validation subset and are not read as results.

## C5. Cached-baseline outputs

- **Verified in code.** In runs made with `--token-class-schema` (`base_used`), `delta`, `repair`, `strong_repair`, the best rows and the summaries use the fresh base.
- **Cached base.** Job 19379904, the TWOS v3 runs and the 2026-09-19 TWOS token-class runs use it for those fields. For the latter this holds in all 35,328 rows, with 8,162 and 8,594 repair-sign disagreements.
- Their per-class `delta_<view>` columns use the fresh base.
- A difference-in-differences cancels a per-receiver base, but absolute deltas and repair flags from those outputs are not comparable.

## C6. The published endpoint's best-candidate selection

Found while correcting the thesis. `summarize_best` keeps, per receiver, the most
negative delta over up to 16 donors and 4 alphas. That favours the arm whose
candidate scores vary more. Anomalous donors can be fewer than benign ones (for
example, 1,012 against 1,088 candidate pairs in the corrected TWOS run's dry run), so τ can be positive
without a donor-specific effect. The base and the reconstruction do cancel within
each arm's donor difference, because both donors patch the same tokens. The
new analyses average over all candidates. The published 0.000758 is unchanged
and is labelled with this caveat.

## H1. Thesis endpoint mismatch

**Verified.** In the dept summary (605 complete receivers), the benign-only
contrast `δ_C,ben − δ_F,ben` is 0.002758. The published endpoint is the
difference-in-differences, 0.000758, with cluster CI [0.000346, 0.001208].

**Fixed.** The equation, donor comparison, best-candidate step, complete-case
aggregation, number and interval now all describe τ. The worked example was
redone. The benign-only quantity is shown as a different number with no interval.

## H2. Thesis claims beyond the evidence

**Fixed.**
- "The selected features read behavior" became "Where the selected features activate".
- Location, represented information and intervention effects are now separated explicitly.
- The abstract no longer anticipates the pending per-class result.
- The "fire comparably often" controls and the "seed reversal" limitation are corrected.
- The LoRA analogy is marked as structural only.
- Score inclusion is separated from context conditioning.
- The thesis compiles with 0 errors and 0 undefined references.

## Status

- **Supported, with scope.** C1, C2 and the location table (T4 corrects its control wording).
- **Withdrawn or qualified.** T1 and T3 (TWOS per-class and seed reversal), L1 and L2 (LANL seen/unseen interpretation), C2's tokenizer argument, the confirmation README's "confirmed" and "never seen by configuration search", and the thesis wording in H1 and H2.
- **Completed repairs.** Code gates and tests (T1, L1), analysis tools (C3, C4), manuscript and README corrections (H1, H2, L2, T3).
- **Pending.**
  - TWOS held-out run 20890658.
  - CERT smoke 20879548, then the environment gate, then full run 20879549.
  - CERT α = 0 reconstruction run.
  - LANL repair A or B.
  - See `docs/EXPERIMENT_PLAN_2026-09-24.md`.
