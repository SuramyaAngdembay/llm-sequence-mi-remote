# Progress record — score decomposition and behaviour-only scoring

Kept deliberately separated into what has been **verified**, what is a
**hypothesis**, and what is an **unresolved limitation**. Results are appended
as they land; nothing here is written into the paper before it is checked.

Last updated: 2026-09-18 (Phase A/B complete with results; Phase C 3B queued).

---

## Verified observations

Facts established by reading the tracked code/artifacts or by a check that was
actually run.

**V1 — no per-token or per-class losses are cached anywhere.**
`extract_adapter_deltas.per_example_nll` returns a sequence average; the saved
rows carry `base_nll`, `adapted_nll`, `delta_nll`, `n_tokens` only. Confirmed
across every `deltas_*/example_scores.parquet` on the cluster. The
decomposition therefore requires new inference. The earlier "CPU-only / zero
SU" claim was wrong.

**V2 — the naive subtraction is not a behaviour score, and can reverse a
ranking.** `s_full − s_profile = (N_B/N)(s_B − s_P)`. With N_P=10, N_B=90:
(s_P,s_B)=(10,2) gives naive −7.20 while (1,1.5) gives +0.45, so the naive
statistic ranks the first example *lower* although its behaviour score is
higher. Checked in `scripts/tests/test_token_class_decomposition.py` (check 1).
The implementation accumulates behaviour targets directly and asserts the
algebraic identity as a cross-check.

**V3 — line-index class assignment is wrong for three of the four factorial
conditions.** `build_session_jsonl_fast.py` drops the `PSY ` line under
`no_psy` and both `PSY `/`DAY ` lines under `no_profile`, so line 1 is PSY in
`full` and SESCOUNT in `no_psy`, and line 0 is DAY in `full` and SESCOUNT in
`no_profile`. The existing `feature_token_attribution.py` maps by line index
and would mislabel those conditions. The new library maps by line **prefix**;
checked for all three conditions (check 5).

**V4 — exact population of the scored pool** (identical for 3B and 8B; the
factorial pools are byte-identical in size and token counts):
159,064 day-rows, 410 users, 70 positive rows. Positive users ACM2278 (5
positive of 176 days), CDE1846 (46/356), CMP2946 (18/316), MBG3183 (1/356).
406 benign users = 405 `val`-split + `PLJ1771` (answer-key user with no
positive day in the matched domain). Token counts: mean 293.2, median 234,
max 1187 — **no example is truncated** at `max_seq_len` 2048.

**V5 — every user in this pool was excluded from adapter training.** Positive
users are reserved for `eval` by `assign_split`; the benign comparison users
are the validation split. The comparison is user-disjoint on both sides. It is
**not** a seen-user attack-detection test and must not be described as one.

**V6 — verified compute balances (2026-09-17, not historical):**
`cis260991-gpu` 400.0 SU (A100), `tra250034-gpu` 95.2 SU (A100),
`tra250034-ai` 732.5 SU (H100), CPU 1000.1 / 17,173.3 SU. Billing is 1 SU per
GPU-hour. Scoring runs on `cis260991-gpu`, preserving the H100 balance for
Phase C.

**V7 — the decomposition library passes 47 focused checks** (naive-formula
failure, partition reconstruction, padded-batch == single-example,
off-by-one detection, boundary-crossing tokens, truncation, the no_psy /
no_profile line shift, DAY_WEEK ⊂ DAY, LANL field spans, and the Phase-C loss
masking arithmetic).

**V8 — Phase C is one new training run per scale per seed, not four.** Cells A
and B are the existing `full` adapter scored two ways; C and D are one
profile-target-masked adapter scored two ways. Implemented behind two flags
whose defaults reproduce the current recipe exactly.

**V9 — class mapping validated against the real tokenizer and real data**
(CPU only on the login node, 400 examples per condition, no allocation cost):

| condition | DAY | PSY | SESCOUNT | SES | profile share p |
|---|---|---|---|---|---|
| full (3B and 8B, identical) | 0.0885 | 0.0601 | 0.0235 | 0.8279 | **0.1485** |
| no_psy | 0.0941 | 0 | 0.0250 | 0.8809 | 0.0941 |
| no_profile | 0 | 0 | 0.0246 | 0.9754 | 0 |

Also measured: **zero** boundary-crossing target tokens and zero tokens whose
class would change under end-assignment instead of start-assignment — the
attribution ambiguity is empirically nil for this serialization, not merely
assumed small. Zero OTHER and zero SPECIAL targets, so `n_targets = n_tokens −
1` exactly. No example truncated. `DAY_WEEK` is 0.0075 of targets, entirely
inside DAY (so `week=` is ~8% of the DAY class: removing DAY removes a small
temporal cue as well as static attributes). The `no_psy` / `no_profile`
conditions show exactly the line shift that would break index-based mapping.

Consequences: the SESCOUNT choice moves 2.35% of targets, so the primary
(includes SESCOUNT) and secondary (SES only) behaviour views are genuinely
different definitions and both are reported. For Phase C, p = 0.1485 means the
HF-default denominator would up-weight each behavioural token by
1/(1−p) = 1.174 — which is why `--loss-denominator all_targets` is specified
for cells C/D.

**V10 — the LANL field schema matches real windows.** On real LANL eval
windows: ID_USER 0.201, ID_HOST 0.201, BEHAV 0.401, HOUR 0.100, OTHER 0.097 of
spans. OTHER is entirely the `" | "` event separator, which the LANL
`behavior_only` view scores as non-identity; this is now stated in the library
rather than left implicit. The LANL *scoring* path has still not been run.

**V11 — the scorer reproduces the cached detector score exactly when hardware
and batch size match; the 8B discrepancy is GPU architecture, not code.**
First scoring attempt (jobs 20807957/20807958, both A100) ran a 256-example
pilot at batch 1 and at batch N against the cached `example_scores.parquet`:

| pilot | hardware then / now | mean \|ΔNLL\| | max \|ΔNLL\| | rank corr |
|---|---|---|---|---|
| 3B, batch 1 | A100 → A100 (matched) | **1.5e-08** | **5.8e-08** | 0.9999999999999998 |
| 3B, batch 16 | A100 → A100 | 1.5e-03 | 9.6e-03 | 0.9940 |
| 8B, batch 1 | **H100 → A100** | 2.9e-03 | 1.0e-02 | 0.9855 |
| 8B, batch 8 | **H100 → A100** | 2.7e-03 | 1.4e-02 | 0.9863 |

The 3B batch-1 row is the decisive one: identical code, identical hardware,
identical batch size reproduces the cached score to 6e-8, so the implementation
is exact. The original 3B extraction ran on the `gpu` partition (A100) and the
original 8B extraction on `ai` (H100) — hence the 8B rows drift by ~3e-3 at
*any* batch size, which is bf16/4-bit kernel difference across architectures,
not a code defect. Batching adds drift of the same order.

Structural checks were perfect in every pilot: partition sums to the total to
2.3e-13, zero partition-count mismatches, `n_targets = n_tokens − 1` for every
example, **zero** boundary-crossing targets, zero OTHER/SPECIAL, and class
shares identical across batch sizes (DAY 0.0888, PSY 0.0605, SESCOUNT 0.0237,
SES 0.8271 on the pilot subset; DAY_WEEK 0.0073).

The first gate correctly refused to spend on a full run it could not anchor.
Its 0.999 rank-correlation threshold was, in hindsight, a cross-architecture
criterion applied to an arithmetic question. The rerun (jobs 20814766 on `ai`
H100 for 8B, 20814767 on `gpu` A100 for 3B) scores at batch 1 on the hardware
that produced each cached baseline, keeps the structural checks hard, and
bounds numerical agreement at mean \|ΔNLL\| ≤ 1e-2 and rank corr ≥ 0.98; the
full run then reports the recomputed `full` view and the published cached score
side by side so the drift's effect on the actual metric is measured rather than
assumed.

Throughput measured: 8B 8.1 ex/s at batch 1 and 16.7 at batch 8 (A100, 11.3 GB
peak); 3B 9.1 at batch 1 and 30.3 at batch 16. Full pool at batch 1 is
therefore ~5 h per scale, ~5 SU each.

**V12 — the Phase C gate caught a 4x loss-normalization error before any
training SU was spent.** Job 20810414 trained 200 steps with **nothing masked**
under the HF-default loss path and under the custom fixed-denominator path;
with nothing masked the two are arithmetically identical, so they must agree.
They did not:

| step | default path | custom path | ratio |
|---|---|---|---|
| 1 | 2.7574 | 11.0802 | 4.02 |
| 20 | 1.2885 | 5.1953 | 4.03 |
| 40 | 0.2557 | 0.9886 | 3.87 |
| 200 | 0.1357 | 0.5206 | 3.84 |

The ratio is `gradient_accumulation_steps` (4 for the 3B recipe). Confirmed in
the installed source — `transformers` 5.16.1 `Trainer.training_step` contains

```
if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    loss = loss / self.current_gradient_accumulation_steps
```

so when `num_items_in_batch` is supplied the trainer skips its own division and
expects `compute_loss` to have normalized over the **whole accumulation
window**. The original implementation normalized per micro-batch, making the
loss — and therefore the gradients and the effective learning rate — 4x too
large. Had it run, cells C/D would have differed from A/B because of the
learning rate, not the objective.

Fix: stop masking `labels`. Labels stay unmasked (only padding is −100) so the
trainer's `num_items_in_batch` counts **every non-pad target**; a separate
`profile_mask` column is padded alongside the batch and applied inside
`compute_loss`, which divides by `num_items_in_batch`. Then

* nothing masked -> `sum(all target losses) / N_all` ≡ the HF default (the gate);
* profile masked -> `sum(behaviour losses) / N_all`, i.e. each retained token
  keeps exactly the weight it has in the unmasked recipe.

`scripts/tests/` check 8 pins this (micro-batch normalization inflates by
~grad_accum; `scored_targets == all_targets / (1-p)`).

**V13 — measured 3B training rate.** The gate's own 200-step runs report
`train_samples_per_second` 7.376, so one epoch of 300k examples is ~11.3 h on
one A100 (~11 SU), not the ~7 h estimated earlier. Phase C at 3B is therefore
~11 SU of training plus ~5 SU of batch-1 scoring, and training is split into
its own job rather than sharing a wall clock with scoring.

**V14 — Phase A/B result: on the audited 8B adapter, removing the profile's
direct score contribution moves unseen-user ranking from chance to 0.938.**
Jobs 20814766 (8B, H100) and 20814767 (3B, A100), batch 1, full pool. The
recomputed `full` view reproduces (see `results/score_decomposition/README.md`) the published `adapted_nll` over all 159,064
rows to max \|ΔNLL\| 1.4e-07 (rank corr 0.9999999999977), so the baseline is
anchored exactly. Full tables in `results/score_decomposition/`.

| 8B, mean of 4 folds | day ROC | day AP | user ROC | rank /407 | within-user ROC |
|---|---|---|---|---|---|
| full (published score) | 0.399 | 0.0001 | 0.532 | 191.0 | 0.751 |
| profile_only | 0.213 | 0.0001 | 0.200 | 326.0 | 0.588 |
| **behavior_only** | **0.836** | **0.0009** | **0.938** | **26.3** | 0.680 |

Paired bootstrap: user ROC +0.406 [+0.163, +0.681]; all four folds improve
(+0.071, +0.805, +0.438, +0.308). Per user 0.919→0.990, 0.155→0.961,
0.527→0.966, 0.527→0.835. At 3B, where `full` already reaches 0.926, the same
change gives +0.018 [−0.086, +0.123] (three of four folds up, MBG3183 down).

The mechanism is visible in the exact mean-gap decomposition (reconstructs to
5e-17): at 8B, PSY contributes **negatively** to the positives-minus-benign gap
for three of four users and SES positively for all four, with the total
negative for three — which is why the full score's day ROC is *below* chance.
`profile_only` is below 0.5 for every user individually. The adapter finds
these malicious users' profiles less surprising than unseen benign users'
profiles, and that term buries a behavioural signal that is in fact strong.

Bearing on H1: score inclusion is a material pathway in the audited 8B run.
This is a scoring intervention on a frozen adapter with the profile still in
context; it does not remove profile influence on behavioural predictions and
says nothing yet about the training objective.

**V15 — the same result's limits, measured rather than assumed.**
(a) Average precision stays near the base rate: 8B day AP 0.0001 → 0.0009 at
prevalence 3.2e-05, so alert precision remains unusable despite the ROC move.
(b) At a 0.1 % false-positive budget the gain is exactly zero in every fold; at
1 % it is +0.044 [0.000, +0.088] from two of four folds. (c) Within-user
ranking, with identity held fixed, gets *worse*: 8B 0.751 → 0.680, 3B
0.739 → 0.568 — the profile contribution helps there. One untested explanation
is that within a user the profile text is constant, so its share of the mean
varies only through N_profile/N_total, which tracks the number of session lines
that day; the full score may partly encode day length. (d) Signed class
contributions cancel, so shares exceed 100 % and flip sign (CMP2946 at 8B: PSY
+382 %, SES −406 %); this is an accounting identity for the mean gap, not an
AUC decomposition and not a mechanism.

---

## Hypotheses (not yet tested)

**H1 — direct score inclusion is a material part of the failure.**
*Status: supported for the audited 8B run* (V14): user ROC 0.532 → 0.938 with
the adapter and input untouched, all four folds improving. Scope: one adapter,
four malicious users, exploratory population, and a scoring change only. It
does not follow that the adapter has stopped using identity, nor that this
transfers to r4.2, LANL, or another detector — that is the portability question
the decision rules put next.

**H2 — being trained to predict profile tokens is a separate harm.** Tested by
B → D in Phase C. Masking targets does not stop behavioural losses from
training the adapter to *use* profile context, so a null result would not show
that identity information is absent.

**H3 — context conditioning is the residue.** Whatever survives both
interventions is profile influence on behavioural predictions. No experiment
here removes it; claiming otherwise would be unsupported.

**V16 — portability runs are set up with their own protocols, not CERT's.**

*LANL* (job 20826171): the `full` adapter, scored with the **field-level**
schema (`su`/`du` -> ID_USER, `sc`/`dc` -> ID_HOST, `at`/`lt`/`or`/`res` ->
BEHAV, bare `t<hh>` -> HOUR, `" | "` separators -> OTHER). Seen/unseen
membership is **read from `eval.jsonl`**, which `lanl_split.py` writes as
`fold`/`seen` fields (md5 leave-users-out, fold 4 unseen) — never recomputed.
Evaluation is window-level AUC/AP within each pool with a cluster bootstrap
over users, anchored to the published `ap_full.json` (seen AUC 0.9489 / AP
0.1769; unseen AUC 0.4966 / AP 0.0133). 52,973 windows, 432 positive, mean
1,256 tokens — ~4x CERT's tokens per example at a third of the examples.
The published `host_anon` condition (unseen AUC 0.6266) is the retrained
comparator, the LANL analogue of `no_profile`.

*r4.2* (job 20826196): the **headline** 8B adapter
(`qwen3_8b_session_qlora_r42_ddp_mb22_gc_on`), a separate line of evidence from
the r6.2 factorial and not to be mixed with it. Pool = all days of the 60
malicious users plus all validation-split benign days = 40,519 rows, 139 users,
1,309 positives, which is the user-disjoint population behind the published
day ROC 0.668 / user ROC 0.565. Sixty malicious users means the cluster
bootstrap has **60 clusters here, against four on r6.2**.

One caveat specific to r4.2: its cached scores came from a **batch-56 forward
pass** (`score_adapter_examples.py` chunks only the cross-entropy, not the
forward), so the exact reproduction available for the r6.2 factorial is not
available here; ~1e-3 drift is expected and the gate tolerates it. The
scientific comparison is within one forward pass, so the drift cancels. The
reference join now keys on `example_id` rather than row position.

**V17 — the r4.2 cached baseline is a batch-56 artifact, so it is not a
reproduction target.** Job 20826196's gate refused the run at mean \|ΔNLL\|
0.0130 / rank corr 0.9608 against the cached scores. Diagnosis on the pilot:

* model loading is **identical** — `score_adapter_examples.py` uses the same
  `BitsAndBytesConfig`, the same dtype and the same `PeftModel`;
* token counts match exactly (0 mismatches), and every structural check is
  clean (partition 1.1e-13, `n_targets = n_tokens − 1`, zero boundary targets);
* the signed difference **tracks sequence length**: corr(diff, n_tokens) =
  0.388, with the shortest length quartile at −0.0093 and the longest at
  +0.0046.

That is the padding signature. `score_adapter_examples.py` chunks only the
cross-entropy (`loss_batch_size`), not the forward pass, so the cached r4.2
scores came from a **batch-56 forward** in which short sequences sat beside
long ones. The rerun uses batch 1, which has no padding at all, so the new
numbers are the *cleaner* estimate and the cached ones are not a gold standard.

Consequence for the protocol: for r4.2 the structural checks stay hard, cache
agreement is reported rather than gated, and the anchor that matters is
metric-level — the recomputed `full` view's pooled user-disjoint ROC against
the published day 0.668 / user 0.565, printed side by side with the cached
scores' own ROC. This does not weaken the comparison of interest: every view
comes from the same batch-1 forward pass, so any batching effect is common to
all of them.

**V18 — cluster maintenance blocks the long jobs.** Reservation
`anvil-maint-2026-q3` runs 2026-09-18 23:30 → 2026-09-20 21:00 across all
nodes (`Flags=MAINT,IGNORE_JOBS`). Jobs whose wall clock cannot fit before
23:30 are deferred past it (Slurm's first estimate for the 8 h LANL job was
2026-09-24). Adjusted: r4.2 resubmitted with a 2 h wall (~1.3 h of work) and is
now pending on `Priority`, i.e. schedulable before the window; LANL resubmitted
at 5 h and the Phase C 3B training (11.3 h) both wait for the window to close.
The Phase C runner auto-resumes from checkpoints, so a maintenance kill would
cost time rather than work.

**V19 — the factorial audit pool double-counts 12 % of its benign days; the
conclusions are unaffected.** Checked after an external suggestion to
deduplicate. The pool is built as `eval.jsonl` + a 12 % sample of `val.jsonl`,
but `val.jsonl` is a strict **subset** of `eval.jsonl` (verified: 140,711 ids,
all present in `eval.jsonl`'s 142,072). So 16,992 validation days appear twice
and carry double weight: 159,064 rows, only 142,072 distinct.

This is inherited from the factorial runners, so it also affects the published
factorial numbers. Impact, measured rather than assumed:

| | 8B day ROC | 8B behaviour-only day ROC | 8B user ROC | 8B behaviour-only user ROC |
|---|---|---|---|---|
| with duplicates | 0.3992 | 0.8357 | 0.5320 | 0.9378 |
| deduplicated | 0.3994 | 0.8357 | 0.5320 | 0.9378 |

User-level metrics are **identical by construction** — max-aggregation over a
user's days is idempotent under duplication — so every user-level number in the
paper and in V14 is unaffected. Day-level metrics move in the fourth decimal.
`eval_score_views.py` now deduplicates on `example_id` and records how many rows
it dropped; `results/score_decomposition/*_dedup/` holds the reruns.

**V20 — the r4.2 padding explanation is now demonstrated, not inferred.** V17
argued from a length correlation over a narrow token range, which is
suggestive only. Job 20827737 (2 min 21 s, ~0.05 SU) rescored the same 256
examples at three forward batch sizes and compared each against the cached
(batch-56 forward) scores:

| scoring batch | mean \|ΔNLL\| | max \|ΔNLL\| | rank corr |
|---|---|---|---|
| 1 | 1.295e-02 | 3.560e-02 | 0.9608 |
| 8 | 1.060e-02 | 3.222e-02 | 0.9520 |
| **56** (matches the cache) | **2.948e-03** | **2.029e-02** | **0.9899** |

Matching the cached run's forward batch cuts the mean disagreement **4.4x** and
lifts rank correlation to 0.990. `mean` and `max` move monotonically with batch
size; rank correlation does not (batch 8 sits just below batch 1), which on 256
examples is within noise and is reported rather than smoothed over. The
residual 2.9e-3 at batch 56 is expected because batch *composition* still
differs — the cached run batched over the full `all.jsonl` in file order, this
one over a 40,519-row subset — so padding patterns are similar, not identical.

Conclusion, stated at the strength the control supports: the disagreement is
**batching-dependent**. Matching the cached forward batch removes most of it,
so the cause lies in how the forward pass was batched rather than in the
decomposition. The control does **not** isolate padding specifically — batch
shape, kernel selection, padding and batch composition vary together, and
composition was not held fixed while varying only the mask. It also does not
show that every score view is equally insensitive to batching; that would need
the views themselves compared across batch sizes, which was not run. Batch 1
remains the reported basis as the no-padding computation; the cache is not a
reproduction target for r4.2.

**V21 — external audit of 2026-09-19: five claims checked, five upheld, and
the corrections are recorded here rather than quietly applied.**

1. *The time-series user-level comparison was not matched.* `eval_channel_views`
   kept only the malicious users' **attack** days while the language-model
   evaluation scores **all** of their days; with max-aggregation that is a
   different population. Recomputed on the LM's own 142,072 rows, full-score
   user AUC is **0.789 / 0.785 / 0.739**, not the 0.477 / 0.508 / 0.296 first
   reported. The claim that these models "rank users near chance" was an
   artifact and is withdrawn. Default changed to the matched population.
2. *A higher user AUC there is not attack localization.* For **0 of 4**
   malicious users, in every model and both views, is the top-scoring day an
   attack day — the models rank those users highly off a *benign* day. The
   evaluator now reports this count next to user AUC.
3. *Forecasting boundary bug.* `build_windows` pads a short history by
   repeating the earliest row, so after dropping the target column a user's
   first day was its own context. 4,000 rows (one per user, 0 positives). Now
   excluded from training and flagged; the profile share moves −0.14 % →
   +0.24 %, so the reading is unchanged.
4. *AP paired with the wrong prevalence.* Fold-average day AP 0.0164 belongs to
   folds of mean prevalence 0.00080, not the pooled 3.23 %. Pooled AP is
   0.0691 → 0.1452 (×2.1) against the fold-average ×3.1. The r4.2 README said
   "0.0164 at 3.2 % prevalence"; corrected.
5. *Two miscounts.* Held-out rank improves for **51** of 60 folds, not 54 — I
   counted 3 ties as improvements. And cache agreement is not "three decimals
   on every metric": ROC metrics agree to ≤ 0.0008 but pooled user AP differs
   by 0.0071.

Also upheld: r6.2's AP did improve (0.0001 → 0.0010, ×10.9), so "unlike r6.2,
not a ROC-only gain" was the wrong contrast. The real difference is that r6.2's
recall at a 0.1 % false-positive budget stayed **exactly zero** while r4.2's
rose to 0.0172.

**V22 — the mechanism hypothesis in V-prose was wrong in its key clause, and is
corrected.** It said the LM's profile term is a *familiarity* signal, "seen
users' profiles cheap, unseen users' expensive". That cannot be what separates
these groups: **every** user in the factorial audit pool is unseen by the
adapter (V5 — positives are eval-split, the benign comparison validation-split).
The defensible version is **typicality**: because the identity block must be
re-predicted from nothing in every independently scored unit, its cost measures
how typical that profile is under the training distribution, not whether its
owner was seen. The paper's own novelty check is consistent — the malicious user
with the most training-set OCEAN neighbours is the one the 8B adapter ranks
least anomalous. A forecaster escapes the whole mechanism by copying the
profile from the unit's own history.

This remains a hypothesis. The three time-series runs change daily aggregates
vs session text, squared error vs token likelihood, personal history vs
independent scoring, and 1.25 M vs 300 k training rows all at once, so they
cannot isolate the cause. The discriminating test is inside the *same* frozen
LM: score current-day targets with no history, with the user's own previous
profile prepended, and with a matched other-user profile as control.

**V23 — Anvil unreachable on 2026-09-19; recorded as a failed check.**
`ssh ... anvil.rcac.purdue.edu` returns `Operation timed out` (port 22). No
inference is drawn about the cause or about the state of the LANL (20827647)
or Phase C (20816765) jobs; they are neither assumed failed nor assumed
queued. Consequences: r4.2's raw per-example scores could not be re-verified
this session, so its verification rests on the **saved per-fold summaries**
(reproduced to ~3.6e-15) rather than on raw scores; and work packages 3 and 4,
which need the r6.2 8B factorial adapter and the r4.2 headline adapter plus its
SAE, cannot run. Independent work proceeded on Aquaman, which is reachable
(both RTX 3070s idle).

**V24 — a shared metric implementation now backs both evaluators.**
`scripts/eval_metrics_core.py`, with 19 checks in
`scripts/tests/test_eval_metrics_core.py`. It fixes a defect the audit found
and one the tests found:

* *Eligibility now governs the bootstrap.* The previous numerical evaluator
  changed the point-estimate population to matched but left the bootstrap
  resampling malicious **attack days only** against validation negatives, so
  the interval did not describe the statistic printed beside it. The core takes
  one `eligible` index array and uses it for point estimates, user maxima,
  within-user metrics, the mean-gap decomposition and the bootstrap alike. In
  the regression test the two populations give user ROC 1.00 versus 0.00 on the
  same scores, which is the size of error this class of mistake can reach.
* *A degenerate test was caught and replaced.* The first version of that
  regression used random scores, under which the statistic was 0.0 for both
  populations and the check passed vacuously. Rebuilt with a construction that
  mirrors the real finding (each malicious user's top row is benign).

Also in the core: duplicate ids must agree on label and score or `dedup_by_id`
raises; `recall_at_fpr` fixes its threshold with `method="higher"` and counts
strictly-greater scores, so ties cannot inflate it; `user_max` returns the row
that won, which is what makes a benign top row visible.

---

## Unresolved limitations

**L1 — four malicious users.** r6.2 has four positive users and the effects are
dominated by CDE1846 (46 of 70 positive days). Cluster bootstrap intervals over
four clusters are descriptive, conditional on these adapters and this fixed
benign cohort. They are not training-seed replication, and more training seeds
would not increase the number of independent positive users.

**L2 — capped-corpus factorial adapters, not the headline adapters.** These
adapters, their pools, their dictionaries, and their baselines are a separate
line of evidence from the paper's mechanistic sections and must not be mixed
with them.

**L3 — numerical reproduction is approximate.** The cached scores were produced
at batch 1 on H100; the recomputation runs on A100, so exact bitwise agreement
is not expected. The gate requires identical token counts, rank correlation
≥ 0.999 and max |Δ NLL| < 0.05 against the cache. The scientific comparison
(full vs behaviour-only) is *within* one recomputation, so any residual
cross-run drift cancels.

**L4 — DAY is not purely static.** It carries `week=` alongside the
organizational fields, so removing the DAY contribution also removes a temporal
cue. `DAY_WEEK` is accumulated separately so this is measurable.

**L5 — behaviour-only is a scoring intervention, not identity removal.** The
profile stays in the context and conditions every behavioural prediction.

**L6 — exploratory status.** These evaluation users have been inspected
repeatedly in earlier phases. Any analysis on them is exploratory and must not
be relabelled independent confirmation.

**L7 — LANL classes are not CERT classes.** The LANL schema is field-level
(`su`/`du` → ID_USER, `sc`/`dc` → ID_HOST, `at`/`lt`/`or`/`res` → BEHAV, bare
`t<hh>` → HOUR). It is implemented but **not yet validated against LANL data**.

**L8 — no novelty is claimed for the arithmetic.** Token-class score
attribution, probing, concept erasure, and the distinction between encoded
information and behavioural use all have prior literature (Amnesic Probing,
LEACE, likelihood-ratio OOD correction, SAE-based direction removal). Any
contribution would have to be an audit that distinguishes the pathways *and*
prospectively picks a mitigation that helps on held-out users or datasets —
which is not established by anything in this record.

---

## Run log

| date | what | status |
|---|---|---|
| 2026-09-17 | Inventory: code, caches, checkpoints, populations, balances | done (V1–V6) |
| 2026-09-17 | `token_class_decomposition.py` + 47 correctness checks | done (V7) |
| 2026-09-17 | `score_token_class_decomposition.py`, `eval_score_views.py` | done |
| 2026-09-17 | `train_qlora.py` target-mask + explicit denominator (defaults unchanged) | done (V8) |
| 2026-09-17 | Schema validation vs real tokenizer/data on the login node (CPU, no SU) | done (V9, V10) |
| 2026-09-17 | Jobs 20807957/20807958 (both A100): pilots ran, **gate refused** the full run — 8B could not be anchored to a cache made on H100 | done (V11) |
| 2026-09-18 | Rerun 20814766 (8B/H100) and 20814767 (3B/A100), batch 1 on matched hardware | **done** — gates passed with exact reproduction; results in `results/score_decomposition/` (V14, V15) |
| 2026-09-17 | Phase C 3B authorized; job 20810414 ran the loss-path gate and **correctly aborted** (4.0x normalization error), ~0.25 SU | done (V12) |
| 2026-09-18 | Phase C 3B resubmitted as job 20816765 with the corrected loss path: gate, then train-only (16 h cap, ~11 SU); scoring follows as a separate batch-1 job | queued |
| 2026-09-18 | Portability round 1: LANL 20826171, r4.2 20826196 | r4.2 **gate refused** (batch-56 cache, V17), ~0.1 SU; LANL deferred by maintenance (V18) |
| 2026-09-18 | Portability round 2: r4.2 job 20827646 (2 h wall, corrected gate) and LANL job 20827647 (5 h) | r4.2 schedulable before maintenance; LANL waits for the window to close |
| — | Phase C 8B | held: ~20-25 SU, and now lower priority than portability - if score masking transfers, the cheap intervention is the result and the training arm is a secondary control |

---

## 2026-09-19 — four-package review response

**V25 (verified).** All three numerical-detector checkpoints reload and
reproduce their cached per-class error sums with max abs difference
**0.000e+00** over 1,393,297 rows (`scripts/timeseries/check_checkpoint.py`,
records in `results/timeseries_probe/{w1,w7,fc7}/checkpoint_gate.json`).

**V26 (verified).** The forecast boundary fix is exactly scoped: the rows that
would contain themselves in their own context are **exactly** the rows excluded
for lack of history (4,000 self-containing, 4,000 excluded; 0 eligible rows
leak). The gate asserts set equality, not just "eligible rows are clean".

**V27 (new limitation, not previously recorded).** 4,000 eligible rows — each
user's *second* day — have a padding-collapsed context: one real prior day
repeated across all seven columns. They leak nothing, but their effective
history is 1 day, not 7, so "7-day forecaster" overstates the context for 2.8 %
of eligible rows.

**V28 (verified).** All models now share ONE eligible set of 141,662 rows
(`shared_matched/common_eligible_ids.csv`, md5 07b7024ddff248e6e03c00980ec77523).
Profile share of the malicious-benign mean score gap: **−3.7 %** (recon w=1),
**+10.2 %** (recon w=7), **+0.24 %** (forecast). Given the same fields the
language model gets, a non-language detector puts essentially none of its score
gap into the profile channels.

**V29 (verified, descriptive).** Replacing each target day's context with a
different user's history drops the forecaster's day ROC from 0.7061 to 0.5725
(Δ −0.1335) but its user ROC only from 0.7389 to 0.7137 (Δ −0.0252). Day-level
detection uses history; user-level ranking largely does not.

**V30 (verified, negative).** For **0 of 4** malicious users, in every model,
for both the full and behaviour-only scores, is the highest-scoring day an
attack day. High user AUC here is not attack localization. Unchanged by the
shared evaluator.

**V31 (verified).** `token_class_nll.py` agrees with the reference Python
implementation in `token_class_decomposition.accumulate_class_losses` (max
|Δ| 1.3e-05 on sums, exact on counts), reconstructs the scalar per-example NLL
(9.5e-07), is independent of batch partners (0.0), and its chunked LM-head path
matches the direct path (1.5e-05). 17 checks in
`scripts/tests/test_token_class_nll.py`.

**V32 (verified).** The history-prefix probe's three conditions score identical
targets with identical per-class counts, no prefix token contributes loss, and
identical current-day predictions give identical per-class sums (max |Δ| 0.0).
10 checks in `scripts/tests/test_history_prefix_scoring.py`, including a
sensitivity check confirming a wrong prefix mask would fail.

**V33 (diagnosed).** Anvil is down. The failure is below SSH and identical on
both accounts, so it is not credential- or account-related:

| check | result |
|---|---|
| DNS for anvil.rcac.purdue.edu | resolves, 8 addresses |
| ICMP ping, 5 packets | 100 % loss |
| TCP port 22 | not reachable |
| SSH as x-sangdembay | connect timed out |

Reported by the user as a **full two-day outage from 2026-09-19**, so expect
service back around **2026-09-21**. The state of jobs 20816765 (Phase C 3B),
20827646 (r4.2 portability) and 20827647 (LANL) remains **unknown**; nothing
about their progress or completion is assumed.

**V34 (capability, verified).** The outage does not block Package 3. Aquaman
holds everything it needs: the r4.2 and r6.2 Qwen3-8B adapters
(`cert-data/{r42,r62}_adapter`, md5 bb68488338a5e283e53bb5df58536fe3 and
1711135fce2932a3b58c0286f9589ac8), the repaired session JSONL for both
releases, and Qwen3-8B in the local HF cache. A virtualenv at
`/data/suramya/insider_mi/lmenv` (transformers 4.53.2, peft 0.14.0,
bitsandbytes 0.45.3, accelerate 1.6.0, inheriting torch 2.7.1+cu118) loads the
8B in 4-bit NF4 across the two RTX 3070s. The probe's loader now mirrors
`score_adapter_examples.py` exactly, so it measures the same model the
published scorer measured.

**V35 (capability, verified).** Package 4 cannot run on Aquaman for **CERT** --
the r4.2 token-delta cache and SAE frontier live only on Anvil. It can be
piloted on **TWOS**, which has a complete pipeline here (adapters, 2.7 GB token
deltas, SAE frontier at layers 12/18/24, prior causal outputs). TWOS uses the
same DAY/PSY/SESSIONS/SES line-prefix serialization as CERT, so the existing
`cert` token-class schema applies unchanged.

| date | what | status |
|---|---|---|
| 2026-09-19 | Package 1: r4.2 headline corrections (empirical ROC operating points; batching-dependent not padding-specific; verification status separating saved-summary checks from raw-score re-verification) | done |
| 2026-09-19 | Package 2: shared metric core, checkpoint gate, multi-model evaluator on one eligible set, corrected README | **done** (V25–V30) |
| 2026-09-19 | Package 3: `run_history_prefix_probe.py` + construction and scoring tests | code done and tested offline (V32); **run blocked on Anvil** (V33) |
| 2026-09-19 | Package 4: per-class accumulation in `eval_token_delta_sae_causal.py`, submitter with smoke mode | code done and tested offline (V31); **run blocked on Anvil** (V33) |


## 2026-09-22 — Phase C 3B trained; scoring and the CERT package-4 run submitted

**V36 (verified).** Phase C 3B training (job 20816765, collaborator allocation)
**completed**, exit 0, 13 h 48 m. The adapter trained with the profile-masked
loss is at `p1/adapter_targetmask/adapter`; `masked_target_frac = 0.2299`, pool
159,064, "identical recipe to the factorial arms". Scoring did not run as part
of the job by design.

**V37 (submitted).** Phase C scoring submitted as job **20861905** on
cis260991-gpu: the exact batch-1 recipe of `decomp_run2.sh` on the identical
pool, structural gate first, then the full pool, then `eval_score_views.py`.
No numerical reproduction gate because no cached score exists for this
adapter. Expected ~5 h from the 3B full-arm throughput of 9.06 ex/s.

**V38 (submitted).** CERT package-4 confirmation run submitted as job
**20861865** (gpu, 24 h) with smoke **20861864** (gpu-debug), both on
cis260991-gpu, after cis230270-gpu was found exhausted (2.2 SU). Inputs are
staged in the owner's world-readable scratch; the staged adapter is
fingerprint-identical to the published one (compared against
`results/provenance/adapter_fingerprints/anvil_r42.json`). A CPU-only validation on the
collaborator environment caught and fixed a cache-path defect (`HF_HUB_CACHE`)
that would have failed the run at model load, and confirmed tokenization is
identical across the two environments.

| date | what | status |
|---|---|---|
| 2026-09-22 | Phase C 3B scoring (20861905) | queued, ~5 SU |
| 2026-09-22 | CERT package-4 smoke (20861864) + full (20861865) | queued, ~24 SU |
| 2026-09-22 | Outputs of 20827646 (r4.2 portability) and 20827647 (LANL) | completed on the collaborator account; **not yet collected** |


## 2026-09-23 — Phase C 3B result: the training mask adds nothing beyond score masking

**V39 (verified).** Phase C 3B scoring (job 20861905, collaborator allocation)
completed, exit 0, 4 h 49 m. Structural gate passed on the full pool: partition
sum error 1.3e-12, zero count mismatches, `n_targets = n_tokens − 1` for every
example, mean reconstruction error 0.0. No numerical reproduction gate exists for
a new adapter. The masked run's class shares equal the full-adapter run's on the
same pool to every printed digit (DAY 0.11416, PSY 0.07871, SESCOUNT 0.03080,
SES 0.77633), so both adapters were scored on identical token classes. The
tokenizer digests differ (`cd33debe…` vs `22576a18…`); the digest hashes file
names and sizes of the adapter's tokenizer files, and identical class counts
over 159,064 examples show the token sequences match. *(Correction
2026-09-24: withdrawn. Identical class counts are consistent with identical
token sequences but do not show them. The external review compared both saved
tokenizers on 256 real records: 0 token-ID mismatches, with differing
serialized backend digests. That supports agreement on those records only.)* Both evaluations drop the
same 16,992 duplicate rows and have identical folds (rows, positives, users and
fold ids checked per fold). Outputs copied to
`results/score_decomposition/3b_targetmask/`.

User ROC, mean of four folds (CERT r6.2, Qwen2.5-3B, four malicious users):

| | score all targets | score behaviour targets only |
|---|---|---|
| train on all target losses | **A** 0.9261 | **B** 0.9440 |
| train on behaviour losses only | **C** 0.8011 | **D** 0.9415 |

Per user, A / B / C / D: ACM2278 0.9951 / 0.9975 / 0.9951 / 0.9975;
CDE1846 0.8030 / 0.9655 / 0.6232 / 0.9754; CMP2946 0.9655 / 0.9975 / 0.9877 /
0.9975; MBG3183 0.9409 / 0.8153 / 0.5985 / 0.7956.

- **B → D, the clean training-objective contrast (H2): −0.0025**, exact
  cluster bootstrap over all 256 resamples of the four users [−0.0148, +0.0074];
  one user up (CDE1846 +0.0099), one down (MBG3183 −0.0197), two tied at the
  ceiling. No detectable effect.
- C → D (score inclusion, masked adapter): +0.140 [+0.006, +0.275], from the
  run's own contrast file; same direction as A → B.
- A → C: −0.125. Diagnostic only, as the spec says: C scores profile targets the
  masked adapter was never trained to predict.

Reading, within the spec's declared limits: at 3B, once profile tokens are left
out of the score, also leaving them out of the training loss changes nothing
measurable. H2 is not supported at this scale, but B was already 0.944, so there
was little headroom, and the spec states a 3B null does not speak to 8B. One
training seed and four positive users.

**V40 (verified) — the masked fraction is reconciled.** V36 recorded
`masked_target_frac = 0.2299` against V9's p = 0.1485. A CPU recount on the
collaborator login node with the same tokenizer, `max_seq_len` 2048 and class
library gives:

| sample | profile / all tokens | mean tokens per example |
|---|---|---|
| train, first 5,000 in file order (what `train_qlora.py` probes) | **0.2299** | 249.5 |
| train, random 5,000 of the first 300k | 0.1936 | 296.4 |
| scoring pool, random 5,000 | 0.1951 | 294.0 |
| scoring pool, all 159,064 (targets, from the manifest) | 0.1929 | — |

The exact match on the first 5,000 confirms that the mask covers DAY and PSY
tokens only. Those examples are shorter days, which raises their profile share.
The training corpus and the scoring pool agree at about 0.19. V9's 0.1485 came
from 400 examples and understates p. With p ≈ 0.19, the HF-default denominator
would up-weight behavioural tokens by about 1.24, not 1.174. Cells C and D used
`--loss-denominator all_targets`, which applies no reweighting, so neither
figure enters this result.

| date | what | status |
|---|---|---|
| 2026-09-23 | Phase C 3B scoring (20861905) | **done** (V39, V40) |
| 2026-09-23 | CERT package-4 smoke (20879548, gpu-debug) + full (20879549, gpu), cis260991-gpu; resubmissions of 20861864/5 after the chunk-path fix | queued (Priority); scheduler estimate 2026-09-27; at 21:21 EDT the estimate was 2026-09-26 ~19:35 (smoke) and ~20:05 (full). Both GPU partitions are saturated (gpu 380 pending / 43 running, ai 393 / 36), so moving to H100 would not help |
| 2026-09-23 | 20840922, same causal script on cis230270-gpu (submitted 2026-09-21) | **cancelled** 22:00 EDT. The owner's GPU allocation will not be refilled; all GPU work now runs on the collaborator account (A100 `gpu`/`gpu-debug` or H100 `ai`) |
| 2026-09-23 | Outputs of 20827646 (r4.2 portability) and 20827647 (LANL) | **collected 2026-09-22**, per `results/score_decomposition/r42_headline/README.md` and `lanl_full/README.md`; the 2026-09-22 row saying "not yet collected" is stale |
| — | Phase C 8B | held, as before |

**Compute policy, 2026-09-23.** The owner's GPU allocations (cis230270-gpu,
cis230270-ai) are exhausted permanently. GPU jobs run on the collaborator account
only: A100 via `cis260991-gpu` or `tra250034-gpu`, H100 via `tra250034-ai`. At
22:05 EDT `sbatch --test-only` put a *new* full-length job at 2026-10-10 on H100
and 2026-10-20 on A100, against 2026-09-26 23:05 for the already-queued 20879549.
Priority here is almost entirely age (PriorityWeightAge 20000, fair-share weight
0, no ACCRUE_ALWAYS), so the queued jobs were kept rather than moved, and no
smoke→full dependency was added, since an unsatisfied dependency stops age from
accruing. A new 30-minute `gpu-debug` job tested at 2026-09-24 02:44. **Correction
(2026-09-24 01:31):** that estimate does not account for the jobs already
waiting. The smoke is 9th by priority in `gpu-debug`, where nothing was running
at either check, so no early start should be expected.

**Queue state 2026-09-24 01:31 EDT.** Smoke 20879548 estimated 2026-09-26 17:20,
full 20879549 17:25; the full run is 41st of 376 pending in `gpu`. The two may
therefore overlap rather than run smoke-first (pre-registration rule 6). Accepted,
because both pre-registered checks are unconditional in the full run too: the
per-class reconstruction check (`eval_token_delta_sae_causal.py` line 833, raises
above 1e-4) and the base/patched count equality (line 1250) run on every batch
whenever `TOKEN_CLASS_SCHEMA` is set, which it is in both modes. The copy on the
collaborator scratch is byte-identical to the tested file (md5 6fe1d865…). A
failure would stop the full run at its first batch, so the overlap risks minutes
of GPU time, not a void result entering the record. A dependency was not added
because it would stop the full run's age accrual.


## 2026-09-24 — review findings verified; repairs; corrected runs queued

Full record: `docs/REVIEW_2026-09-24_ISSUE_LEDGER.md`; plan:
`docs/EXPERIMENT_PLAN_2026-09-24.md`.

**V41 (verified) — the TWOS token-class runs used an undefined ranking.** Both
runs read the benign-only frontiers `twos_work/v3_sae_s42|s43`, whose rankings
were computed on benign rows only (0 finite gaps of 8,192), and patched
`[0, 1, 2, 3, 4]` against the most active features. The original
`v3_causal_s42` read the valid re-ranked frontier. The per-class results and
the seed-reversal reading are withdrawn (READMEs and audits annotated).

**V42 (verified) — LANL sampling and folds shared a hash.** The seen pool has
988 negative-only users, all in fold 0; the unseen pool has 23 users, all
attack users. The seen/unseen interpretation is withdrawn, the AUCs are kept as
descriptions of those pools, and the paper's LANL claims are corrected.
Descriptive: seen AUC with unseen-like composition 0.916; mean within-user AUC
0.883 seen, 0.738 unseen.

**V43 — repairs.** Ranking gates, benign-only frontiers without a ranking
file, re-selection manifests, selection manifests and disjointness checks,
independent salted LANL hashes with population checks. Tests: 39 ranking checks
and 12 LANL checks pass; all 8 test files pass on CPU.

**V44 — corrected TWOS run submitted.** The ranking was redone on 8 discovery
users only, with confirmation users' rows excluded: top5
`[6036, 7375, 1197, 7420, 3218]`, every gate passed. Tokenization matched on
all 2,275 records across environments. Bounded held-out run with an α = 0
reconstruction arm: job **20890658** (gpu-debug, cis260991-gpu). Plan committed
before submission (`d6516b3`).

**V45 — CERT package-4 analysis fixed before results.** Dated addendum to the
pre-registration; `analyze_intervention_endpoints.py` and
`compare_intervention_runs.py` (both self-tested). The environment gate
compares collaborator smoke 20879548 with owner smoke 20840686.

| date | what | status |
|---|---|---|
| 2026-09-24 | TWOS held-out bounded run (20890658) | queued, ≤ 0.5 SU |
| 2026-09-24 | CERT package-4 smoke 20879548 / full 20879549 | queued; smoke must pass the environment gate before the full run is read |
| 2026-09-24 | CERT α = 0 reconstruction run; LANL repair option A | planned (tier 2), not submitted |

**V46 — CERT package-4 environment gate PASSED on 2026-09-24. The collaborator smoke
(job 20879548; transformers 5.16.1, PEFT 0.20.0) and the owner smoke (job
20840686; transformers 4.51.3, PEFT 0.13.2) match on all 1,186 rows (same
receivers, donors, arms and alpha; both torch 2.5.1 on A100-SXM4-40GB). The
largest per-row difference in any view's delta is 9.5e-07 nats (tolerance
1e-3), and every selected-minus-control contrast agrees within 6e-09
(tolerance 1e-4). Report: `results/cert_package4_env_gate/GATE_REPORT.txt`.
This verifies that the environment change does not move these smoke rows
(team, alpha 1, 24 receivers). It is not a check of the other context modes
or alphas, which only the full run contains. The smoke's own contrasts are a
validation subset and are not read as results.**

**V47 — corrected TWOS held-out run (job 20890658): inconclusive; reconstruction
confounds the raw contrast.** Every pre-declared gate passed: selection held out,
68 receivers from the 8 confirmation users, planned feature ids. Benign donors at
alpha 1, selected minus control: behaviour −0.0001 [−0.0050, +0.0052], profile
−0.0129 [−0.0333, +0.0061] (8 users; every interval includes zero). Alpha 0 shows
the controls, active on 59 of 68 receiver days against 43 for the selected set,
gain +0.0121 of profile loss from reconstruction alone. Net of reconstruction the
profile difference is −0.0021 [−0.0194, +0.0111]. Seed 43 not triggered. Details:
`results/twos_corrected_bounded/README.md`. Consequence for CERT: the
reconstruction-only control is run before the per-class differences are read
(addendum, item 8 implementation).

**V48 — CERT package-4 result.** After jobs 20879549 and 20898582 completed and
passed every gate. Full record: `results/cert_package4_token_class/README.md`.
- **Primary comparison** (benign donors, alpha 1, per receiver user,
  net of reconstruction): behaviour selected − control is +0.0156 to +0.0179
  across the four context modes, every interval excluding zero. Profile is
  +0.0065 to +0.0086, every interval including zero. The direct
  profile-minus-behaviour contrast is −0.0073 to −0.0114, every interval
  including zero. No view falls within the ±0.0002 margin.
- **Reconstruction** is identical in both arms here, because every receiver
  day is patched by both arms, so it cancels in every difference.
- **All alphas.** The behaviour difference is slightly negative at 0.25 and
  0.5 (mostly within noise) and positive at 0.75 and 1.
- **The prediction is contradicted** on every count and is reported as-is.
- **The historical best-candidate endpoint** is reproduced exactly (dept
  0.000758) when chosen with a constant base. Averaged over candidate donors
  at alpha 1 it becomes negative in all four modes (secondary analysis,
  motivated by ledger C6 before results).

**What this means.** The selected features' edits act on behaviour-token
predictions, where the features fire. Whether that reflects anomaly-specific
content or just larger edits needs a dose-matched control. The published
"benign donor lowers the score more" reading holds only for the best of up to
64 candidate edits per day, not for the average edit.
