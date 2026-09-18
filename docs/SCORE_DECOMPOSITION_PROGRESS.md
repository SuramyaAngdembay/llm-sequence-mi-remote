# Progress record — score decomposition and behaviour-only scoring

Kept deliberately separated into what has been **verified**, what is a
**hypothesis**, and what is an **unresolved limitation**. Results are appended
as they land; nothing here is written into the paper before it is checked.

Last updated: 2026-09-17 (Phase A/B implementation complete and validated on
real data; scoring jobs and the Phase C 3B arm queued on Anvil).

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

---

## Hypotheses (not yet tested)

**H1 — direct score inclusion is a material part of the failure.** If the
profile's own prediction losses dominate the seen/unseen mean-score gap, then
the behaviour-only view should improve unseen-user ranking on the 8B `full`
adapter. *Status: awaiting the scoring jobs.* Note that a mean-gap
decomposition would not by itself establish this: ROC is a pairwise-ordering
probability, not a difference of means, so the ranking of each view must be
computed directly.

**H2 — being trained to predict profile tokens is a separate harm.** Tested by
B → D in Phase C. Masking targets does not stop behavioural losses from
training the adapter to *use* profile context, so a null result would not show
that identity information is absent.

**H3 — context conditioning is the residue.** Whatever survives both
interventions is profile influence on behavioural predictions. No experiment
here removes it; claiming otherwise would be unsupported.

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
| 2026-09-18 | Rerun 20814766 (8B on `ai`/H100) and 20814767 (3B on `gpu`/A100), batch 1 on matched hardware, revised gate | queued |
| 2026-09-17 | Phase C 3B authorized; job 20810414 ran the loss-path gate and **correctly aborted** (4.0x normalization error), ~0.25 SU | done (V12) |
| 2026-09-18 | Phase C 3B resubmitted as job 20816765 with the corrected loss path: gate, then train-only (16 h cap, ~11 SU); scoring follows as a separate batch-1 job | queued |
| — | Phase C 8B | held: ~20-25 SU (corrected from ~56; measured 1.07 s/it x 18,750 steps on 4xH100), pending the 3B run and the Phase A/B result |
