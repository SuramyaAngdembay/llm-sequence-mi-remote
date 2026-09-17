# Experiment manifest — token-class score decomposition and behaviour-only scoring

Opened 2026-09-17. Repository state at start: `110fd1e`.

## Question

Where does the profile-related detection failure enter the pipeline? Three
distinct pathways, not assumed to be the same thing:

1. **Score inclusion** — profile-token prediction losses are part of the
   anomaly score.
2. **Training objective** — the adapter is trained to predict profile tokens.
3. **Context conditioning** — profile information changes the predictions of
   *behavioural* tokens.

Phase A/B (this manifest) tests pathway 1 on frozen adapters and measures the
size of the direct profile contribution. Phase C prepares the training
experiment that separates pathway 2. Pathway 3 is what remains after both, and
is deliberately *not* claimed to be removed by either intervention: the profile
text stays in the context for every view scored here.

Representing identity is not assumed to mean using it harmfully, and profile
memorization is not treated as the established explanation.

## Checkpoints, tokenizers, inputs

Two *capped-corpus train-matched factorial* adapters. These are **not** the
original headline adapters of the paper's mechanistic sections; their scores,
populations, dictionaries and baselines must not be mixed with those.

| | 8B (primary: the observed failure) | 3B (comparison) |
|---|---|---|
| base model | `Qwen/Qwen3-8B` | `Qwen/Qwen2.5-3B` |
| config | `configs/qwen3_8b_qlora_session_targeted.yaml` | `configs/qwen3b_qlora_session.yaml` |
| adapter | `$SCRATCH/p1_8b/adapter_full/adapter` | `$SCRATCH/p1/adapter_full/adapter` |
| tokenizer | from the adapter dir (fast); Qwen2.5/Qwen3 share the vocabulary — token counts are identical across the two pools (mean 293.2, max 1187) | same |
| training | 300k benign examples, 2k val, seed 42, eff. batch 16 (4×2×2 DDP) | 300k/2k, seed 42, eff. batch 16 (4×4, 1 GPU) |
| serialized input | `$SCRATCH/p1/jsonl_r62_full` (`build_session_jsonl_fast.py --profile-mode full`) | same |
| scored pool | `$SCRATCH/p1_8b/pool_full/eval.jsonl` | `$SCRATCH/p1/pool_full/eval.jsonl` |
| cached baseline | `$SCRATCH/p1_8b/deltas_full/example_scores.parquet` | `$SCRATCH/p1/deltas_full/example_scores.parquet` |
| published fold rows | `results/train_matched_factorial/8b/fold_full/` | `.../3b/fold_full/` |

`$SCRATCH` is the collaborator account `x-bbhusal1` on Purdue Anvil.
Later checks (after the implementation is validated): r4.2 headline adapters
and the LANL conditions, whose identity/behaviour target groups must be defined
from LANL's own field-level serialization — CERT line classes do not transfer.

## Population

The scored pool is built by the factorial runners as: **all** days of the
answer-key (`eval`-split) users, plus a seeded 12 % sample of `val`-split
benign days (`random.Random(42)`, `r < 0.12`). Verified composition, identical
for 3B and 8B:

- 159,064 day-rows; 410 users; 70 positive rows.
- 4 positive users: ACM2278 (5 positive of 176 days), CDE1846 (46/356),
  CMP2946 (18/316), MBG3183 (1/356).
- 406 benign users: 405 `val`-split, plus `PLJ1771` — an answer-key user with
  no positive day inside the matched session domain.

**Every user in this pool was excluded from adapter training** (positives by
construction; the benign comparison users are the validation split). This is a
user-disjoint comparison. It is *not* a seen-user attack-detection test, and
must not be described as one.

Fold construction reuses `eval_fold_aligned_detector_metrics.make_folds`
(imported, not reimplemented): one held-out malicious user per fold, benign
cohort `min(800, 406) = 406`, so 407 test users per fold and the same benign
cohort in all four folds. Seed 42.

## Serialization and truncation

`serialize_text` emits, in order: `DAY <org fields>`, `PSY <O C E A N>`,
`SESSIONS total= kept=`, then one `SES ` line per session. Tokenization is the
adapter's fast tokenizer with `truncation=True, max_length=2048`; the observed
maximum is 1187 tokens, so no example in this pool is truncated.

Token classes are assigned by **line prefix**, never by line index:
`build_session_jsonl_fast.py` deletes the `PSY ` line under `no_psy` and both
`PSY `/`DAY ` lines under `no_profile`, so line 1 is PSY in one condition and
SESCOUNT in another. A token takes the class of the span containing its first
character (the convention already used by `feature_token_attribution.py`);
tokens whose span crosses a class boundary are counted separately so the
ambiguous mass is measured rather than assumed negligible.

`DAY` contains `week=` as well as static organizational attributes, so removing
the whole `DAY` contribution is **not** purely removing static attributes. The
`week=` span is additionally accumulated as the sub-class `DAY_WEEK` (a subset
of DAY, reported alongside it, not a partition class).

## Score definitions (frozen before any comparison)

The detector score is the mean next-token NLL over attention-valid targets
(`extract_adapter_deltas.per_example_nll`). With classes `c` and counts `N_c`:

    s_full = sum_c (N_c / N) * s_c ,   s_c = loss_sum_c / N_c

Views (defined once in `token_class_decomposition.CERT_VIEWS`):

| view | target classes | note |
|---|---|---|
| `full` | all | reproduces the published `adapted_nll` |
| `profile_only` | DAY, PSY | direct profile contribution |
| `behavior_only` (**primary**) | SESCOUNT, SES, OTHER, SPECIAL | every non-profile target; **includes SESCOUNT** |
| `behavior_ses_only` (secondary) | SES | **excludes SESCOUNT** |
| `day_only`, `psy_only` | DAY / PSY | diagnostics |

`s_full - s_profile` is **not** implemented as a behaviour score: it equals
`(N_B/N)(s_B - s_P)` and can reverse a behaviour-based ranking (demonstrated in
the unit tests). Behaviour targets are accumulated directly in the full forward
pass; the algebraic identity
`s_B = ((N_P+N_B) s_full - N_P s_P) / N_B` is asserted as a cross-check rather
than used as the implementation. No isolated-prefix rescoring is used.

All views are scored on identical examples with the same frozen adapter and the
**full input context**: profiles are always present in the context. A
behaviour-only view removes the profile's direct score contribution only.

## Metrics

Day-level ROC-AUC and average precision; user-level ROC/AP after
max-aggregation over days; held-out user rank out of 407; within-user ROC for
the held-out malicious user (identity held fixed); day-level recall at
prespecified false-positive budgets 0.1 % and 1 %; user top-1 % recall.
Population sizes and prevalence reported with every table.

Uncertainty: paired cluster bootstrap (10,000 draws) resampling the four
malicious users, with identical draws across views. With four clusters these
intervals are **descriptive**, conditional on these adapters and this fixed
benign cohort. They are not training-seed replication.

Mean-score-gap decomposition uses the exact identity
`ΔE[s] = Σ_c ( E_pos[w_c s_c] − E_neg[w_c s_c] )` with `w_c s_c = loss_sum_c /
n_targets` per example — never `E[w]·E[s]`. Contributions are signed and can
cancel, so a class share may exceed the net gap or be negative. A mean-gap
decomposition is not an AUC decomposition and not a causal mechanism.

## Cache status (checked, not assumed)

`extract_adapter_deltas.py` saves **sequence-average** `base_nll`,
`adapted_nll`, `delta_nll` and `n_tokens` — there are no per-token or per-class
losses anywhere in the tracked artifacts or on the cluster. The decomposition
therefore **requires new inference**. It is not CPU-only and not free; the
earlier "zero SU" claim was unsupported.

Only the adapted model is needed (the detector score is `adapted_nll`), so this
is one forward pass per example, with no hidden-state dumps — cheaper than the
original extraction, which ran base + adapted with token-level deltas at three
layers.

## Compute authorization (verified 2026-09-17, not historical)

| allocation | type | balance |
|---|---|---|
| `cis260991-gpu` | A100 (`gpu`) | 400.0 SU |
| `tra250034-gpu` | A100 (`gpu`) | 95.2 SU |
| `tra250034-ai` | H100 (`ai`) | 732.5 SU |
| `cis260991` / `tra250034` | CPU | 1000.1 / 17,173.3 SU |

Phase A/B scoring runs on `cis260991-gpu` (A100), leaving the H100 balance for
the Phase C training experiment. Debug-queue smoke test first
(`gpu-debug`, job 20807873): unit tests under the cluster interpreter, 256
examples at batch 8 and at batch 1, checked against the cached `adapted_nll`.
Bounded pilot estimate and the full-pool budget are recorded in
`docs/SCORE_DECOMPOSITION_PROGRESS.md` once the smoke test reports throughput;
no full-pool job is launched before that number exists.

## Deliverables

- `scripts/token_class_decomposition.py` — schema + accumulation library.
- `scripts/tests/test_token_class_decomposition.py` — correctness checks.
- `scripts/score_token_class_decomposition.py` — GPU scorer, writes the cache.
- `scripts/eval_score_views.py` — matched-population evaluation of the views.
- `docs/SCORE_DECOMPOSITION_PROGRESS.md` — verified observations, hypotheses,
  unresolved limitations.
