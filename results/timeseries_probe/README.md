# Time-series reconstruction probe on CERT r6.2

Run 2026-09-18 on Aquaman (2× RTX 3070), **zero cluster SU**.

## The question

The paper's Discussion leaves a prediction untested: *"sequence autoencoders,
which share the sequence-prediction objective but not the pretrained
representation, should fall between the tabular detectors and the LM in profile
reliance; auditing their latent spaces with this framework would locate the
shortcut's origin more finely."*

This fills that cell. It is a **non-language** detector given **the same fields
as the language model**, trained benign-only, with a score whose profile and
behaviour parts separate exactly — the channel-level analogue of the token-class
decomposition.

## Matching, verified not assumed

Channels are pinned to the LM's own serialization: PROFILE = the DAY/PSY block
minus `week` (12 channels, identical to `cross_arch_probe.PROFILE_COLS`), BEHAV
= `build_session_jsonl.SESSION_COLS` plus `n_sessions` (32). Splits use the LM
branch's `stable_hash_frac` with the answer-key positive users reserved for
eval, and reproduce it exactly:

| | rows | users |
|---|---|---|
| train | 1,251,225 | 3,590 |
| val | 140,711 | 405 |
| eval | 1,361 | 5 |

70 positive rows from 4 users in the matched domain. Every profile channel is
**perfectly constant within each user** (measured fraction 1.000) — the premise
of the shortcut argument.

Two traps were caught while building this and are worth recording, because
either would have decided the experiment by construction:

* the raw integer `user` code survives `sanitize_frame`'s rename to `user_id`
  and is numeric, so a naive "all numeric columns" selection hands the model an
  explicit identity channel the LM never sees. Excluded.
* split assignment must key on the **answer-key** positive users (5 on r6.2),
  as `build_session_jsonl` does, not on users still holding a positive day after
  the merge to the matched session domain (4). Keying post-merge moved
  `PLJ1771` into train/val and desynchronised the populations by one user and
  157 rows.

## Detector

Small MLP autoencoder over (window × channels), reconstructing the current day;
benign-only on the train split; standardisation from training statistics only.
`--window 1` is the per-day autoencoder (a learned nonlinear analogue of the
published PCA-reconstruction baseline); `--window 7` gives the user's previous
seven days as context, never crossing a user boundary.

Training is ~7 s for 5 epochs over 1.25 M rows on one RTX 3070.

## Result — one shared metric implementation, one common eligible set

Every number below comes from `scripts/eval_metrics_core.py`, the same module
the language-model evaluator uses, and every model is scored on the **identical**
141,662 rows. The forecaster cannot score a user's first day, so those 410
population rows are dropped from **all three** models rather than leaving each
model on its own row set. 70 positives, 4 malicious users, 410 users.

| variant | view | day ROC | day AP | user ROC | within-user ROC |
|---|---|---|---|---|---|
| recon, w=1 | full | 0.6739 | 0.0010 | 0.7894 | 0.5405 |
| recon, w=1 | profile_only | 0.5927 | 0.0007 | 0.7328 | 0.4767 |
| recon, w=1 | behavior_only | 0.6856 | 0.0013 | 0.7734 | 0.5457 |
| recon, w=7 | full | 0.6829 | 0.0015 | 0.7851 | 0.5878 |
| recon, w=7 | profile_only | 0.5968 | 0.0008 | 0.8023 | 0.5788 |
| recon, w=7 | behavior_only | 0.6979 | 0.0019 | 0.7777 | 0.5808 |
| forecast | full | 0.7061 | 0.0010 | 0.7389 | 0.6295 |
| forecast | profile_only | 0.5900 | 0.0007 | 0.6773 | 0.5822 |
| forecast | behavior_only | 0.7068 | 0.0010 | 0.7389 | 0.6282 |

A view's score is the exact conditional mean over its channel classes (class
error sum / class count), never `s_full - s_profile`, which reorders examples
relative to the conditional behaviour score.

Mean-score-gap decomposition (exact; reconstructs the total to < 1e-9):

| variant | PROFILE contribution | BEHAV contribution |
|---|---|---|
| recon, w=1 | −0.00076 (**−3.7 %** of the gap) | +0.02146 (+103.7 %) |
| recon, w=7 | +0.00418 (**+10.2 %**) | +0.03700 (+89.8 %) |
| forecast | +0.00038 (**+0.24 %**) | +0.16139 (+99.8 %) |

**This is the finding.** Given the same fields the language model gets, a
non-language detector puts essentially none of its malicious-versus-benign score
gap into the profile channels. The language model puts a large share into the
profile tokens. The shortcut is therefore not forced by the field set.

### Primary comparison, declared

The pre-declared test is the **profile share of the malicious-benign mean score
gap**, per model. View-level ROC contrasts are secondary and, with 4 malicious
users, descriptive.

### Checkpoint gates — every model reloads and reproduces its cache

Verdicts are recorded per model in `{w1,w7,fc7}/checkpoint_gate.json`
(`rescore_max_abs_diff`), not asserted here.

`scripts/timeseries/check_checkpoint.py` reloads each saved `model.pt` with its
stored standardisation and rescores all 1,393,297 rows.

| model | rescore max abs diff vs cache | verdict |
|---|---|---|
| recon, w=1 | 0.000e+00 | pass |
| recon, w=7 | 0.000e+00 | pass |
| forecast | 0.000e+00 | pass |

The gate also checks that no eligible row appears in its own context, that the
rows excluded for lack of history are **exactly** the rows that would contain
themselves (4,000 self-containing, 4,000 excluded), and that no context window
crosses a user boundary.

Two things the gate caught, recorded because both were real:

* It refused a run whose `--window` did not match the checkpoint, which is how
  the forecaster's window was found to be 8 (7 context days after dropping the
  target), not 7. The script now reads the window from the training manifest.
* **A limitation the earlier write-up missed.** 4,000 eligible rows — each
  user's *second* day — have a padding-collapsed context: one real prior day
  repeated across all seven columns. They leak nothing, but their effective
  history is one day, not seven, so "7-day forecaster" overstates the context
  for 2.8 % of eligible rows.

### Does the forecaster use history at all?

Attaching each target day to a **different user's** context (no same-user donors
remain) and rescoring:

| metric | true context | shuffled context | Δ |
|---|---|---|---|
| day ROC | 0.7061 | 0.5725 | −0.1335 |
| user ROC | 0.7389 | 0.7137 | −0.0252 |

Day-level detection genuinely depends on the user's own history. User-level
ranking barely does: it survives having the history replaced by a stranger's.
So the user-level number is close to a statement about how atypical a user's
days look on their own, not about history-conditioned surprise. This is
descriptive and was not thresholded.

### Paired cluster bootstrap, 4 malicious users, 2,000 draws

**12 intervals, uncorrected** (3 models × 2 views × 2 metrics), on 4 clusters.

With 4 clusters these intervals are **descriptive**, not inferential.

| model | metric | view vs full | Δ | 95 % interval |
|---|---|---|---|---|
| recon, w=1 | day ROC | profile_only | −0.081 | [−0.200, −0.049] |
| recon, w=1 | day ROC | behavior_only | +0.012 | [+0.010, +0.030] |
| recon, w=7 | day ROC | profile_only | −0.086 | [−0.115, −0.009] |
| recon, w=7 | day ROC | behavior_only | +0.015 | [−0.006, +0.022] |
| forecast | day ROC | profile_only | −0.116 | [−0.482, +0.123] |
| forecast | day ROC | behavior_only | +0.001 | [−0.002, +0.002] |

Every **user**-ROC contrast spans zero. For the forecaster, behaviour-only and
full are identical to four decimals because BEHAV carries 99.8 % of the gap.

### Corrections to earlier versions of this file

**The user-level comparison was not matched.** The first evaluation kept only
the malicious users' *attack* days, while the language-model evaluation scores
**all** of their days; with max-aggregation that is a materially different
population. Corrected, user AUC for the full score is 0.789 / 0.785 / 0.739,
not the 0.477 / 0.508 / 0.296 first reported. The claim that these models "rank
users near chance" was an artifact of the mismatch and is **withdrawn**.

**A higher user AUC is not attack localization.** For **0 of the 4** malicious
users, in every model, for both the full and behaviour-only scores, is the
highest-scoring day an attack day. These models rank malicious users above
benign users on the strength of one of those users' *benign* days. Only
`profile_only` at w=7 and forecast ever lands on an attack day, and then for 1
user of 4. The evaluator reports this count beside user AUC because
max-aggregated ROC does not otherwise reveal it.

**Forecasting boundary.** `build_windows` pads a short history by repeating the
earliest available row, so after dropping the target column a user's **first**
day had itself as its own context — not forecasting. Those 4,000 rows (one per
user; 0 positives) are excluded from training and flagged in the cache. The fix
moves the forecast profile share from −0.14 % to +0.24 %: still nil, so it does
not change the reading.

**Each model was previously scored on its own eligible rows.** That is the same
class of defect as the population mismatch above. All models now share one
eligible set, recorded as `shared_matched/common_eligible_ids.csv`.

## What this does not show

* **Not a pretrained model, and a small one.** These are from-scratch MLPs, so
  "architecture" here means objective and input structure, not a specific
  published model.
* **Absolute performance is weak**: day AP 0.0007–0.0019 at a prevalence of
  0.00049, and recall at a 0.1 % false-positive budget is 0.000–0.029. Nothing
  here is a usable detector, and the ROC differences should not be read as
  operational gains. The recall figures are empirical ROC operating points
  calibrated on the same negatives they are evaluated against, not deployment
  thresholds.
* **Four malicious users.** Every interval above is descriptive. No claim in
  this file rests on an interval alone.
* **No model localizes the attack day.** Whatever the user-level AUC says, the
  day these models flag hardest is not the attack day.
* Not a pretrained model. UniTS/MOMENT with pretrained weights are the next
  comparator; this is the from-scratch baseline that makes their results
  interpretable.

## Files

`shared_matched/` is the authoritative result: `pooled_summary.csv`,
`fold_rows.csv` / `fold_means.csv` (equal-malicious-user folds),
`max_day_per_malicious_user.csv`, `contrasts.csv`, `mean_gap.csv`,
`provenance.json`, and `common_eligible_ids.csv` (held on Aquaman, 141,662 ids,
md5 `07b7024ddff248e6e03c00980ec77523`).

`w1/`, `w7/`, `fc7/`: `train_manifest.json` and `checkpoint_gate.json`. The
per-model CSVs in those directories predate the shared evaluator and are kept
only for provenance; **use `shared_matched/`**.

`r62_userday_manifest.json` records the matrix provenance and verification
counts.

Reproduce: `scripts/timeseries/build_userday_matrix.py` →
`train_recon_detector.py` (once per variant) →
`scripts/timeseries/check_checkpoint.py` (gate each checkpoint) →
`scripts/timeseries/eval_channel_views.py --models w1=... w7=... fc7=...`
(one invocation, all models, one eligible set).

The superseded single-model evaluator is kept as
`eval_channel_views_HISTORICAL.py` and should not be used for new numbers.
