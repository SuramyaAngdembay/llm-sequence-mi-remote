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

## Result — user-disjoint, on the population the language model is scored on

142,072 rows (141,662 for forecasting, which drops each user's first day — see
below), 70 positive, 4 malicious users.

| variant | view | day ROC | day AP | user ROC | within-user ROC |
|---|---|---|---|---|---|
| recon, w=1 | full | 0.6739 | 0.0010 | 0.7894 | 0.5408 |
| recon, w=1 | profile_only | 0.5927 | 0.0007 | 0.7315 | 0.4766 |
| recon, w=1 | behavior_only | 0.6856 | 0.0013 | 0.7734 | 0.5459 |
| recon, w=7 | full | 0.6830 | 0.0015 | 0.7851 | 0.5876 |
| recon, w=7 | profile_only | 0.5969 | 0.0008 | 0.8017 | 0.5792 |
| recon, w=7 | behavior_only | 0.6980 | 0.0019 | 0.7777 | 0.5806 |
| forecast | full | 0.7061 | 0.0010 | 0.7389 | 0.6300 |
| forecast | profile_only | 0.5900 | 0.0007 | 0.6773 | 0.5820 |
| forecast | behavior_only | 0.7068 | 0.0010 | 0.7389 | 0.6287 |

Mean-score-gap decomposition (exact; reconstructs the total to < 1e-9):

| variant | PROFILE contribution | BEHAV contribution |
|---|---|---|
| recon, w=1 | −0.00077 (**−3.7 %** of the gap) | +0.02154 (+103.7 %) |
| recon, w=7 | +0.00417 (**+10.1 %**) | +0.03704 (+89.9 %) |
| forecast | +0.00038 (**+0.24 %**) | +0.16139 (+99.8 %) |

### Two corrections to an earlier version of this file

**The user-level comparison was not matched.** The first evaluation kept only
the malicious users' *attack* days, while the language-model evaluation scores
**all** of their days; with max-aggregation that is a materially different
population. Corrected, user AUC for the full score is 0.789 / 0.785 / 0.739,
not the 0.477 / 0.508 / 0.296 first reported. The earlier claim that these
models "rank users near chance" was an artifact of the mismatch and is
withdrawn. `eval_channel_views.py` now defaults to the matched population.

**A higher user AUC here is not attack localization.** For **0 of the 4**
malicious users, in every model and for both the full and behaviour-only
scores, is the highest-scoring day actually an attack day. These models rank
malicious users above benign users on the strength of one of the users' *benign*
days. The evaluator now reports this count alongside user AUC, because
max-aggregated user ROC does not otherwise reveal it.

**Forecasting boundary.** `build_windows` pads a short history by repeating the
earliest available row, so after dropping the target column a user's **first**
day had itself as its own context — not forecasting. Those 4,000 rows (one per
user; 0 positives) are now excluded from training and flagged in the cache. The
fix moves the forecast profile share from −0.14 % to +0.24 %: still nil, so it
does not change the reading.

## What this does not show

* **Not a pretrained model, and a small one.** These are from-scratch MLPs, so
  "architecture" here means objective and input structure, not a specific
  published model.
* **Absolute performance is weak**: day AP 0.0011–0.0019 at a prevalence of
  0.0005, and recall at a 0.1 % false-positive budget is 0.000–0.029. Nothing
  here is a usable detector, and the ROC differences should not be read as
  operational gains.
* **Four malicious users**, so the cluster bootstrap is descriptive. The
  behaviour-only day-ROC gain is +0.012 [+0.010, +0.030] at window 1 and
  +0.015 [−0.006, +0.022] at window 7 — the second interval spans zero.
* **User-level ROC is near chance** (0.48–0.53) for every view, so the
  user-level column carries little signal either way here.
* Not a pretrained model. UniTS/MOMENT with pretrained weights are the next
  comparator; this is the from-scratch baseline that makes their results
  interpretable.

## Files

`w1/`, `w7/`: `channel_view_summary.csv` (all metrics per view),
`channel_view_contrasts.csv` (paired cluster bootstrap vs the full score),
`channel_mean_gap.csv` (signed class contributions), `channel_view_meta.json`,
`train_manifest.json`. `r62_userday_manifest.json` records the matrix
provenance and verification counts.

Reproduce: `scripts/timeseries/build_userday_matrix.py` →
`train_recon_detector.py` → `eval_channel_views.py`.
