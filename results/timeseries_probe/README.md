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

## Result — user-disjoint (140,781 rows, 70 positive, 4 malicious users)

| variant | view | day ROC | day AP | user ROC | within-user ROC |
|---|---|---|---|---|---|
| recon, w=1 | full | 0.6753 | 0.0011 | 0.4765 | 0.5408 |
| recon, w=1 | profile_only | 0.5937 | 0.0007 | 0.4463 | 0.4766 |
| recon, w=1 | behavior_only | 0.6870 | 0.0013 | 0.5074 | 0.5459 |
| recon, w=7 | full | 0.6841 | 0.0015 | 0.5080 | 0.5876 |
| recon, w=7 | profile_only | 0.5976 | 0.0008 | 0.5346 | 0.5792 |
| recon, w=7 | behavior_only | 0.6992 | 0.0019 | 0.4883 | 0.5806 |
| **forecast, 7-day context** | full | 0.7020 | 0.0010 | 0.2957 | 0.6298 |
| **forecast** | profile_only | 0.5973 | 0.0006 | 0.3574 | 0.6149 |
| **forecast** | behavior_only | 0.7029 | 0.0010 | 0.2957 | 0.6291 |

Mean-score-gap decomposition (exact; reconstructs the total to < 1e-9):

| variant | PROFILE contribution | BEHAV contribution |
|---|---|---|
| recon, w=1 | −0.00077 (**−3.7 %** of the gap) | +0.02154 (+103.7 %) |
| recon, w=7 | +0.00417 (**+10.1 %**) | +0.03704 (+89.9 %) |
| **forecast** | −0.00023 (**−0.14 %**) | +0.15956 (+100.14 %) |

The forecasting variant is the one the paper's prediction actually names — the
target day is **excluded** from the input, so the model predicts it from the
user's strictly earlier days.

## Reading

**This detector does not take the profile shortcut the language model takes.**
Behaviour channels carry ~90–104 % of the positives-versus-benign score gap and
profile channels carry −4 % to +10 %. Removing the profile channels from the
score moves day ROC only 0.675 → 0.687 (window 1) and 0.684 → 0.699 (window 7).

The contrast with the 8B language model on the same benchmark is the point:
there, the profile term was *anti*-predictive (profile-only user ROC 0.200,
below chance for every malicious user individually), the full score's day ROC
sat **below chance** at 0.399, and removing the profile's direct score
contribution moved unseen-user ranking 0.532 → 0.938. Here there is little
shortcut to remove.

That is the predicted ordering: a sequence model with no pretrained language
representation sits with the tabular detectors (published r6.2: IF 0.641,
one-class SVM 0.491, PCA 0.616 — these 0.675–0.702 are at the top of that
range) rather than with the language model.

## Why — a mechanism this suggests, and how to test it

Under **forecasting** the profile contributes −0.14 % of the score gap: nothing
at all. There is a simple reason, and it may locate the shortcut's origin more
precisely than "language model versus not".

Profile channels are constant within a user. A forecasting model is given that
user's **earlier days**, so the profile is simply *copyable from context*: its
forecast error is near zero for everyone, seen or unseen, and carries no
familiarity signal. Nothing has to be memorised about *which* user this is.

The language model is in the opposite situation. Each example is a single
user-day scored independently, and the profile block sits at the **start** of
that text with no prior context to copy from. The only way to assign it high
probability is to have memorised the population distribution of profiles — so
seen users' profiles are cheap and unseen users' profiles are expensive, which
is exactly the familiarity signal the r6.2 8B decomposition shows
(`profile_only` below chance for every malicious user).

If that is right, the shortcut's origin is not "being a language model" but
**scoring each unit independently with the identity block re-predicted from
nothing**. Two cheap tests would discriminate it, neither run yet: give the LM
the user's previous day as context and see whether profile capture drops; and
score the forecasting model *without* context on the profile channels, which
should make its profile term behave like the LM's. Stated as a hypothesis with
a test, not a finding.

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
