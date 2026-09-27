# Pilot 2 (exploratory): identity context, familiarity, and what the selected features track

Job 20916308 (gpu-debug, `cis260991-gpu`, one A100, 3 min 48 s = 0.06 GPU-h;
its gpu-partition duplicate 20921140 was cancelled before starting). The first
attempt, 20907024, failed after 73 s on a tensor carrying gradient (0.02
GPU-h). The fix detached it and changed nothing else. Code as committed in
`11640f2` plus the `no_grad` fix (`257d4a6`). Hypotheses H3, H4 and H7 of
`docs/HYPOTHESIS_LEDGER_2026-09-25.md`; analysis by
`scripts/analyze_pilots.py pilot2|pilot3`, committed before any output.
**Exploratory: the receivers were inspected in package 4.**

**Population and swaps.** 207 attack days of the 30 confirmation users, each
with the same user's nearest benign day. Each day was scored four times:

- original;
- DAY and PSY lines swapped for a same-department never-malicious eval
  user's (`swap_same`; not possible for department 13);
- swapped for a different-department eval user's (`swap_other`);
- swapped for a same-department **training** user's (`swap_train`; familiar
  to the adapter, since no eval user is).

The receiver's week value and every session line are kept. Each input was
scored with the adapter on and off.

**Validity.** On-the-fly deltas match the cache: 3.3% median relative
difference and SAE active-set Jaccard median 1.0 (16 receivers, 3,280 tokens).

## Results (behaviour tokens; swap minus original; mean of user means, 95% user bootstrap)

| swap | adapted model | base model | F1: adapted − base |
|---|---|---|---|
| same department, unfamiliar | +0.123 [+0.089, +0.160] | +0.006 | **+0.117 [+0.083, +0.155]**, 25/29 users |
| other department | +0.183 [+0.149, +0.218] | +0.007 | **+0.176 [+0.141, +0.212]**, 30/30 |
| same department, training user (familiar) | +0.112 [+0.081, +0.146] | +0.004 | **+0.109 [+0.076, +0.144]**, 27/30 |

These are malicious days. The same users' benign days respond the same way:
F1 is +0.128, +0.196 and +0.120. F2 (attack minus benign day) is −0.008 to
−0.020, and only the training-user swap excludes zero, at −0.011
[−0.022, −0.001].

- **F4, identity familiarity** (training-user minus unfamiliar same-department
  swap, adapted minus base): −0.004 [−0.031, +0.022] on attack days and
  −0.003 [−0.040, +0.032] on benign days. **No detectable familiarity effect
  was found under the tested profile substitutions.** The interval crossing
  zero is not an equivalence test. The two partners also differ in attributes
  other than familiarity (department alone is matched), so this contrast does
  not isolate familiarity.
- **F3, selected-feature activity on session tokens** (swap minus original,
  selected minus control): −0.0005 [−0.0016, +0.0006] for the other-department
  swap. This statistic is a signed change in each record's *mean* activation
  over session positions, and it cancels across tokens, features and records.
  Record-level signed and absolute changes, separately (other-department swap;
  saved per-record means, 414 records):

  | feature | signed change | absolute change | original mean | absolute / original | records changed |
  |---|---|---|---|---|---|
  | 4596 (sel) | −0.0011 | 0.0028 | 0.112 | 2.5% | 58% |
  | 3673 (sel) | −0.0004 | 0.0039 | 0.090 | 4.4% | 81% |
  | 3455 (sel) | +0.0002 | 0.0055 | 0.067 | 8.2% | 69% |
  | 2302, 7693 (sel) | ≈0 | ≈0 | < 0.001 | not meaningful | 3–7% |
  | controls (mean of 5) | | 0.0026 | | | |

  Selected and control features change by similar absolute amounts (0.0027
  versus 0.0026). The ratios for rare features are large only because their
  means are tiny. Token-level trajectories were not saved, so cancellation
  within a record cannot be assessed here. **The tested average
  session-position activation statistic did not detect a clear
  selected-versus-control response; mediation remains untested.**
  *(Corrected 2026-09-27; the earlier text said the per-feature changes were
  "about 1% or less", which holds only for signed means.)*
- **H4, within-field associations** (`analysis_pilot3.txt`; session positions
  predicting a session token, stratified by next-token id, per user). The
  results differ by feature:

  | feature | eligible users (attack / benign) | adapted loss, active − inactive (attack) | base loss (attack) | adapted − base (attack) | adapted − base (benign) |
  |---|---|---|---|---|---|
  | 4596 (sel) | 12 / 12 | **−0.83 [−1.17, −0.53]**, 12/12 users | +0.22 [+0.03, +0.42] | −1.04 [−1.31, −0.78] | −0.71 [−1.03, −0.40] |
  | 3673 (sel) | 8 / 7 | −0.000 [−0.001, +0.000] | +2.67 [+0.39, +5.71] | −2.67 [−5.71, −0.39] | −1.79 [−3.74, −0.12] |
  | 3455 (sel) | 15 / 15 | **+2.33 [+1.55, +3.16]** | +0.27 | +2.07 [+1.27, +2.92] | +0.20 [−0.13, +0.52] |
  | 2302, 7693 (sel) | 0 / 0 | no eligible within-token comparisons | | | |
  | 6596 (ctl) | 17 / 17 | −0.000 | −7.61 | +7.61 [+5.63, +9.54] | +8.03 |

  Selected set, adapted minus base, on attack days: **+0.20 [−0.71, +1.02]
  (inconclusive)**; on benign days −0.46 [−0.93, −0.09]. The selected-minus-
  control contrast (−3.5 [−5.2, −1.8]) is driven largely by the controls,
  above all 6596, and does not show a selected-set-wide gain.

## Reading (corrected 2026-09-27)

1. **Adaptation substantially increased behaviour prediction's sensitivity to
   the profile lines.** The base model barely conditions on them. Replacing a
   person's organisation and personality lines raises the adapter's
   session-token loss by 0.11 to 0.18 nats per token, and more for a
   different department. This is profile dependence, or profile-conditioned
   prediction, created by fine-tuning, on attack and ordinary days alike. It
   is not shown to be a harmful shortcut: department and role can
   legitimately inform predictions of normal behaviour, and whether the
   dependence hurts detection or robustness was not tested here.
2. **No detectable familiarity effect** under these substitutions (see F4).
   This does not exclude memorisation.
3. **Mediation by the selected features remains untested.** The average
   activation statistic did not detect a selected-versus-control response.
   That is not evidence that the influence bypasses the features.
4. **Feature 4596 has a reproducible association with lower adapted-model
   loss** at its active positions, beyond the base model. Other selected
   features behave differently (3455 goes the opposite way), so this does not
   generalise to the selected set. These are associations conditional on the
   next token, not identified functions or causal effects. H4 as posed
   (surprise tracking) is not supported for 4596; at the level of the whole
   selected set the association is inconclusive.

Not established here: where the dependence is computed. The H6 pilot
(`results/h6_2026_09_27/README.md`, development users) has since localized
it. The information comes from the organisation line and enters session
positions in blocks 22 to 25. It is carried by the layer-26 adapter delta, not
by the 5 selected coefficients. Also not established: any attack-specific
separation, or a detection benefit.

## Files

`pilot2_rows.csv`, `pilot2_orig_tokens.npz`, `validity.json`,
`pilot2_manifest.json`, `analysis_pilot2.{json,txt}`,
`analysis_pilot3.{json,txt}`, the job log.
