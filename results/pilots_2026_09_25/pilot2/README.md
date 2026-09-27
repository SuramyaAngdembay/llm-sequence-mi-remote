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
  −0.003 [−0.040, +0.032] on benign days. No familiarity effect.
- **F3, selected-feature activity on session tokens** (swap minus original,
  selected minus control): −0.0005 [−0.0016, +0.0006] for the other-department
  swap. The per-feature changes are about 1% of the mean activation or less.
- **H4, within-field association** (`analysis_pilot3.txt`; session positions
  predicting a session token, stratified by next-token id). Where feature 4596
  fires, next-token loss under the adapted model is 0.83 nats lower
  [0.53, 1.17] (12 of 12 users, both day types). Under the base model it is
  not lower (+0.22 and −0.05). Across the selected set, active positions carry
  a much larger adaptation gain (adapted minus base loss) than the controls
  (−3.5 [−5.2, −1.8]).

## Reading

1. **Adaptation made behaviour prediction depend on the profile lines.** The
   base model barely conditions on them. Replacing a person's organisation and
   personality lines raises the adapter's session-token loss by 0.11 to 0.18
   nats per token, which is larger for a different department. This is input
   sensitivity created by fine-tuning, on attack and ordinary days alike.
2. **It is not identity memorisation.** A profile the adapter was trained on
   disrupts no more than an unseen colleague's. What matters is how different
   the attributes are (the department effect).
3. **The profile's influence does not pass through the selected features.**
   Their activity on session tokens hardly changes under swaps that move the
   loss by 0.1 to 0.2 nats.
4. **The selected features mark adapter-learned predictive regularities.**
   They fire where the adapter predicts much better than the base model, not
   where it is surprised. H4 as posed (surprise tracking) is rejected in
   favour of this reading. It matches Pilot 1, where removing 3673 or 4596
   hurts prediction.

Not established: where in the network the profile dependence is computed (H6,
mediation or path patching, not run); whether any of this separates attack
days from ordinary days (it does not here); a detection benefit.

## Files

`pilot2_rows.csv`, `pilot2_orig_tokens.npz`, `validity.json`,
`pilot2_manifest.json`, `analysis_pilot2.{json,txt}`,
`analysis_pilot3.{json,txt}`, the job log.
