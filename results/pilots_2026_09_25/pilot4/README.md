# Pilot 4 (exploratory, H8): is the sparse dictionary necessary?

Job 20908321 (`gpu` partition, `cis260991-gpu`, one A100, 5 min 49 s =
0.10 GPU-h; its gpu-debug duplicate 20916290 was cancelled before starting).
Protocol frozen at commit `b4d2732` before scoring; endpoints E8a–c declared
in `pilot4_manifest.json` and in `scripts/analyze_pilots.py` before any
output. **Everything here is exploratory: the receivers were inspected in
package 4.** Same 197 malicious days of 29 confirmation users, their matched
benign days, donor policy and batches as Pilot 1.

## What varies and what does not

Every condition edits exactly the tokens the SAE selected edit (rpU_S, from
Pilot 1) touches, with exactly its per-token size. Only the direction of the
edit differs, so the comparison is between directions, not between
decompositions as detectors.

| condition | direction |
|---|---|
| rpU_S | the five selected SAE features, residual-preserving, union support |
| randI_S / randD_S | isotropic random / other dictionary directions (as Pilot 1) |
| pcaTop5 | the 5 leading principal directions of the standardized deltas |
| pcaGap5 | the 5 of the leading 64 with the largest malicious-minus-benign gap |
| pcaCtrl5 | the 5 of the leading 64 with the smallest gap |
| meanDiff | the malicious-minus-benign mean-difference direction |

The directions were fitted on the SAE's own populations
(`manifests/pilots_2026_09_25/baseline_directions.json`): PCA on a stride-4
sample of every eval token row (2.97 M rows, 149 users); gap and mean
difference on the 158,065 discovery-user positive rows the SAE ranking used
and 9.7 M benign rows of non-confirmation users; no confirmation user enters
either. Each baseline edit moves the token's projection toward the benign
donors' prototype and is then rescaled to the SAE edit's size.

## Validity (`validity.json`)

Token alignment for all 394 receivers; zero edit equals the unhooked model
exactly; per-token norms equal the SAE edit's to 2.4e-7. The conditions shared
with Pilot 1 (zero, rpU_S, randI_S, randD_S) reproduce Pilot 1's per-receiver
losses to within 3.6e-7 although the two jobs ran on different nodes and
partitions. Rescaling gains (reference size over natural edit size), median
[p95]: pcaGap5 1.25 [2.5], pcaTop5 1.8 [3.9], pcaCtrl5 4.5 [8.4], meanDiff
13.2 [76]; 3.2% of meanDiff tokens exceeded the declared cap of 100 and were
left unedited. So every baseline received at least its natural edit size,
often several times more.

## Results, behaviour tokens (nats per token, mean of user means, 95% user bootstrap)

| edit alone | malicious days | matched benign days |
|---|---|---|
| **rpU_S (SAE)** | **+0.0213 [+0.0135, +0.0297]** | +0.0232 [+0.0136, +0.0333] |
| randI_S | +0.0008 [−0.0007, +0.0024] | +0.0033 [+0.0017, +0.0051] |
| randD_S | +0.0041 [+0.0016, +0.0070] | +0.0070 [+0.0032, +0.0113] |
| pcaTop5 | −0.0000 [−0.0023, +0.0022] | +0.0046 [+0.0024, +0.0072] |
| pcaGap5 | −0.0011 [−0.0026, +0.0002] | +0.0014 [+0.0003, +0.0026] |
| pcaCtrl5 | −0.0013 [−0.0027, −0.0000] | +0.0035 [+0.0013, +0.0059] |
| meanDiff | −0.0022 [−0.0047, −0.0000] | +0.0045 [+0.0023, +0.0068] |

| endpoint | value |
|---|---|
| E8a baseline − rpU_S, malicious | −0.0214 to −0.0236 for all four baselines, every interval excluding zero (27–28 of 29 users negative) |
| E8b baseline − randI_S, malicious | pcaTop5 −0.0008 [−0.0025, +0.0007]; pcaGap5 −0.0019 [−0.0035, −0.0004]; pcaCtrl5 −0.0021 [−0.0041, −0.0004]; meanDiff −0.0031 [−0.0053, −0.0010] |
| **E8c [baseline − randI_S], malicious minus benign** | **meanDiff −0.0042 [−0.0066, −0.0019]**; pcaTop5 −0.0021 [−0.0044, −0.0001]; pcaCtrl5 −0.0022 [−0.0047, −0.0001]; pcaGap5 −0.0000 [−0.0019, +0.0017]; rpU_S (SAE) +0.0006 [−0.0024, +0.0034] |

Profile tokens and the full score: every baseline's interval includes zero.
The 5,000-draw intervals (`analysis_pilot4.txt`) change no conclusion.

## Post-hoc diagnostic (`edit_sign_diagnostic.json`, dated 2026-09-26, after the results)

The malicious-minus-benign contrast could arise if the edit were built
differently for the two groups, since each edit moves a token from wherever
it is toward the donors' value. It is not: along the mean-difference
direction, 54% of malicious-day and 58% of benign-day edited tokens are moved
toward the benign side (per-user medians 50% and 56%), the mean coefficient
change is −2.54 and −2.60, and the natural edit sizes are 9.2 and 9.3; the
PCA sets' natural sizes agree between groups within 3%. But because the
meanDiff edit is amplified about 13-fold to the SAE edit's size, it overshoots
the donors' value by roughly an order of magnitude: it is a size-matched push
along ±d, not a move to the benign value.

## Reading

1. **For changing the adapted model's session predictions, the dictionary
   matters.** At the same tokens and sizes, the SAE edit raises behaviour
   loss ten times more than any principal-component or mean-difference
   direction, and those simpler directions do no more than a random
   direction. This holds although the baselines were amplified beyond their
   natural size. The obvious reviewer objection, that the SAE adds nothing a
   simpler decomposition would find, is not supported for this purpose.
2. **For anything anomaly-specific, it does not.** The SAE edit's effect is
   the same on attack days and on the same users' ordinary days (E8c +0.0006,
   as in Pilot 1). The only attack-day-specific response in the round comes
   from the simplest supervised baseline: a size-matched push along the
   malicious-minus-benign direction raises benign-day loss and slightly lowers
   attack-day loss (E8c −0.0042 [−0.0066, −0.0019]; two PCA sets show
   borderline differentials of −0.002). The effect is small (a fifth of the
   SAE edit's generic effect), it is one of five declared E8c comparisons,
   and its edit is an amplified push rather than a donor replacement.
3. Together with Pilot 1: the SAE isolated a direction the adapted model
   genuinely uses to predict session content (mainly feature 3673, plus a
   large union-support artifact), which is why its edits are potent. That
   potency reflects ordinary behavioural prediction. It is not what separates
   attack days from ordinary days.

What this does not establish: that meanDiff carries anomaly *information*
(the contrast is a differential sensitivity to a size-matched push at tokens
chosen by the SAE, with two groups of days from the same users); any detection
benefit; anything about positions other than the SAE's edited tokens. A test of
the mean-difference direction at its own most-active positions, with a
non-amplified (replacement-size) edit and a fresh confirmation population,
would be needed before any claim.

## Files

`pilot4_rows.csv`, `edit_balance*.{csv,json}`, `validity.json`,
`pilot4_manifest.json` (with prototypes and PC sets),
`edit_sign_diagnostic.json` (post-hoc), `analysis_pilot4.{json,txt}`, the job
log.
