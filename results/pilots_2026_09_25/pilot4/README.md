# Pilot 4 (exploratory, H8): do simpler decompositions of the same deltas give the same intervention answers?

Job 20908321 (`gpu` partition, `cis260991-gpu`, one A100, 5 min 49 s =
0.10 GPU-h; its gpu-debug duplicate 20916290 was cancelled before starting).
Protocol frozen at commit `b4d2732` before scoring; endpoints E8a–c declared
in `pilot4_manifest.json` and in `scripts/analyze_pilots.py` before any
output. **Everything here is exploratory: the receivers were inspected in
package 4.** Same 197 malicious days of 29 confirmation users, their matched
benign days, donor policy and batches as Pilot 1.

## What varies and what does not

Every condition edits the tokens the SAE selected edit (rpU_S, from Pilot 1)
touches, with its per-token size, and only the direction differs, **with one
exception**: the mean-difference condition skipped 44 of the 1,396
reference-edited tokens because their rescaling gain exceeded the declared cap
(27 tokens on 27 attack receivers, 17 tokens on 16 benign receivers), and its
norm check excluded those zero edits. The PCA conditions have no such
exception. The comparison is between directions, not between decompositions
as detectors. *(Qualified 2026-09-27.)*

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

1. **Under this intervention design, the selected SAE directions influence
   the adapted model's session predictions more than the tested PCA and
   mean-difference directions.** At the same tokens and sizes (see the
   mean-difference exception above), the paired differences on attack days
   are clearly positive: SAE minus gap-ranked PCA +0.0225 [+0.0144, +0.0311],
   minus top-5 PCA +0.0213, minus lowest-gap PCA +0.0226, minus mean
   difference +0.0236 (every interval excluding zero; benign days similar).
   The simpler directions do no more than a random direction, although they
   were amplified beyond their natural size. This does not show that SAEs are
   necessary in general, identify what the directions represent, or show a
   detection benefit. *(Corrected 2026-09-27: absolute paired differences
   replace an earlier "ten times" ratio against near-zero effects.)*
2. **No attack-specific differential is detected for the SAE edit** (E8c
   +0.0006 [−0.0024, +0.0034], as in Pilot 1; not an equivalence claim). The
   only attack-day-specific response in the round comes
   from the simplest supervised baseline: a size-matched push along the
   malicious-minus-benign direction raises benign-day loss and slightly lowers
   attack-day loss (E8c −0.0042 [−0.0066, −0.0019]; two PCA sets show
   borderline differentials of −0.002). It is small in absolute terms
   (0.004 nats per token), it is one of five declared E8c comparisons, it is
   subject to the 44-token matching exception, and its edit is an amplified
   push rather than a donor replacement. It does not show improved anomaly
   detection.
3. Together with Pilot 1: the selected edit's influence comes mainly from
   feature 3673's own-support edit plus a large union-support artifact. No
   attack-specific differential is detected for it at current precision
   (E8c +0.0006 [−0.0024, +0.0034]), which is not an equivalence claim.

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
