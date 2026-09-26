# Pilot 1 (exploratory): intervention specificity and implementation decomposition

Job 20907022 (gpu-debug, `cis260991-gpu`, one A100 40 GB, 10 min 28 s =
0.17 GPU-h). Code at commit `e77b1d6` (unchanged through the run; md5-checked
on the cluster). Hypotheses H1 and H2 of `docs/HYPOTHESIS_LEDGER_2026-09-25.md`;
endpoints declared in `pilot1_manifest.json` before scoring; analysis by
`scripts/analyze_pilots.py pilot1` (committed before any output).
**Everything here is exploratory: the receivers were inspected in package 4.**

Population: 197 malicious days of 29 confirmation users (up to 10 per user,
hash order) and, for each, the same user's nearest benign day; donor policy
per department (mean prototype over 8 fixed benign colleagues); JJM0203
excluded (no colleague in department 13). Alpha 1. Loss changes are patched
minus unedited, nats per token, mean of user means, 95% user-clustered
percentile bootstrap (10,000 draws; the 5,000-draw intervals are in
`analysis_pilot1.txt` and change nothing).

## Validity (`validity.json`)

Token rows equal the tokenizer length for all 394 receivers; right padding;
the published procedure equals residual-preserving edit plus reconstruction
error to 7.6e-06; a true zero edit reproduces the unhooked model exactly
(max abs 0.0). Every condition was scored in the same batches.

## Measured edits (`edit_balance.json`, `coefficient_changes.json`)

Selected-set edits touch 3.5 tokens per day (1.2 organisation-line, 2.3
session), total 205 residual units; control edits 4.8 tokens, 50 units. The
random baselines copy the selected edit's positions and per-token norms. The
package-4 prototype removes 93–95% of the three session features where they
fire (4596: 24.2 → 1.65; 3673: 19.1 → 1.16; 3455: 16.4 → 0.77) and writes
about 21 units of feature 2302 into session positions where it was zero.

## Results, behaviour tokens, malicious days

| condition | loss change |
|---|---|
| orig_S (published procedure) | +0.0642 [+0.0511, +0.0786] |
| orig_C | +0.0440 [+0.0302, +0.0598] |
| recon_S = recon_C (reconstruction only) | +0.0441 [+0.0302, +0.0600] |
| **rpU_S** (residual-preserving, union support) | **+0.0213 [+0.0135, +0.0297]** |
| rpO_S (own support) | +0.0091 [+0.0054, +0.0130] |
| randI_S (isotropic, matched) | +0.0008 [−0.0007, +0.0024] |
| randD_S (other dictionary directions, matched) | +0.0041 [+0.0016, +0.0070] |
| rpU_C / rpO_C / randI_C / randD_C | −0.0001 / +0.0001 / −0.0003 / −0.0001 |
| single 3673 (session duration) | +0.0097 [+0.0059, +0.0139] |
| single 4596 | +0.0014 [+0.0003, +0.0028] |
| single 3455 | −0.0029 [−0.0050, −0.0011] |
| single 2302 (own support = organisation token) | +0.0002 [−0.0001, +0.0006] |
| single 7693 | 0 (fires on 2 days) |

| endpoint | malicious days | matched benign days |
|---|---|---|
| E1a rpU_S − randI_S | **+0.0205 [+0.0131, +0.0284]**, 26/29 users | +0.0199 [+0.0114, +0.0286] |
| E1b rpU_S − randD_S | **+0.0172 [+0.0108, +0.0239]** | +0.0163 [+0.0097, +0.0232] |
| E2 rpU_C − random | +0.0003 [−0.0001, +0.0006] | −0.0002 [−0.0005, +0.0001] |
| E3 rpU_S − rpU_C | +0.0214 [+0.0136, +0.0297] | +0.0234 [+0.0138, +0.0334] |
| E4 orig − rpU − recon (interaction) | S: −0.0012 [−0.0034, +0.0009]; C: 0.0000 | S: −0.0031 [−0.0049, −0.0014] |
| E5 rpU_S − rpO_S (writing into zero coordinates) | **+0.0122 [+0.0077, +0.0170]** | +0.0109 [+0.0066, +0.0156] |
| E7 rpO_S − Σ singles (interaction) | +0.0007 [+0.0003, +0.0013] | −0.0003 [−0.0005, −0.0001] |
| orig_S − orig_C (published procedure, this donor policy) | +0.0202 [+0.0124, +0.0282] | +0.0202 [+0.0115, +0.0293] |
| **E6 [rpU_S − rpU_C], malicious minus benign** | **−0.0020 [−0.0058, +0.0014]** | |
| E6b [rpU_S − randI_S], malicious minus benign | +0.0006 [−0.0024, +0.0034] | |

Profile tokens: rpU_S +0.0031 [−0.0059, +0.0158]; every profile-token
endpoint includes zero.

## Reading (against the interpretation map written before the results)

1. **Direction-specific, not generic disruption.** Equally large edits at the
   same tokens do almost nothing when their direction is random (isotropic
   +0.0008) and a fifth as much when drawn from other dictionary directions
   (+0.0041). The H1 stopping rule ("within 20% of the selected effect with
   an interval covering zero") is not triggered for either baseline.
2. **More than half of the effect is the implementation, not the model.**
   Restricting each feature to its own active positions removes 57% of the
   effect (E5 = +0.0122 of +0.0213). The difference is what union support
   writes into coordinates that were zero, by far the largest of which is
   feature 2302's organisation-line value (about 21 units) written into
   session positions; writes of 1–2 units of the session features into the
   organisation token are the rest and are not separated here.
3. **The model part is essentially one feature.** Of the own-support effect
   (+0.0091), feature 3673 alone gives +0.0097: removing 94% of the
   session-duration feature's activation makes the adapted model worse at
   predicting session tokens. Feature 3455's removal *improves* prediction
   (−0.0029), 4596 contributes +0.0014, 2302 nothing at its own positions,
   and 7693 does not fire. The joint edit is nearly additive (E7 +0.0007).
4. **Not specific to malicious days.** Benign days of the same users respond
   the same way (rpU_S +0.0232 vs +0.0213; E6 −0.0020 [−0.0058, +0.0014]).
   The directions carry information the adapted model uses to predict session
   tokens on any day: ordinary behavioural prediction, not anomaly content.
5. **Reconstruction interaction is small** (E4 −0.001 to −0.003): the
   published contrast is not an artifact of the reconstruction replacement,
   although absolute published effects are dominated by it (+0.044 of +0.064).
6. **The published estimand reproduces under this donor policy** (orig_S −
   orig_C +0.0202, against +0.016 to +0.018 in package 4 with single donors).

What this does not establish: what feature 3673 represents (its edit lowers
prediction quality on session tokens; that is causal influence on prediction,
not represented content); any detection benefit; anything about profile
tokens; a circuit. Random baselines draw an independent direction per token,
whereas the selected edit is coherent across tokens; a coherent random
direction would be a stronger control and was not run.

## Files

`pilot1_rows.csv` (all conditions, per receiver, per view), `edit_balance_rows.csv`,
`edit_balance.json`, `coefficient_changes.json`, `validity.json`,
`pilot1_manifest.json`, `analysis_pilot1.{json,txt}`, the job log.
