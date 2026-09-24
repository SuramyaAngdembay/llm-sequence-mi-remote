# TWOS corrected intervention, held-out users (job 20890658, 2026-09-24)

The plan, committed before submission: `docs/TWOS_CORRECTED_INTERVENTION_PLAN.md`
(`d6516b3`). The analysis is the pre-specified one,
`scripts/analyze_intervention_endpoints.py --alpha 1.0`, run inside the job and
reproduced locally with identical numbers.

## Gates, all passed before any number was read

| gate | result |
|---|---|
| run completed; per-batch reconstruction and count assertions | completed on an A100-SXM4-40GB in 12 min 17 s; nothing raised |
| selection held out | `feature_selection_manifest.json`: `held_out_from_selection: true`, ranking sha256 `12b10690…` as planned |
| feature sets as planned | top5 `[6036, 7375, 1197, 7420, 3218]`, control5_active `[7962, 972, 8082, 7232, 3157]` |
| receivers | 68 positive days of exactly the 8 confirmation users; 8,400 rows; no same-user donor |
| baseline | fresh recomputation on every row |
| tokenization across environments | identical on all 2,275 records (checked before submission) |

## Result

Context `team`, benign donors (the declared primary), alpha 1. Deltas are
per-token loss changes (patched minus fresh base; negative = lower loss). The
aggregate is the mean of 8 user means, with a 95% cluster bootstrap over users.

| view | base | selected | control | selected − control | users −/+ |
|---|---|---|---|---|---|
| behaviour only | 0.4253 | +0.0020 | +0.0021 | −0.0001 [−0.0050, +0.0052] | 3/5 |
| profile only | 3.0786 | −0.0058 | +0.0071 | −0.0129 [−0.0333, +0.0061] | 5/3 |
| full | 1.5383 | −0.0013 | +0.0045 | −0.0058 [−0.0146, +0.0018] | 4/4 |

The direct profile-minus-behaviour contrast of the difference is −0.0127
[−0.0350, +0.0081]. Every interval includes zero.

**Reconstruction.** Alpha 0 decodes the unedited code, so it measures what
replacing a patched day's deltas by their dictionary reconstruction does on
its own. The selected features are active on 43 of the 68 receiver days, the
controls on 59, so the control arm reconstructs more days.

| view | recon only, selected | recon only, control | edit net of recon, selected − control |
|---|---|---|---|
| behaviour only | +0.0019 | +0.0018 | −0.0002 [−0.0055, +0.0057] |
| profile only | +0.0013 | +0.0121 | −0.0021 [−0.0194, +0.0111] |
| full | +0.0017 | +0.0064 | −0.0011 [−0.0078, +0.0047] |

Most of the alpha-1 profile contrast (−0.0129) is the control arm's larger
reconstruction effect on profile tokens (+0.0121), not the feature edit.

Anomalous donors give the same pattern (profile −0.0149 at alpha 1, −0.0041
net of reconstruction; behaviour −0.0009 and −0.0010; all intervals include
zero). The historical best-candidate donor difference-in-differences on the
full score is −0.00075 [−0.0026, +0.0005], 2 users positive and 6 negative.

## Reading, within the plan's declared rules

- **Primary question, inconclusive.** With 8 held-out users, the selected features' edit does not measurably differ from the control edit on behaviour-token or profile-token loss. No equivalence margin was declared, so this is not evidence that the edit does nothing.
- **Reconstruction must be separated from the edit.** Without the alpha-0 arm, the alpha-1 profile contrast of −0.013 would have read as the selected features acting on profile tokens. It is mostly the controls reconstructing more days. The same risk applies to CERT, where controls are active about twice as often as the selected features.
- **Decision rule in the plan.** A seed-43 run was reserved for an effect worth testing. None was found, so it is not triggered.
- **What this is not.** It is not a replication or refutation of any CERT result, not evidence about what the features represent, and not a test with enough users to rule out a moderate effect.

## Files

`endpoints.json`, `feature_selection_manifest.json`,
`token_delta_sae_causal_summary.json` are tracked. The candidate rows
(`token_delta_sae_causal_candidate_rows.csv`, sha256
`2fbfe5f2da58d7a18fd700ecf88d93f98c67aebea12a90ee34348359145922e4`) are kept
locally and on the collaborator scratch; CSV files are not tracked in git.
