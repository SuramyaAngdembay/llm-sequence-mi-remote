# r6.2 Leave-One-User-Out Discovery/Confirmation — Causal + Necessity

Selection-free confirmation for r6.2, where only 4 malicious users exist:
for each fold, token-SAE features are re-selected using the other 3 users'
positives only, the configuration is frozen (layer 18, m=4, k=8, top5 vs
control5_active, same-user exclusion), and effects are estimated on the
held-out user alone.

## Jobs

- VRAM probe `19406535` (gpu-debug): batch 16 recommended, 25.7 GiB worst-case
  peak (r6.2 sequences shorter than r4.2).
- Reselect `19406536` (gpu-debug, all 4 folds in one pass over the 956 GB
  cache, 27:56): fold artifacts in `reselect/`.
- Evals `19413036`--`19413043` (gpu, batch 16): total GPU time ~81 min.
- Bootstraps `19432859`.

## Feature-selection stability

Folds holding out ACM2278 / CMP2946 / MBG3183 select nearly identical top5
sets (`[14358, 12848, 4196, 13580, ...]`). The fold holding out CDE1846
(46/70 positive days) selects a **fully disjoint** top5
(`[10028, 2454, 15773, 11195/11441, ...]`): r6.2's standard feature identity
is largely defined by CDE1846's data. (Contrast r4.2, where a 30/30 user
split changed only 1 of 5 features.)

## Causal (held-out user per fold; receiver-level 95% CI)

| Held-out user | days | best context | estimate | 95% CI |
| --- | ---: | --- | ---: | --- |
| **CDE1846** | 46 | project_role | **+0.003789** | **[+0.001993, +0.005569]** |
| | | role | +0.003716 | [+0.001917, +0.005620] |
| | | dept_role | +0.003713 | [+0.001907, +0.005605] |
| ACM2278 | 5 | dept_role | +0.000047 | [-0.000305, +0.000510] |
| CMP2946 | 18 | role | +0.000083 | [-0.001171, +0.001278] |
| MBG3183 | 1 | project_role | +0.002159 | (single receiver) |

`team` is n/a in all folds (no finite anomalous-control comparison under
same-user exclusion, as in the full-population runs).

**Read: the causal mechanism passes its hardest test.** Features selected
from the other three users' 24 days produce a significant repair effect on
CDE1846's 46 held-out days, despite being a disjoint feature set. The small
users' held-out effects are near zero, consistent with the cluster-bootstrap
finding that r6.2 causal effect concentrates where the anomalous activity is.

## Necessity (held-out user per fold)

| Held-out user | best context | estimate | 95% CI |
| --- | --- | ---: | --- |
| CMP2946 | team | +0.042935 | [+0.030477, +0.055019] |
| | dept_role | +0.015428 | [+0.000864, +0.029385] |
| ACM2278 | role | +0.029945 | [+0.008443, +0.048921] |
| | project_role | +0.026859 | [+0.002466, +0.048823] |
| **CDE1846** | all contexts | **-0.0044 to -0.0080** | negative |
| MBG3183 | all contexts | -0.022 to -0.028 | (single pair) |

**Read: sufficiency transfers; necessity is user-specific.** Ablating the
disjoint alternate features does not disrupt CDE1846's anomaly evidence --
that user's necessity pathway runs through the features its own data defines.
The two multi-day small users show significant held-out necessity. Combined
with the feature-flip above, the refined r6.2 claim is: the layer-18 SAE
space contains multiple causally sufficient repair directions discoverable
from disjoint user subsets, but which features are *necessary* for a given
user's anomaly evidence is user-specific.

## Paper implications

- Limitations no longer needs "LOUO remains future work" -- replace with
  these results.
- r6.2 causal upgrades to held-out-confirmed on the dominant user; the
  small-user near-nulls stay reported.
- The sufficiency/necessity transfer asymmetry is a distinct finding worth a
  discussion paragraph; it sharpens (not weakens) the benchmark-specificity
  story: even within one benchmark, necessity structure is user-local.
