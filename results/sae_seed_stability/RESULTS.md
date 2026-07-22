# SAE Seed-Stability Controls — Non-Identifiability Answer

Addresses: "cross-benchmark transfer failure could be SAE non-identifiability
rather than benchmark-specificity." Two controls, both complete.

## 1. Decoder-space alignment (all SAEs decode into the same d_model=4096)

Seeds 43/44 retrained per benchmark on the SAME cached deltas (adapter
untouched; jobs 19433591-94, ~26 min r4.2 / ~48-56 min r6.2 each). Metric:
for each source SAE's top-5 features (by row_gap), best-match |cos| into the
target dictionary, vs whole-dictionary median as the chance baseline.

| pair | top5 mean best-match | chance baseline |
| --- | ---: | ---: |
| r6.2 s42<->s43 | 0.888 / 0.929 | 0.586 |
| r6.2 s42<->s44 | 0.881 | 0.579 |
| r6.2 s43<->s44 | 0.927 | 0.579 |
| r4.2 s42<->s43 | 0.881 / 0.955 | 0.645 |
| r4.2 s42<->s44 | 0.922 | 0.638 |
| r4.2 s43<->s44 | 0.955 | 0.635 |
| **r6.2 -> r4.2 (any seed)** | **0.079--0.112** | 0.075--0.093 |

Within-benchmark cross-seed alignment is far above chance on every pair;
cross-benchmark alignment is AT (once below) the chance floor in both
directions across three seeds per side. The reviewer-prescribed signature
`same-benchmark cross-seed >> cross-benchmark` holds decisively.

Caveat to state: the cross-benchmark comparison is at each benchmark's native
layer (18 vs 26); the matched-depth case is covered causally by the direct
transfer experiment (r6.2 layer-18 SAE on r4.2 layer-18 deltas, negative).

## 2. Cross-seed causal rediscovery (r6.2, jobs 19433619/20)

Fresh-seed SAEs + their own re-selected features, full protocol (top5 vs
control5_active, same-user excluded, 70 receivers):

| context | seed 42 (orig) | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| role | 0.00685 | 0.00713 | 0.00188 |
| dept_role | 0.00682 | 0.00493 | 0.00306 |
| project_role | 0.00420 | 0.00483 | 0.00194 |
| team | n/a | n/a | n/a |

Positive under every seed in every finite context. Native rediscovery is
seed-robust (magnitudes vary ~2-3x across seeds, sign never flips).

## Interpretation (post-attribution)

Combined with the feature-token attribution, these controls say: the r6.2
profile-novelty directions and the r4.2 behavioral directions are each
stable, rediscoverable subspaces of their own model's residual stream — and
they have no counterpart in the other benchmark's dictionary. The
benchmark-specificity is real, mechanistically explained (different positive-
population structure -> different signal type), and not an artifact of SAE
training randomness.
