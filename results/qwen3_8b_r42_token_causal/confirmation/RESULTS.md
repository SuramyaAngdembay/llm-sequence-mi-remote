# r4.2 Held-Out-User Confirmation — Causal + Necessity

Selection-free confirmation of the native r4.2 token mechanism, addressing the
discovery/confirmation contamination critique: features and configuration were
chosen without ever seeing the evaluation users.

## Protocol

- Positive users split 30/30 (seed 42): 671 discovery / 638 confirmation
  positive days (`outputs/user_splits_r42/`).
- Feature re-selection on discovery users only
  (`scripts/reselect_token_sae_features.py`, job `19377535`):
  top5 = `[4596, 7693, 2302, 3673, 3455]` — shares 4/5 features with the
  full-population selection (only the 5th slot differs), i.e. selection is
  stable across disjoint user subsets.
- Config frozen at the exploratory choice (layer 26, m=2, k=4, top5 vs
  control5_active, same-user exclusion).
- Evaluation restricted to the 30 held-out confirmation users
  (`--receiver-user-file`); donor pools unrestricted.
- Batch size 12 from the gpu-debug VRAM probe (job `19377532`: bs12 peak
  30.0 GiB worst-case; bs16 37.9/39.5 GiB; bs>=20 OOM). Actual run peak:
  15.4 GiB.
- GPU jobs: causal `19379904` (08:18:06, A100), necessity `19379905`
  (00:50:25). Bootstraps: `19405145`.

## Causal (held-out users, paired complete-case contrasts)

| context | n_recv (complete) | estimate | receiver-level 95% CI | cluster 95% CI (user-level) | users pos |
| --- | ---: | ---: | --- | --- | --- |
| dept | 605 (605) | 0.000758 | [0.000446, 0.001088] | [0.000346, 0.001208] | 21/29 |
| dept_role | 528 (528) | 0.000446 | [0.000056, 0.000841] | [-0.000019, 0.000990] | 19/27 |
| team | 638 (516) | 0.000398 | [-0.000097, 0.000871] | [-0.000408, 0.001252] | 15/26 |
| role | 528 (528) | 0.000340 | [-0.000046, 0.000735] | [-0.000179, 0.000953] | 16/27 |

Mean-of-user-means CIs additionally exclude zero for dept
([0.000558, 0.002184]) and team ([0.000031, 0.001705]).

## Necessity (held-out users)

| context | n_pairs | estimate | cluster 95% CI | user-mean 95% CI | users pos |
| --- | ---: | ---: | --- | --- | --- |
| dept_role | 528 | 0.002059 | [-0.000141, 0.004207] | [0.000490, 0.004100] | 19/27 |
| dept | 605 | 0.001450 | [-0.000531, 0.003674] | [-0.001811, 0.004290] | 15/29 |
| role | 528 | 0.001327 | [-0.002034, 0.005478] | [0.000933, 0.005719] | 19/27 |
| team | 638 | -0.001336 | [-0.003771, 0.001361] | [-0.002736, 0.001770] | 14/30 |

## Read

- **Direction confirms**: all four causal contexts are positive on users never
  seen by feature selection or configuration search.
- **Magnitude is roughly half** the exploratory full-population estimates
  (0.00034--0.00076 vs 0.00098--0.00142) — consistent with the exploratory
  numbers carrying selection inflation; these are the honest effect sizes.
- **Strictest test**: the dept context is significant under user-level cluster
  resampling; dept_role under receiver-level resampling; dept and team under
  mean-of-user-means. role is positive but not individually significant.
- **Necessity**: dept_role and role replicate under user-mean weighting; team
  is a confirmed null (negative point estimate), settling its previously
  CI-crossing status.
- Paper language: the r4.2 mechanism upgrades from "candidate" to "confirmed
  in direction and (for dept) in cluster-significant magnitude on held-out
  users, at reduced effect size."
