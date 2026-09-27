# Frozen evaluation protocol (2026-09-27, before any evaluation world existed)

Configuration: `config_frozen.json`. Code: `synth.py` at the commit that adds this file.
Development runs: `results/dev_percentile`, `results/dev_t`, `results/dev_t30`. Threshold tuning:
`results/p2_threshold_tuning.json`.

## Worlds and truth

- The case definitions are the ones in `config_dev.json`. Evaluation uses six fresh world
  seeds (2001 to 2006). Each seed draws a new dictionary, new scales, a new downstream readout
  and new units.
- **Primary estimand.** θ2 is the mean over clusters of the per-unit rescue under the
  decoder-anchored ideal I2. Truth is computed on the whole 300-cluster pool of each world by
  executing I2 exactly.
- **Secondary estimand.** θ1 is the same quantity under the minimum-change ideal I1.
- **Margin.** τ = 0.05 × |mean full-restoration rescue| of the world.
- **Truth classes.** θ2 is negligible if |θ2| < τ and non-negligible otherwise. It is
  ill-posed if any unit's full-code target is empty. For P5, a world is lift-dependent if θ1
  and θ2 fall on different sides of τ.

## Procedures and decisions

Each dataset holds 30 clusters of 3 units. Intervals are 90% cluster-t intervals over
cluster means, the two one-sided tests at α = 0.05 each.

| decision | rule |
|---|---|
| negligible | the interval lies strictly inside (−τ, τ) |
| non-negligible | the interval lies entirely outside [−τ, τ] |
| inconclusive | anything else |

The procedures:

- **P0**: calls the effect negligible if the 95% interval of the applied rescue contains zero.
  This is the common practice, reported for contrast.
- **P1**: tests the applied rescue for equivalence.
- **P2**: gives P1's decision if the mass-weighted realization error is at most 0.05, and
  inconclusive otherwise.
- **P3**: directly executes I2 and tests its rescue. It is ill-posed if any target is empty.
- **P4a**: widens P1's interval by B_i = K_i × ‖x̂_i − x*_i‖, with a certified regional K.
  The lower end is the lower bound of mean(e − B) and the upper end the upper bound of
  mean(e + B). It is ill-posed if any target is empty.
- **P4b**: as P4a with the sampled K. That K is not a bound, and P4b is reported as a
  diagnostic.
- **P4c**: as P4a with ‖x̂ − x_dec‖ + H_eq‖B x_dec − d‖. It is unavailable when the
  inequalities bind.
- **P5**: runs P3 for both I1 and I2. It reports "not identified" when one is negligible and
  the other non-negligible.

## Metrics

Every metric is taken per condition over 300 datasets (6 worlds × 50):

- decision shares;
- false-negligible rate among datasets with non-negligible truth;
- false-non-negligible rate among datasets with negligible truth;
- correct-decision rate;
- coverage of θ2 by each interval;
- median width.

## Success criteria for the proposed procedure (P4a)

All four must hold for P4a to count as useful:

1. **Validity.** Its false-negligible rate is at most 0.05 in every condition.
2. **Informativeness.** It is correct and decisive in at least 50% of datasets in at least
   half of the conditions whose realization is inexact. Those conditions are the ones whose
   mean realization error exceeds 0.01.
3. **Improvement over simpler checks.** Its wrong-plus-inconclusive rate, pooled over
   conditions with a defined estimand, is lower than P2's.
4. **Value beyond direct execution.** It is decisive in some condition where P3 is
   unavailable. In these synthetic worlds P3 is always available when P4a is, so this
   criterion can only be met through an argument about settings where the ideal cannot be
   executed. Such an argument must be stated explicitly and is not a measurement.

If criteria 1 to 3 fail, the result is recorded as negative for P4a as a decision method.
