# H6 intervention-validity check: frozen protocol (2026-09-27)

Written after the TWOS code-path tests and before any CERT state or behavioural outcome of these
interventions existed. The work is exploratory: the H6 users are development data.

## Part 2: actual states

`scripts/h6_state_capture.py` recreates the H6 population and batches on an A100. It verifies
its path-A codes against H6's saved codes and records the following.

- **Codes.** Full TopK supports and values of every aligned session token's O and R codes, under
  both arithmetic paths:
  - path A: bf16 subtraction, then a float32 cast;
  - path B: float32 subtraction.
- **The H6 decoder edit** and the patched state captured after the hook.
- **Vector execution error.** ‖applied − intended‖, not a norm ratio.
- **Zero-edit checks.** An identity hook, and adding zero through H6's `put()`.

`analyze_capture.py` builds each full-code target's support directly. It counts over-k targets
by tokens, receivers, users and requested squared change, and keeps the selected-only target
separate.

## Part 3a: feasibility and projection precheck (`project_targets.py`, CPU)

**Arithmetic.** New targets and verifications use path B, which is H6's verification arithmetic.
H6's original edit (path A) is reproduced unchanged as the "dec" condition.

**Tokens.** All aligned session tokens on all 120 receivers where the path-B selected request
z_O,S − z_R,S is nonzero. Nothing is excluded. A token that fails verification stays at its
unedited context state in the I2 and I1 conditions, and it is counted.

**Targets.** Protected coordinates are all others, held at their context values.

| intervention | context | specified coordinates | full-code target |
|---|---|---|---|
| restoration (selected S) | R | S from O | τ = z_R with S replaced by z_O,S |
| restoration (control C) | R | C from O | the same with C |
| noising, for necessity (S) | O | S from R | τ = z_O with S replaced by z_R,S |

**Declared replacements.** Each changes the intervention being studied, and is disclosed wherever
it is used.

- **Over-k (swap-in).** Drop the weakest non-selected winner of the context code, repeatedly
  until k remain. The intervention becomes "restore S and remove the weakest other active
  feature".
- **Under-k (refill).** Admit the highest-preactivation non-winner of the context, excluding S,
  with a free value, until k are active. The intervention becomes "remove S and let the next
  feature enter".

**State-selection rules.**

- **I2, decoder-anchored.** The exact Euclidean projection of x_dec = x_ctx + σ ⊙ D_S(τ_S − z_ctx,S).
- **I1, minimum change.** The exact projection of x_ctx.
- Noising and the control restoration use I2 only.

**Metric.** Raw Euclidean norm per token over 4,096 dimensions. Tokens are independent because the
constraints and the metric separate.

**Margins.** The schedule is (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) in code units. Each token uses the
smallest margin whose solution verifies.

**Verification.** Cast the solution float64 → float32 → bf16 with round-to-nearest-even, as the
model receives it. Re-encode it with path B in float32. A token passes if both hold:

- the support equals the target's (fixed plus refill);
- each fixed coordinate is within max(0.05, 0.02·|τ_j|).

The pilot repeats this verification on the GPU after the hook.

**Solver certificate.** Every accepted projection satisfies the KKT conditions of the convex QP:
- primal feasibility within 1e-5 on inequalities and 1e-6 on equalities;
- nonnegative multipliers;
- complementary slackness within 1e-6.

### Precheck pass criteria (all must hold)

- **P-a, coverage.** Verified targets for at least 95% of edited tokens and at least 95% of
  requested squared change. This must hold for restoration I2, restoration I1 and noising I2.
- **P-b, interpretability.** Both of the following must hold for I2 and I1 restoration:
  - the median over edited tokens of ‖x* − x_R‖ / ‖x_O − x_R‖ is at most 1, so the "feature
    restoration" moves the state no further than restoring the whole state would;
  - the mass-weighted mean of ‖x*₂ − x_dec‖ / ‖x_dec − x_R‖ is at most 1, so the correction is
    no larger than the edit.

  If either fails, the result is "valid but hard to interpret" and there is no behavioural
  pilot.
- **P-c, solver.** The KKT certificate holds for every accepted token.

If any criterion fails, stop and report. Do not escalate.

## Part 3b: behavioural pilot (only if the precheck passes)

The pilot recreates the same batches. It first checks bitwise that the layer-26 O and R states
equal the capture; if they do not, it stops. It then writes each projected state with `put(add=False)`.

| condition | context | intervention |
|---|---|---|
| R, O | — | baselines |
| zero_R, zero_O | R, O | identity hooks (checks) |
| dec | R | H6's decoder edit (path A, added), to reproduce H6's `sel_sess` |
| I2_sel, I1_sel | R | the projected selected restorations |
| I2_ctl | R | projected control-feature restoration |
| rand_I2 | R | at the I2 tokens, an isotropic random direction per token with norm ‖x*₂ − x_R‖ |
| noise_I2 | O | the projected noising |
| rand_noise | O | a norm-matched random direction at the noising tokens |

**Outcome.** The behaviour-only loss, as defined in H6, per receiver.

**Estimands.**
- Restoration rescue: ρ = L(R) − L(R + intervention).
- Necessity: ν = L(O + noising) − L(O).

**Aggregation.**
- Mean of user means, for attack and benign days separately.
- The paired attack-minus-benign difference.
- Equivalence decisions use a 90% cluster-t interval over users, the two one-sided tests at
  α = 0.05.
- Reporting also gives a 95% user bootstrap with 10,000 draws and seed 42.

**Margin.** τ = 0.012 nats per token.
- This is the margin declared before this work: 5% of the H6 attack-day input effect of 0.243.
- It applies to the behaviour-only loss, mean of user means, over the H6 discovery users.
- It is applied to both day types. The benign-day equivalent, 5% of 0.280, would be 0.014.

**Decisions.** For ρ(I2), ρ(I1), ν(I2), and the contrasts ρ(I2) − ρ(I2_ctl), ρ(I2) − ρ(rand_I2)
and ν(I2) − ν(rand_noise):

| decision | rule |
|---|---|
| negligible | the 90% interval lies inside (−τ, τ) |
| non-negligible | the 90% interval lies outside [−τ, τ] |
| inconclusive | anything else |

If ρ(I1) and ρ(I2) fall on opposite sides (one negligible, one non-negligible), the feature
restoration effect is **not identified** by the codes.

**Claims kept separate.**
- Restoration uses ρ.
- Necessity uses ν.
- Interaction is not tested, because no joint intervention is run.
- A restoration null is not evidence against necessity.
- No result is read as evidence that 7693 matters causally or that it explains the H6 null.

**Budget.** One A100 job of at most 25 minutes. The capture and pilot together stay within the
0.54 GPU-hours left under the follow-up cap. There is no further escalation.
