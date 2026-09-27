# Method status

**Status: Deferred. The current generic bound is not informative enough.**

This covers the implementation-aware equivalence interval: the applied-effect interval widened
by K × ‖applied − declared ideal‖, followed by an equivalence decision at a pre-declared margin.
It is parked, not refuted. Nothing below is a paper claim.

## Preserved

| item | where |
|---|---|
| Frozen protocol, committed as `8aa4fcd` before evaluation | `FROZEN_PROTOCOL.md`, `config_frozen.json` |
| Code and tests (7 tests pass; an independent reproduction matched all 48,000 rows) | `synth.py`, `test_synth.py` |
| Development and evaluation outputs | `results/dev_*`, `results/eval/` |
| Estimand, derivations, counterexamples A–F and failed approaches | `ESTIMAND_AND_DERIVATIONS.md` |
| Prior-art comparison | `PRIOR_ART.md` |
| Applied H6 validity work | `h6_states/` |

## Valid mathematical results (proved under stated assumptions)

1. **Full-code fibres.** A full-code target's margin-qualified fibre is one polyhedron.
   Over-k targets are empty at any margin. A positive-margin infeasibility does not prove
   zero-margin impossibility.
2. **Hoffman and declared ideals.** A Hoffman bound controls the distance to the feasible
   *set*. For a decoder-anchored ideal, the triangle inequality gives
   ‖x̂ − x*₂‖ ≤ ‖x̂ − x_dec‖ + dist(x_dec, P).
3. **Exact distance.** The exact distance decomposes into an equality part and a
   least-distance part, given an orthonormal null-space basis. It is at most any Hoffman bound.
4. **Validity of the widened test.** It controls false-negligible declarations for the ideal
   effect when three things hold:
   - K is a valid regional Lipschitz bound on a convex set containing x̂ and x*;
   - the ideal is precisely declared;
   - the confidence bounds are valid.
5. **Accuracy needed for near-zero decisions.** A near-zero equivalence decision needs
   E[K‖x̂ − x*‖] < τ − |E e| − (sampling half-width).

## Limitations of this implementation

- **K is worst-case over all directions** in a ball, so it ignores the direction of the actual
  correction.
- **Only the equality-subsystem Hoffman bound was implemented.** No full-system Hoffman constant
  was attempted.
- **Only one sensitivity certificate was used**, a product of norms in the synthetic readouts. No
  tight, tractable regional bound was established for the Qwen suffix and loss used in H6.
  Bounded-domain attention bounds exist (Castin et al., ICML 2024; Yudin et al., 2025; cited from
  the review, not evaluated here).
- **Cluster-t intervals are approximate.** Zero observed false negligibles is not a finite-sample
  guarantee.

## Benchmark-specific findings

These were measured in synthetic worlds with 12 features in 20 dimensions and deliberately
invisible directions. They should not be generalized to real SAEs.

- **Decision counts** over 5,700 datasets with a defined ideal:

  | procedure | correct | incorrect | inconclusive |
  |---|---|---|---|
  | widened bound | 31.5% | 0% | 68.5% |
  | applied-effect test | 81.2% | 15.8% | 3.0% |
  | realization gate | 37.5% | 0% | 62.5% |
  | direct execution | 98.0% | 0% | 2.0% |

- The widened bound was informative in 1 of 14 inexact conditions.
- Near-zero decisions needed the applied state within 0.3% to 0.9% of the edit's length from
  the ideal. Large effects were decidable with more error, as in condition C2b.
- The two lifts disagreed in several worlds. A restoration null coexisted with necessity and an
  interaction.

## Unresolved questions

- Would a direction-sensitive or structure-restricted bound be informative on real overcomplete
  SAE geometry? The synthetic geometry is not representative.
- Is there a setting where direct execution of the ideal is costly or unavailable while a bound
  is still computable? Criterion 4 of the frozen protocol was never tested.
- How large are the correction distances and arithmetic floors on real states?
  `h6_states/` measures these for H6 and should inform any reopening.

## Reopening routes (at most two)

### Route A: second-order bounds along the actual correction direction

- **Bottleneck addressed.** The first-order bound K‖d‖ is worst-case over directions and scales
  linearly in ‖d‖, where d = x* − x̂.
- **Idea.** Compute the first-order term ∇F(x̂)·d exactly. One backward pass per unit covers
  every nearby candidate ideal. Then bound only the remainder
  |F(x*) − F(x̂) − ∇F(x̂)·d| ≤ ½ M ‖d‖², where M is a certified bound on the curvature along the
  segment. For small corrections, ‖d‖² can be far below ‖d‖.
- **Assumptions.**
  - F is twice differentiable along the segment. That holds for the RMSNorm, attention,
    activation-function and softmax suffix, but not across TopK.
  - M is certified on the segment.
  - A sampled Hessian-vector product would be a *heuristic*, not a certificate.
- **Nearest prior work to check.** Attribution patching (Nanda 2023; Syed et al. 2023; Kramár et
  al. 2024, AtP\*) uses the first-order term without certified remainders. It is still unchecked
  whether error bounds for attribution patching already exist.
- **Smallest falsifiable experiment.** In the frozen synthetic benchmark, add a procedure using
  the exact gradient plus an analytic curvature bound, which exists for the score readouts. Freeze
  its decision rule, run it on fresh evaluation seeds, and compare three-way decisions with P1,
  P2, P3 and P4a.
- **Stopping criterion.** Stop if, at zero false negligibles, it is decisive in fewer than half
  of the inexact conditions. Also stop if no certified M can be obtained for the Qwen suffix
  within a bounded effort of one CPU-day.

### Route B: certified bounds over a restricted family of lifts

- **Bottleneck addressed.** A lift picks one state from a fibre, and different lifts can behave
  differently (counterexamples B and E). Direct execution evaluates one point and cannot certify a
  whole family.
- **Idea.** Restrict the ideal to a declared low-dimensional family. An example is the feasible
  states in the span of the selected and replaced features' decoder atoms, within a bounded box.
  Bound the loss range over that family by bound propagation over a low-dimensional input domain.
  That yields a certified interval covering all lifts in the family.
- **Assumptions.**
  - The restriction must be scientifically justified.
  - The domain must be bounded.
  - Bound propagation through the Qwen suffix, with NF4 weights, must be tractable at usable
    tightness for low-dimensional inputs. This is unknown.
- **Nearest prior work to check.** Bound propagation for transformers (Shi et al., ICLR 2020;
  DeepT, Bonaert et al. 2021; auto_LiRPA, Xu et al. 2020). On the lower-bound side, the
  SAE-unreliability paper optimizes behaviour within feature constraints. It is unchecked whether
  certified intervention families already exist in the interpretability literature.
- **Smallest falsifiable experiment.** Take TWOS (Qwen2.5-3B) or a two-block toy transformer, and
  20 edited tokens with a two-dimensional family. Compare the certified loss range with the
  empirical range from dense sampling plus optimization.
- **Stopping criterion.** Stop if the certified width exceeds ten times the empirical range on
  more than 90% of tokens, or if bounds cannot be computed within one CPU-hour per 20 tokens.

## Reopening criteria (all five required)

A candidate is developed further only if it:

1. has a coherent, feasible intervention target, with its existence verified, as in `h6_states/`;
2. gives a justified uncertainty statement: a valid derivation, or an explicit "heuristic"
   label with no certificate claimed;
3. improves informative decisions over simpler checks at comparable error control, reported as
   correct, incorrect and inconclusive separately;
4. shows value in a setting where direct execution of the ideal is costly or unavailable;
5. contributes something beyond combining established tools.
