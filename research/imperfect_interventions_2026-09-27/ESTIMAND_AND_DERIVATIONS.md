# Estimand and derivations

**Status: exploratory research, kept separate from the paper's claims.** The mathematics below is
elementary or established, and each result carries its proof status. Two questions are kept
apart throughout:

- **Q1.** When can an observed near-zero effect of an imperfect feature intervention support a
  negligible effect for a precisely specified ideal intervention?
- **Q2.** Does the component participate in the mechanism?

A certified answer to Q1 says nothing about necessity, redundancy or interactions (Q2).

## 1. The objects

**Units and inputs.** A unit i is one record. In H6 that is a receiver day, clustered by user.
It has two inputs:

- the original input O_i;
- the profile-replaced input R_i. In H6 the DAY and PSY lines come from a foreign partner, and
  the week value and session lines are unchanged.

**Site.** The site is the residual stream at one layer ℓ (hidden state 26 in H6) at a declared
set of positions T_i (the aligned session positions in H6). A state is
x ∈ R^{|T_i| × d} in **raw** model coordinates. Only raw coordinates are consumed by the model.

**Outcome.** F_i(x) is the behaviour-only loss of the R_i forward pass when the site's states are
replaced by x. Everything else is computed from R_i, including the base-model states b_t used by
the delta-SAE. The rescue of a state x is ρ_i(x) = F_i(x_R,i) − F_i(x).

**Evaluation distribution.** This is the declared population of units. H6 uses 30 discovery
users with two attack days and two benign days each. The estimand is the mean over users of the
per-user mean rescue. In the synthetic study it is the mean over the whole pool of cluster means.

**SAE code and coordinates.** Per token,

```
u = (x − b − μ) ⊘ σ,   p(x) = W u + β = R x + g,   c(x) = TopK(ReLU(p(x))),
R = W diag(1/σ),       g = β − W((b + μ) ⊘ σ).
```

- Distances and Lipschitz constants are measured in raw coordinates.
- The same state change has a different length in standardized units whenever σ is not uniform.
  The test suite checks this.

**Targeted and protected coordinates.** S is the selected set; H6 has five features. The
declared target is the full code

```
τ_t,S = c(x_O,t)_S,   τ_t,j = c(x_R,t)_j for every j ∉ S.
```

It restores S from the original and holds every other coordinate at its swapped value. Holding
*every* other coordinate is what "change only these features" means. A weaker protected set
would let the other coordinates move freely, and the feasible set would become a non-convex union
of TopK cells.

## 2. The ideal intervention

Many hidden states share one code, because the encoder reads only the row space of R. So "set the
code to τ" does not define a state. An ideal intervention needs a declared **selection rule**:

| rule | definition | question it answers | limitation |
|---|---|---|---|
| I2, decoder-anchored | x*₂ = Π_P(x_dec), with x_dec = x_R + σ ⊙ D_S(τ_S − c(x_R)_S) the declared decoder edit | What does writing the features along their decoder directions do once the write is minimally corrected so the SAE reads exactly τ? | It inherits the decoder's directions, including any component the encoder cannot see. |
| I1, minimum change | x*₁ = Π_P(x_R) | What does the smallest raw-norm change that makes the SAE read τ do? | It moves along encoder-row directions scaled by the metric, which may not be how the model represents the feature. It depends on the metric (see counterexample E). |
| nearest to applied (A) | Π_P(x̂) | What is the nearest code-exact correction of what was actually applied? | The target moves with the implementation error, so two implementations of the same declared edit have different estimands. It is not a property of the component. |
| fibre-wide | all x ∈ P, or P ∩ ball(x_R, r) | Could *any* state with this code change behaviour by more than τ? | The fibre is unbounded (dimension at least d − k per token). Without a radius, the range is unbounded. With a radius, an upper bound needs certified sensitivity over a large set. |

**Recommendation.**

- Declare I2 as the primary estimand, because it is the natural reading of what H6 set out to
  do.
- Report I1 alongside it. If the two lifts disagree about negligibility, the code alone does not
  identify a "feature restoration effect"; that is procedure P5.
- Rule A answers a question about the implementation, not the component, and is not used.

**Three kinds of error.** Execution and specification error are distinct, and the selection gap
is the quantity a bound must control.

- **Execution error** is ‖x̂ − x_dec‖: the applied state against the declared edit. H6's check
  of applied norm over intended norm (median 1.001) measures this.
- **Specification or realization error** is x_dec ∉ P: the declared edit does not produce
  code τ. H6's normalized target error (q90 1.0) measures this. It is not an execution failure.
- **Selection gap** is ‖x̂ − x*‖, for whichever ideal was declared.

## 3. Results

**Proposition 1 (the fibre of a full-code target is one polyhedron; over-k targets are empty).**

Let S_τ = supp(τ), and fix a margin m > 0. Define

```
P(τ) = { x : R_{S_τ} x = τ_{S_τ} − g_{S_τ},  R_l x + g_l ≤ θ for all l ∉ S_τ },
θ = min τ_{S_τ} − m  if |S_τ| = k,   θ = −m  if |S_τ| < k.
```

Every x ∈ P(τ) satisfies c(x) = τ. Every x with c(x) = τ and margin at least m lies in P(τ). If
|S_τ| > k, no x has c(x) = τ.

*Proof.* On S_τ the code equals the preactivation. Outside it, TopK keeps a feature at zero exactly
when its preactivation is at most the smallest winner's value (with |S_τ| = k), or at most zero
(with |S_τ| < k). The margin makes the ordering strict, so ties cannot occur. A TopK code has at
most k nonzeros. The TopK cells themselves are SplInterp's Theorem B.1 polyhedra.

*Status:* proved, elementary. The test suite checks that projected states carry the target code.

*Corollary (H6).* Every one of 20,000 sampled layer-26 tokens had at least k = 4 positive
preactivations (median 571), so every swapped code has exactly four actives. A target that
restores a selected feature inactive in R, while holding the four winners, has five nonzeros, so
P(τ) is empty. That applies to 173 of the 369 edited tokens, on 23 receivers, which carry 95.3%
of the requested squared change (`results/h6_realizability_*.json`). **For those tokens, H6's
declared restoration had no ideal state at all.** The q90 normalized error of exactly 1.0 is
consistent with rejected insertions, where the requested coordinate stays at zero. Per-token
realized codes were not saved, so this reading is not verified token by token.

**Proposition 2 (distance to a set is not distance to a declared ideal).**

A Hoffman bound controls dist(x̂, P) = ‖x̂ − Π_P(x̂)‖, which is rule A. For any other rule, a
residual at x̂ does not bound the selection gap. For example, if x̂ ∈ P and x̂ ≠ x*₁, the residual
is zero but the gap is not. For I2 the triangle inequality gives

```
‖x̂ − x*₂‖ ≤ ‖x̂ − x_dec‖ + dist(x_dec, P) ≤ ‖x̂ − x_dec‖ + H · r(x_dec).
```

The first term is the measured execution error. The second is a Hoffman bound at the *declared*
state, not the applied one.

*Status:* proved, by the triangle inequality and Hoffman's theorem (Peña, Vera and Zuluaga,
Proposition 5).

**Proposition 3 (the exact projection beats the Hoffman constant).**

Eliminate the equalities (x = x₀ + N z, with N a basis of null(B) and x₀ the equality projection
of y). Then

```
dist(y, P)² = ‖B⁺(d − B y)‖² + ‖z*‖²,
```

where z* solves the least-distance program min ‖z‖ s.t. (A N) z ≤ a − A x₀. If z* = 0, meaning the
equality projection already satisfies the inequalities, then

```
dist(y, P) = ‖B⁺ r_eq‖ ≤ ‖r_eq‖ / σ_min(B),
```

which is the equality-subsystem Hoffman bound. For the full system, H is uniform over right-hand
sides and tight only in the worst case. Its computation is "a notoriously difficult and largely
unexplored computational challenge" (Peña, Vera and Zuluaga, §1); their experiments stay at 100 or
fewer rows, while our layer-26 cell has about 33,000 inequality rows in dimension 4,096. The exact
distance for an observed point is a convex QP with a closed-form fast path.

*Measured:* where the equality bound is valid, it is 1.10 to 1.42 times the exact distance
(development worlds). It is unavailable wherever the inequalities bind, which covers the deletion
and coherent cases.

*Status:* proved, standard. Pointwise, the Hoffman route adds nothing over the projection except a
uniform guarantee for points never observed.

**Proposition 4 (widened equivalence test).**

Suppose F_i is K_i-Lipschitz, in raw Euclidean norm, on a convex set containing x̂_i and x*_i. Let
e_i be the applied rescue and B_i = K_i ‖x̂_i − x*_i‖. Then ρ_i(x*_i) ∈ [e_i − B_i, e_i + B_i],
and the ideal estimand θ* satisfies

```
θ* ∈ [ E(e − B), E(e + B) ].
```

Declare "negligible" when a (1−α) lower confidence bound for E(e − B) exceeds −τ and a (1−α)
upper confidence bound for E(e + B) is below τ. That controls the false-negligible rate for θ* at
α, provided three things hold:

- each K_i is valid;
- users are independent;
- the confidence bounds are valid.

The argument is the intersection–union argument of two one-sided tests, applied to the endpoints
of an identified interval (the same logic as partial-identification confidence intervals).

*Status:* proved, elementary. The synthetic evaluation found no false negligible in any
condition.

**Proposition 5 (when the widened test can decide).**

A "negligible" verdict requires

```
E(K‖x̂ − x*‖) < τ − |E e| − (sampling half-width).
```

With a regional K, that means the applied-to-ideal distance must be at most about τ/K. In the
evaluation worlds τ/K was 0.005 to 0.014 raw units, against ideal moves of 1.3 to 1.6. The applied
state must therefore sit within **0.3% to 0.9% of the edit's length** of the ideal. Measured
distances were 5% under crosstalk and 30% under execution noise
(`results/eval/bound_scale_check.json`). Proposition 4 is valid but almost never informative
unless realization is exact.

*Status:* immediate from Proposition 4, with measured scales.

## 4. Counterexamples and failed approaches

- **A. A sampled Lipschitz constant is not a bound.** Along the segment from x̂ to x*, let
  F(t) = h·sigmoid((t − 1/2)/w). The slope at the observed ends is h·e^{−1/(2w)}/w, which is
  about 10⁻⁹h for w = 0.02, while the change is about h. This is
  `test_sampled_lipschitz_is_not_a_bound`. The synthetic worlds happened to show no coverage
  failure for the sampled version, which does not make it valid.
- **B. Zero code error does not give behavioural fidelity** (many-to-one, C4). Both lifts realize
  the code exactly, yet the decoder-anchored rescue is 0.90 to 0.96 while the minimum-change rescue
  is −0.07 to 0.16. The decoder direction carries a component the encoder cannot see.
- **C. A restoration null does not show non-necessity** (interaction, C5). The I2 rescue is exactly
  zero. Noising the same feature in the clean run raises the loss by 0.20 to 0.22, and the joint
  restoration's interaction contrast is 0.44 to 0.48 across the six evaluation worlds. This is the pure-indirect-effect point of the
  multiple-mediators paper (Proposition 3.1) and the AND-gate example of Heimersheim and Nanda.
- **D. A small realization error can still create a spurious effect** (crosstalk, C1b). A
  mass-weighted realization error of 0.07 lands on a protected feature that the loss reads. The
  applied effect is then called non-negligible in 35% of datasets while the ideal effect is
  negligible. A realization gate cannot see which coordinates the loss reads.
- **E. The minimum-change lift depends on the metric.** In C1a the I2 rescue is exactly 0, but
  I1 ranges from −0.02 to +0.04, above τ in some worlds. A raw-norm-minimal move along W_s/σ is a
  move along W_s/σ² in standardized units, which leaves the subspace the SAE can see. With uniform
  σ (C1h) both lifts give 0.
- **F. An ill-conditioned protected set makes the ideal itself disruptive** (coherent, C1g). When
  the protected row is nearly collinear with the selected row (cosine 0.97), holding it fixed
  while moving the selected feature needs a move of 7.5 to 13 raw units. The ideal rescue of an
  irrelevant feature then ranges from −1.23 to +0.36. The estimand is well defined but has little
  to do with the feature.
- **Failed approach: uniform Hoffman constants.** They are not computable at layer-26 scale, and
  where the equality version applies it is looser than the exact distance.
- **Failed approach: a certified K for the transformer.** None is available for ten
  transformer blocks followed by cross-entropy. Products of spectral norms are vacuous even for the
  32-unit MLP in the synthetic worlds, where K rises about sixfold. Local verification tools do
  not scale to an 8B model.

## 5. Three sources of uncertainty, kept separate

- **Implementation uncertainty** is B_i. It is deterministic given a valid K and an exact
  distance, and it is zero when the ideal is executed directly.
- **Sampling uncertainty** is handled by a cluster-level t interval over users. On development
  worlds with 16 clusters, the worst condition's coverage was 0.79 with a percentile bootstrap and
  0.81 with the t interval, against a nominal 0.90. Thirty clusters, as in H6, raised the worst
  condition to 0.89, and that setting was frozen. On the evaluation worlds, direct execution's
  coverage ranged from 0.88 to 1.00.
- **Interactions or changing context** are not covered by either. The estimand is restoration into
  the R context. Necessity (noising in O) and joint restorations are different estimands, needing
  their own experiments (counterexample C).

## 6. Equivalence margins, declared before evaluation

- **Synthetic study.** τ = 5% of each world's full-restoration rescue (`FROZEN_PROTOCOL.md`).
- **H6, for any future direct-execution check.** τ = 0.012 nats per token, which is 5% of the
  attack-day input effect of 0.243. The margin is scaled by the input effect, not by any
  feature-intervention outcome.
