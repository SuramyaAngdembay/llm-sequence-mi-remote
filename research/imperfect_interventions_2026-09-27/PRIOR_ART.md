# Prior-art comparison (read 2026-09-27)

**How the papers were read.** Three reading agents fetched the arXiv full text (HTML, or PDF when
the HTML failed) and read the sections listed below. Figures were seen only through their
captions. I then checked the quotes this comparison relies on against the saved full texts:
- Peña, Vera and Zuluaga §1;
- CIF §5;
- the multiple-mediators paper, Proposition 3.1;
- the SAE-unreliability paper §4;
- SplInterp Theorem B.1;
- the divergence paper §3.1 and Appendix A.3.

Targeted reading does not establish that no prior work exists.

## Comparison

| work | what it guarantees or shows | assumptions and target quantity | relevance here |
|---|---|---|---|
| [Peña, Vera and Zuluaga, Hoffman constants](https://arxiv.org/abs/1905.02894) | dist(u, P) ≤ H · (constraint violation) for mixed systems (Proposition 5), tight in the worst case. For equality-only systems, 1/H is the smallest positive singular value (eqs. 17–18). | H is uniform over right-hand sides and tight only existentially. Computing it is "notoriously difficult" (§1), and the experiments stop at m ≤ 100. No NP-hardness statement. | It supplies our Proposition 2 inequality. For an observed point, the exact projection is computable and tighter. |
| [SplInterp](https://arxiv.org/abs/2505.11836) | TopK SAE regions are polyhedra forming a K-th order power diagram (Theorem 2.3, Theorem B.1). | It studies the encoder partition and reconstruction. The text has no interventions, feasibility, error bounds or downstream model. | It supplies the cell geometry behind our Proposition 1. The full-code fibre and the over-k emptiness are immediate from it. |
| [SAE interventions are unreliable](https://arxiv.org/abs/2606.18322) | Behaviour suppressed by a feature clamp can be recovered while the clamped features stay near their values. A perturbation is found by projected gradient descent (PGD) with feature-preservation constraints (§4). Behaviour can re-emerge through the SAE error term (Appendix J). | Gemma-scale and GPT-2 SAEs, with the activation type not stated. It holds only the selected features, not the full code. It gives no bound, no equivalence test and no TopK treatment, and it does not report how far the defended state re-encodes from the clamp. | It already shows "same selected features, different behaviour" by optimization. That is not available to us as a novelty claim, and it is the lower-bound side of our fibre-wide estimand. |
| [Divergent representations](https://arxiv.org/abs/2511.04638) | For most manifolds, coordinate patching guarantees states outside the natural support (§3.1, A.2). Divergence is harmless if it lies in the behavioural null space (§4). Algorithm 1 is a heuristic harmlessness check that "only approximates harmlessness" (A.3). A counterfactual-latent loss is introduced (§5). | Toy ReLU and affine circuits, and SAE reconstructions of Llama-3-8B. It gives no certificate and no statistics. | It offers different selection rules for an ideal state (a local PCA projection, or an average of natural states). It warns that minimizing divergence does not remove hidden pathways, which matches our counterexamples B and E. |
| [Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) | Hoeffding, anytime-valid and adaptive confidence sequences for the mean of a bounded interventional score (Proposition 1, Theorems 1–2), paired comparisons and multiplicity (§3.5–3.6). It certifies one-sided claims (Algorithm 2). | i.i.d. units, outcomes in [0, 1], the intervention as actually executed. It addresses "the estimation problem … but not the identification problem" (§5). No implementation error, no equivalence testing, no non-unique ideal. | It covers the sampling part of our procedures. A confidence interval or a stopping rule is not a contribution. |
| [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510) | Denoising computes the pure indirect effect, and noising adds the interaction (Proposition 3.1). It gives a second-order form of the interaction (Theorem 3.2) and a Möbius decomposition (Theorem 4.1). It shows empirically that components with near-zero pure indirect effect can still matter. | Heads and MLPs, with SAE latents only mentioned. Its uncertainty is a normal approximation, with no equivalence test and no imperfect interventions. | Our counterexample C and the "restoration versus necessity" split are applications of it. |
| Lead: [Shi et al. 2024](https://arxiv.org/abs/2410.13032) | Equivalence-type circuit tests in which equivalence is the null hypothesis (Eq. 2). | Rejecting the null shows non-equivalence, and retaining it does not certify equivalence (§5). It tests a sign, not an effect size. | Our two one-sided tests at a declared margin run in the correct direction for a negligible-effect claim. |
| Lead: [Makelov, Lange and Nanda 2024](https://arxiv.org/abs/2405.08366) | Compares SAE edits with a ground-truth edit, where the exact edit search is NP-hard (§5.1). | It measures the gap to an ideal edit and does not bound it. | It is the closest "applied versus ideal edit" comparison, and it is empirical. |
| Lead: [Beckers, Eberhardt and Halpern 2019](https://arxiv.org/abs/1906.11583) | Approximate abstraction composes errors with a Lipschitz-type constant (Proposition 5.2). | Deterministic and worst-case, with no statistics and no split between ideal and implemented intervention. | It is structurally like our distance-times-Lipschitz step. |
| Lead: [Hindupur et al.](https://arxiv.org/abs/2503.01822) | TopK is a projection onto the union of k-sparse subspaces (Theorem D.3). | The projection is in preactivation space, not a projection of hidden states onto a code fibre. | Background only. |
| Lead: Stein 2016, Math. Program. 156 (abstract only; the full text is paywalled) | A-priori and a-posteriori feasibility and optimality bounds for rounded LP relaxations. | The problem is mixed-integer LP rounding. | The same pattern of Hoffman distance times sensitivity, in another field. |

## What the proposed method would add, and whether that is open

The proposal was an interval for the effect of a declared ideal feature intervention. It would
widen the applied-effect interval by K × (distance to the ideal) and apply an equivalence margin.

- **Unclaimed in these sources?** Yes. No source combines TopK code feasibility,
  a distance to a declared ideal, a sensitivity bound and an equivalence decision.
- **Open in the sense that matters?** No. Our own analysis and synthetic evaluation show three
  things (`README.md`):
  1. The widened interval is valid but decides only when realization is essentially exact, where
     it adds nothing.
  2. When the ideal state can be computed, executing it directly is simpler, exact and decisive.
  3. The hard remainder is certified regional sensitivity for a transformer, and upper bounds on
     the fibre-wide effect range. Both are verification problems, not a bounded extension.
- **What remains unclaimed but small.** A feasibility-first protocol for feature interventions:
  1. certify whether the declared target code exists, including an over-k count;
  2. execute declared ideal lifts directly, and verify them by re-encoding;
  3. compare two lifts to check identification;
  4. test equivalence at a pre-declared margin;
  5. keep restoration, necessity and interaction as separate estimands.

  Each component is established. Together they could support a methods note or an application
  section, not a new method.
